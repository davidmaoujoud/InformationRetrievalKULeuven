from BagOfWords import BagOfWords
from ImageFeatures import ImageFeatures
from ImageTextMatching import ImageTextModel
import numpy as np


class PerformanceTester:

    def __init__(self, entities_path, model: ImageTextModel):
        self.model = model
        self.bow: BagOfWords = self.model.bow
        self.image_features: ImageFeatures = self.model.image_features
        self.entities_path = entities_path
        self.nb_queries = 100

    def set_nb_queries(self, nb_queries):
        self.nb_queries = nb_queries

    def compute_map_10(self):
        queries = self.bow.matrix_sentences_test
        mean_average_precision = 0

        concatenated_captions_and_features = []
        for k in range(len(queries)):
            concatenated_captions_and_features.append((queries[k][0], queries[k][1], self.image_features.test_features[k][1]))

        nb_queries = 0
        for k in range(len(queries)):
            for l in range(5):
                if nb_queries >= self.nb_queries:
                    break
                nb_queries += 1

                print(">>> Calculating the average precision (AP) for query number " + str(nb_queries))
                query = queries[k][1][l]
                query_embedding = self.model.text_model.predict(np.array([query, ]))

                features_score = []
                for f in concatenated_captions_and_features:
                    feature_embedding = self.model.image_model.predict(np.array([f[2], ]))
                    similarity = self.dot(query_embedding[0], feature_embedding[0])
                    features_score.append((similarity, f[0], f[1]))
                features_score = sorted(features_score, key=lambda tup: tup[0], reverse=True)[:10]
                precision_denominator2 = sum(query)
                average_precision = 0
                if precision_denominator2 > 0:
                    for f_s in features_score:
                        word_presence = [sum(x) for x in zip(f_s[2][0], f_s[2][1], f_s[2][2], f_s[2][3], f_s[2][4])]
                        precision_numerator2 = 0
                        for i in range(len(query)):
                            if word_presence[i] > 0 and query[i] > 0:
                                precision_numerator2 += 1
                        precision = precision_numerator2 / precision_denominator2
                        average_precision += precision
                average_precision /= 10
                mean_average_precision += average_precision

                print("AP = " + str(average_precision))

        mean_average_precision /= nb_queries
        print(">>> The final MAP@10 score is: ")
        print("MAP = " + str(mean_average_precision))

    @staticmethod
    def dot(x, y):
        sum = 0
        for i in range(len(x)):
            sum += x[i]*y[i]
        return sum


def main():
    bow = BagOfWords("./flickr30kentities/")
    bow.load_matrix_sentences_train()
    bow.load_matrix_sentences_val()
    bow.load_matrix_sentences_test()
    features = ImageFeatures("./flickr30kentities/", "./flickr30kimagefeatures/")
    model = ImageTextModel(bow, features, load_weights=False)
    performance_tester = PerformanceTester("./flickr30kentities/", model)
    performance_tester.compute_map_10()


if __name__ == "__main__":
    main()
