import BagOfWords
import ImageFeatures
import random
import numpy as np


class DataProcessor:

    def __init__(self, bow: BagOfWords, features: ImageFeatures):
        self.bow = bow
        self.features = features

        sentences_train = self.bow.matrix_sentences_train
        features_train = self.features.train_features
        sentences_val = self.bow.matrix_sentences_val
        features_val = self.features.val_features

        X_train_images_anchor = []
        X_train_text_anchor = []
        X_train_images_negative = []
        X_train_text_negative = []
        for i in range(len(features_train)):
            f = features_train[i]
            s = sentences_train[i]
            random_feature_position = random.randint(0, len(features_train)-1)
            random_sentence_position = random.randint(0, 4)
            for j in range(5):
                X_train_images_anchor.append(f[1])
                X_train_text_anchor.append(s[1][j])
                X_train_images_negative.append(features_train[random_feature_position][1])
                X_train_text_negative.append(sentences_train[random_feature_position][1][random_sentence_position])

        self.X_train = [X_train_images_anchor, X_train_text_anchor, X_train_images_negative, X_train_text_negative]
        self.Y_train = np.zeros((len(X_train_images_anchor), 1))

        X_val_images_anchor = []
        X_val_text_anchor = []
        X_val_images_negative = []
        X_val_text_negative = []
        for i in range(len(features_val)):
            f = features_val[i]
            s = sentences_val[i]
            random_feature_position = random.randint(0, len(features_val) - 1)
            random_sentence_position = random.randint(0, 4)
            for j in range(5):
                X_val_images_anchor.append(f[1])
                X_val_text_anchor.append(s[1][j])
                X_val_images_negative.append(features_val[random_feature_position][1])
                X_val_text_negative.append(sentences_val[random_feature_position][1][random_sentence_position])

        self.X_val = [X_val_images_anchor, X_val_text_anchor, X_val_images_negative, X_val_text_negative]
        self.Y_val = np.zeros((len(X_val_images_anchor), 1))

    def return_data(self):
        return self.X_train, self.Y_train, self.X_val, self.Y_val

    def return_test_data(self):
        return self.X_test, self.Y_test


def main():
    bow = BagOfWords.BagOfWords("./flickr30kentities/")
    bow.load_matrix_sentences_train()
    bow.load_matrix_sentences_val()
    bow.load_matrix_sentences_test()
    features = ImageFeatures.ImageFeatures("./flickr30kentities/", "./flickr30kimagefeatures/")
    data_processor = DataProcessor(bow, features)


if __name__ == "__main__":
    main()
