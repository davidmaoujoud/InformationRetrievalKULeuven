from BagOfWords import BagOfWords
from ImageFeatures import ImageFeatures
from Model import ImageTextModel
from PerformanceTester import PerformanceTester
import glob
import shutil
import os
import sys

file_path_to_flickr30k_images = "C:/Users/david/OneDrive/Documents/University/KULeuven/flickr30k-images/"

file_path_to_flickr30k_image_features = "C:/Users/david/OneDrive/Documents/University/KULeuven/flickr30kimagefeatures/"

file_path_to_flickr30k_entities = "C:/Users/david/OneDrive/Documents/University/KULeuven/flickr30kentities/"


class ImageRetrievalEngine:

    def __init__(self, nb_train=10000, nb_val=1000, nb_test=1000):
        self.bow = BagOfWords(file_path_to_flickr30k_entities, nb_train=nb_train, nb_val=nb_val, nb_test=nb_test, nb_most_frequent=1000)
        self.image_features = ImageFeatures(file_path_to_flickr30k_entities, file_path_to_flickr30k_image_features, nb_train=nb_train, nb_val=nb_val, nb_test=nb_test)
        self.model = ImageTextModel(self.bow, self.image_features, nb_epochs=500, load_weights=True)
        self.performance_tester = PerformanceTester(file_path_to_flickr30k_entities, self.model)

    def train_model(self):
        self.model.train_model()

    def compute_map_10(self):
        self.performance_tester.compute_map_10()

    def sort_txt_files_alphabetically(self):
        self.bow.sort_train_txt()
        self.bow.sort_val_txt()
        self.bow.sort_test_txt()

    def create_encoded_sentences(self):
        self.bow.create_word_frequency_dictionary()
        self.bow.truncate_word_frequency_dictionary()
        self.bow.create_matrix_sentences_train()
        self.bow.save_matrix_sentences_train()
        self.bow.create_val_sentences()
        self.bow.save_matrix_sentences_val()
        self.bow.create_test_sentences()
        self.bow.save_matrix_sentences_test()

    def load_encoded_sentences(self):
        self.bow.load_matrix_sentences_train()
        self.bow.load_matrix_sentences_val()
        self.bow.load_matrix_sentences_test()
        self.bow.load_word_frequency_dictionary()

    def separate_image_features_into_train_val_test(self):
        self.image_features.create_feature_files()

    def load_model_weights(self):
        self.model.load_models_weights()

    def save_test_image_embeddings(self):
        self.model.save_test_image_embeddings()

    def load_test_image_embeddings(self):
        self.model.load_test_image_embeddings()

    def retrieve_matching_images(self, query):
        images = self.model.return_matching_images(query)
        print(images)
        files = glob.glob('./RetrievedImages/*')
        for f in files:
            os.remove(f)
        for file in images:
            shutil.copy(file_path_to_flickr30k_images + file[1] + ".jpg", "./RetrievedImages")
        return images

    def load_image_features(self):
        self.image_features.load_image_features()

    def set_nb_queries_for_map10(self, nb_queries):
        self.performance_tester.set_nb_queries(nb_queries)


def main():
    image_retrieval_engine = ImageRetrievalEngine()

    if len(sys.argv) == 1:
        "An argument is missing."
        return -1

    if sys.argv[1] == "-setup":
        print("This may take a while...")
        print("Sorting .txt files...")
        image_retrieval_engine.sort_txt_files_alphabetically()
        print("Encoding sentences...")
        image_retrieval_engine.create_encoded_sentences()
        print("Separating the .csv file (image features) into train, val, and test files...")
        image_retrieval_engine.separate_image_features_into_train_val_test()
        print("Done!")

    if sys.argv[1] == "-embed":
        print("Creating the embedded test images...")
        image_retrieval_engine.load_model_weights()
        image_retrieval_engine.load_encoded_sentences()
        image_retrieval_engine.load_image_features()
        image_retrieval_engine.save_test_image_embeddings()

    if sys.argv[1] == "-train":
        print("Training the model...")
        image_retrieval_engine.load_encoded_sentences()
        image_retrieval_engine.load_image_features()
        image_retrieval_engine.train_model()

    if sys.argv[1] == "-map10":
        if len(sys.argv) == 2:
            nb_queries = 100
        else:
            nb_queries = int(sys.argv[2])
        print("Computing the MAP@10 score on " + str(nb_queries) + " queries...")
        image_retrieval_engine.set_nb_queries_for_map10(nb_queries)
        image_retrieval_engine.load_encoded_sentences()
        image_retrieval_engine.load_image_features()
        image_retrieval_engine.compute_map_10()

    if sys.argv[1] == "-predict":
        image_retrieval_engine.load_model_weights()
        image_retrieval_engine.load_encoded_sentences()
        image_retrieval_engine.load_image_features()
        image_retrieval_engine.load_test_image_embeddings()
        while True:
            print(">>> Your caption: ")
            caption = input()
            if caption == ":q":
                break
            else:
                image_retrieval_engine.retrieve_matching_images(caption)


if __name__ == "__main__":
    main()
