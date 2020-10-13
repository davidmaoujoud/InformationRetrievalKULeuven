import BagOfWords
import ImageFeatures
import numpy as np
import sys


class DataProcessor:

    def __init__(self, bow: BagOfWords, features: ImageFeatures, which_sentence=0):
        self.bow = bow
        self.features = features
        self.number = 0
        self.which_sentence = which_sentence

        self.images_train = [im[1] for im in self.features.train_features]
        self.Y_images_train = [im[0] for im in self.features.train_features]
        self.captions_train0 = [cap[1][0] for cap in self.bow.matrix_sentences_train]
        self.captions_train1 = [cap[1][1] for cap in self.bow.matrix_sentences_train]
        self.captions_train2 = [cap[1][2] for cap in self.bow.matrix_sentences_train]
        self.captions_train3 = [cap[1][3] for cap in self.bow.matrix_sentences_train]
        self.captions_train4 = [cap[1][4] for cap in self.bow.matrix_sentences_train]

        self.Y_captions_train = [cap[0] for cap in self.bow.matrix_sentences_train]
        self.S_train = np.zeros((len(self.images_train), len(self.captions_train0)), dtype="int")

        for i in range(len(self.images_train)):
            self.S_train[i][i] = 1

        #print(self.Y_images_train)
        #print(self.Y_captions_train)

        self.images_val = [im[1] for i in range(5) for im in self.features.val_features]
        self.Y_images_val = np.zeros((len(self.images_val), 1))
        self.captions_val = [cap[1][j] for j in range(5) for cap in self.bow.matrix_sentences_val]
        self.Y_captions_val = np.zeros((len(self.captions_val), 1))
        self.S_val = np.zeros((len(self.images_val), len(self.images_val)))

        for i in range(0,len(self.images_val),5):
            for j in range(5):
                self.S_val[i][i + j] = 1
                self.S_val[i + 1][i + j] = 1
                self.S_val[i + 2][i + j] = 1
                self.S_val[i + 3][i + j] = 1
                self.S_val[i + 4][i + j] = 1

    def return_train_data(self):
        return self.images_train, self.Y_images_train, self.captions_train0, self.Y_captions_train, self.S_train

    def return_next_batch(self):
        if self.number == 0:
            self.number += 1
            print("Using batch number 1")
            return self.captions_train1
        elif self.number == 1:
            self.number += 1
            print("Using batch number 2")
            return self.captions_train2
        elif self.number == 2:
            self.number += 1
            print("Using batch number 3")
            return self.captions_train3
        elif self.number == 3:
            self.number += 1
            print("Using batch number 4")
            return self.captions_train4
        elif self.number == 4:
            self.number = 0
            print("Using batch number 0")
            return self.captions_train0



    def return_val_data(self):
        return self.images_val, self.Y_images_val, self.captions_val, self.Y_captions_val, self.S_val

    def return_test_data(self):
        return self.X_test, self.Y_test


def main():
    bow = BagOfWords.BagOfWords("./flickr30kentities/")
    bow.load_matrix_sentences_train()
    bow.load_matrix_sentences_val()
    bow.load_matrix_sentences_test()
    features = ImageFeatures.ImageFeatures("./flickr30kentities/", "./flickr30kimagefeatures/", nb_train=100, nb_val=100, nb_test=100)
    features.load_image_features()
    data_processor = DataProcessor(bow, features)


if __name__ == "__main__":
    main()
