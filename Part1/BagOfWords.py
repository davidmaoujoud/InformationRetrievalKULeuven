import csv
import flickr30k_entities_utils as f30k
import nltk
import heapq
import re
import numpy as np
import pickle


class BagOfWords:

    def __init__(self, entities_path, nb_train=15000, nb_val=1000, nb_test=1000, nb_most_frequent=1000):
        self.words_and_frequencies = {}
        self.nb_train = nb_train
        self.nb_val = nb_val
        self.nb_test = nb_test

        self.train_labels_file = entities_path + "train.txt"
        self.sorted_train_labels_file = "SortedTxtFiles/train_sorted.txt"  # Alphabetically

        self.val_labels_file = entities_path + "val.txt"
        self.sorted_val_labels_file = "SortedTxtFiles/val_sorted.txt"  # Alphabetically

        self.test_labels_file = entities_path + "test.txt"
        self.sorted_test_labels_file = "SortedTxtFiles/test_sorted.txt"  # Alphabetically

        self.sentences_path = entities_path + "Sentences/"

        self.words_and_frequencies_file = "WordsAndFrequencies/words_and_frequencies.csv"
        self.sentences_train = []
        self.matrix_sentences_train = []
        self.matrix_sentences_val = []
        self.matrix_sentences_test = []

        self.most_frequent_words = []
        self.nb_most_frequent = nb_most_frequent

    def create_word_frequency_dictionary(self):
        train_names = open(self.sorted_train_labels_file)
        sentences = []
        for index, file_name in enumerate(train_names.readlines()):
            if index == self.nb_train:
                break
            file_name = file_name.split("\n")[0] + ".txt"
            for i in range(5):
                sentences.append((file_name, f30k.get_sentence_data(self.sentences_path + file_name)[i].get("sentence")))
        for (n,s) in sentences:
            s = s.lower()
            s = re.sub(r'\W', ' ', s)
            s = re.sub(r'\s+', ' ', s)
            self.sentences_train.append((n, s))
            tokens = nltk.word_tokenize(s)
            for t in tokens:
                if t not in self.words_and_frequencies.keys():
                    self.words_and_frequencies[t] = 1
                else:
                    self.words_and_frequencies[t] += 1

    def create_val_sentences(self):
        val_names = open(self.sorted_val_labels_file)
        for index, file_name in enumerate(val_names.readlines()):
            if index == self.nb_val:
                break
            matrix = []
            for i in range(5):
                s = f30k.get_sentence_data(self.sentences_path + file_name.split("\n")[0] + ".txt")[i].get("sentence")
                s = s.lower()
                s = re.sub(r'\W', ' ', s)
                s = re.sub(r'\s+', ' ', s)

                tokens = nltk.word_tokenize(s)
                s_vector = []
                for t in self.most_frequent_words:
                    if t in tokens:
                        s_vector.append(1)
                    else:
                        s_vector.append(0)
                matrix.append(s_vector)
            matrix = np.asarray(matrix)
            self.matrix_sentences_val.append((file_name.split("\n")[0], matrix))

    def create_test_sentences(self):
        test_names = open(self.sorted_test_labels_file)
        for index, file_name in enumerate(test_names.readlines()):
            if index == self.nb_test:
                break
            matrix = []
            for i in range(5):
                s = f30k.get_sentence_data(self.sentences_path + file_name.split("\n")[0] + ".txt")[i].get("sentence")
                s = s.lower()
                s = re.sub(r'\W', ' ', s)
                s = re.sub(r'\s+', ' ', s)

                tokens = nltk.word_tokenize(s)
                s_vector = []
                for t in self.most_frequent_words:
                    if t in tokens:
                        s_vector.append(1)
                    else:
                        s_vector.append(0)
                matrix.append(s_vector)
            matrix = np.asarray(matrix)
            self.matrix_sentences_test.append((file_name.split("\n")[0], matrix))

    def create_bag_of_word_encoding(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub(r'\W', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        tokens = nltk.word_tokenize(sentence)
        s_vector = []
        for t in self.most_frequent_words:
            if t in tokens:
                s_vector.append(1)
            else:
                s_vector.append(0)
        return s_vector

    def create_matrix_sentences_train(self):
        for i in range(0, len(self.sentences_train), 5):
            matrix = []
            for j in range(5):
                tokens = nltk.word_tokenize(self.sentences_train[i + j][1])
                sentence_vector = []
                for t in self.most_frequent_words:
                    if t in tokens:
                        sentence_vector.append(1)
                    else:
                        sentence_vector.append(0)
                matrix.append(sentence_vector)
            matrix = np.asarray(matrix)
            self.matrix_sentences_train.append((self.sentences_train[i][0][:len(self.sentences_train[i][0]) - 4], matrix))

    def save_word_frequency_dictionary(self):
        writer = csv.writer(open(self.words_and_frequencies_file, "w"))
        for key, value in self.words_and_frequencies.items():
            writer.writerow([key, value])

    def load_word_frequency_dictionary(self):
        reader = csv.reader(open(self.words_and_frequencies_file, "r"))
        for row in reader:
            if len(row) > 0:
                self.words_and_frequencies[row[0]] = int(row[1])
        self.truncate_word_frequency_dictionary()

    def truncate_word_frequency_dictionary(self):
        self.most_frequent_words = heapq.nlargest(self.nb_most_frequent, self.words_and_frequencies, key=self.words_and_frequencies.get)

    def save_matrix_sentences_train(self):
        with open('EncodedSentences/matrix.sentences.train', 'wb') as file:
            pickle.dump(self.matrix_sentences_train, file)

    def load_matrix_sentences_train(self):
        with open('EncodedSentences/matrix.sentences.train', 'rb') as file:
            self.matrix_sentences_train = pickle.load(file)

    def save_matrix_sentences_val(self):
        with open('EncodedSentences/matrix.sentences.val', 'wb') as file:
            pickle.dump(self.matrix_sentences_val, file)

    def load_matrix_sentences_val(self):
        with open('EncodedSentences/matrix.sentences.val', 'rb') as file:
            self.matrix_sentences_val = pickle.load(file)

    def save_matrix_sentences_test(self):
        with open('EncodedSentences/matrix.sentences.test', 'wb') as file:
            pickle.dump(self.matrix_sentences_test, file)

    def load_matrix_sentences_test(self):
        with open('EncodedSentences/matrix.sentences.test', 'rb') as file:
            self.matrix_sentences_test = pickle.load(file)

    def sort_train_txt(self):
        sort_input_file = open(self.train_labels_file, "r")
        sorted_file = open(self.sorted_train_labels_file, "w")
        lines = sort_input_file.readlines()
        lines.sort()
        for l in lines:
            sorted_file.write(l)
        sort_input_file.close()
        sorted_file.close()

    def sort_val_txt(self):
        sort_input_file = open(self.val_labels_file, "r")
        sorted_file = open(self.sorted_val_labels_file, "w")
        lines = sort_input_file.readlines()
        lines.sort()
        for l in lines:
            sorted_file.write(l)
        sort_input_file.close()
        sorted_file.close()

    def sort_test_txt(self):
        sort_input_file = open(self.test_labels_file, "r")
        sorted_file = open(self.sorted_test_labels_file, "w")
        lines = sort_input_file.readlines()
        lines.sort()
        for l in lines:
            sorted_file.write(l)
        sort_input_file.close()
        sorted_file.close()


def main():
    bow = BagOfWords("./flickr30kentities/")
    bow.sort_train_txt()
    bow.sort_val_txt()
    bow.sort_test_txt()
    bow.create_word_frequency_dictionary()
    bow.save_word_frequency_dictionary()
    bow.load_word_frequency_dictionary()
    bow.truncate_word_frequency_dictionary()
    bow.create_matrix_sentences_train()
    bow.save_matrix_sentences_train()
    bow.load_matrix_sentences_train()
    bow.create_val_sentences()
    bow.save_matrix_sentences_val()
    bow.load_matrix_sentences_val()
    bow.create_test_sentences()
    bow.save_matrix_sentences_test()
    bow.load_matrix_sentences_test()


if __name__ == "__main__":
    main()
