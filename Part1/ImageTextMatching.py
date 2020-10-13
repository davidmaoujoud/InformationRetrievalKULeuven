from keras.models import Input
from keras.layers import Dense
from keras import backend as K
from keras.models import Model
from keras.layers import dot, concatenate
import BagOfWords
import ImageFeatures
import DataPreprocessing
from keras.optimizers import Adam
import keras
import numpy as np
import pickle


class ImageTextModel:

    def __init__(self, bow: BagOfWords, image_features: ImageFeatures, nb_epochs=2, load_weights=True):
        self.load_weights = load_weights
        self.bow: BagOfWords = bow
        self.image_features: ImageFeatures = image_features
        self.nb_epochs = nb_epochs
        self.training_model, self.image_model, self.text_model = self.create_models()
        if self.load_weights:
            self.load_models_weights()
        self.test_feature_embeddings = []

    def create_models(self):
        anchor_input_image = Input(shape=(2048,))
        anchor_input_text = Input(shape=(self.bow.nb_most_frequent,))

        negative_input_image = Input(shape=(2048,))
        negative_input_text = Input(shape=(self.bow.nb_most_frequent,))

        # Image layers
        #image2 = Dense(300, activation="relu")
        image3 = Dense(256, activation="relu",
                       #kernel_regularizer=regularizers.l2(0.05)
                       )

        # Text layers
        text2 = Dense(256, activation="relu",
                      #kernel_regularizer=regularizers.l2(0.05)
                      )

        # Shared Latent Space
        shared_latent_space = Dense(128, activation="relu")

        # Branches
        # Anchor image branch
        anchor_image_branch = shared_latent_space(image3(anchor_input_image))
        # Anchor text branch
        anchor_text_branch = shared_latent_space(text2(anchor_input_text))
        # Negative image branch
        negative_image_branch = shared_latent_space(image3(negative_input_image))
        # Negative text branch
        negative_text_branch = shared_latent_space(text2(negative_input_text))

        # Dot product as distance metric (normalize = True gives cosine distance, normalize = False gives dot product)
        anchor_distance = dot([anchor_image_branch, anchor_text_branch], axes=1, normalize=True)
        negative_image_distance = dot([anchor_image_branch, negative_text_branch], axes=1, normalize=True)
        negative_text_distance = dot([anchor_text_branch, negative_image_branch], axes=1, normalize=True)

        # Concatenate dot products
        output = concatenate([anchor_distance, negative_image_distance, negative_text_distance])

        training_model = Model(inputs=[anchor_input_image, anchor_input_text, negative_input_image, negative_input_text], outputs=output)
        image_model = Model(inputs=anchor_input_image, outputs=anchor_image_branch)
        text_model = Model(inputs=anchor_input_text, outputs=anchor_text_branch)

        return training_model, image_model, text_model

    def train_model(self):
        adam = Adam(lr=0.001)
        self.training_model.compile(loss=self.bidirectional_ranking_loss, optimizer=adam, metrics=[self.accuracy_image, self.accuracy_text])

        checkpoints = keras.callbacks.ModelCheckpoint("./Models/training_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        data_preprocessing = DataPreprocessing.DataProcessor(self.bow, self.image_features)
        [a_im, a_txt, n_im, n_txt], Y_train, X_val, Y_val = data_preprocessing.return_data()

        for epoch in range(self.nb_epochs):
            np.random.shuffle(n_im)
            np.random.shuffle(n_txt)
            self.training_model.fit([a_im, a_txt, n_im, n_txt], Y_train,
                                    validation_data=[X_val, Y_val],
                                    epochs=1,
                                    batch_size=100,
                                    callbacks=[checkpoints])

        self.image_model.save_weights("./Models/image_model.h5")
        self.text_model.save_weights("./Models/text_model.h5")

    def save_test_image_embeddings(self):
        self.test_feature_embeddings = []
        for f in self.image_features.test_features:
            feature_embedding = self.image_model.predict(np.array([f[1], ]))
            self.test_feature_embeddings.append((f[0], feature_embedding))
        with open('ImageEmbeddings/test.embeddings', 'wb') as file:
            pickle.dump(self.test_feature_embeddings, file)

    def load_test_image_embeddings(self):
        with open('ImageEmbeddings/test.embeddings', 'rb') as file:
            self.test_feature_embeddings = pickle.load(file)

    def return_matching_images(self, query):
        matches = []
        query = self.bow.create_bag_of_word_encoding(query)
        query_embedding = self.text_model.predict(np.array([query, ]))
        for f in self.test_feature_embeddings:
            similarity = self.dot(query_embedding[0], f[1][0])
            matches.append((similarity, f[0]))
        matches = sorted(matches, key=lambda tup: tup[0], reverse=True)[:10]
        print(matches)

        test_sentences = self.bow.matrix_sentences_test
        precision_denominator2 = sum(query)
        average_precision = 0
        if precision_denominator2 > 0:
            for m in matches:
                for test_sentence in test_sentences:
                    if test_sentence[0] == m[1]:
                        word_presence = [sum(x) for x in
                                         zip(test_sentence[1][0], test_sentence[1][1], test_sentence[1][2],
                                             test_sentence[1][3], test_sentence[1][4])]
                        precision_numerator2 = 0
                        for i in range(len(query)):
                            if word_presence[i] > 0 and query[i] > 0:
                                precision_numerator2 += 1
                        precision = precision_numerator2 / precision_denominator2
                        average_precision += precision
            average_precision /= 10
            print("AP = " + str(average_precision))
        else:
            print("AP = 0")
        return matches

    def load_models_weights(self):
        self.training_model.load_weights("./Models/training_model.h5")

    @staticmethod
    def dot(x, y):
        sum = 0
        for i in range(len(x)):
            sum += x[i] * y[i]
        return sum

    @staticmethod
    def bidirectional_ranking_loss(y_true, y_pred):
        anchor_distance = y_pred[:, 0]
        negative_image_distance = y_pred[:, 1]
        negative_text_distance = y_pred[:, 2]
        return K.sum(K.maximum(0., .1 - anchor_distance + negative_image_distance) + K.maximum(0., .1 - anchor_distance + negative_text_distance))

    @staticmethod
    def accuracy_image(y_true, y_pred):
        anchor_distance = y_pred[:, 0]
        negative_image_distance = y_pred[:, 1]
        return K.mean(negative_image_distance < anchor_distance)

    @staticmethod
    def accuracy_text(y_true, y_pred):
        anchor_distance = y_pred[:, 0]
        negative_text_distance = y_pred[:, 2]
        return K.mean(negative_text_distance < anchor_distance)


def main():
    bow = BagOfWords.BagOfWords("./flickr30kentities/")
    bow.load_matrix_sentences_train()
    bow.load_matrix_sentences_val()
    bow.load_matrix_sentences_test()
    image_features = ImageFeatures.ImageFeatures("./flickr30kentities/", "./flickr30kimagefeatures/")
    model = ImageTextModel(bow, image_features)
    model.train_model()


if __name__ == "__main__":
    main()
