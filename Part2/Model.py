import tensorflow as tf
import BagOfWords
import ImageFeatures
import DataPreprocessing
import numpy as np
import pickle
from scipy.spatial.distance import hamming
import sys
np.set_printoptions(threshold=sys.maxsize)


class ImageTextModel:

    def __init__(self, bow: BagOfWords, image_features: ImageFeatures, nb_epochs=1, load_weights=True):
        self.F = None
        self.G = None
        self.B = None
        self.load_weights = load_weights
        self.bow: BagOfWords = bow
        self.image_features: ImageFeatures = image_features
        self.nb_epochs = nb_epochs
        self.c = 32
        self.test_feature_embeddings = []
        self.batch_size = 128
        self.alpha = 1
        self.gamma = 1
        self.eta = 1

        self.optimizer = tf.optimizers.SGD(lr=5e-1)

        self.image_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(2048,)),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(self.c)
        ])

        self.caption_model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.bow.nb_most_frequent,)),
            tf.keras.layers.Dense(8192),
            tf.keras.layers.Dense(self.c),
        ])

        if self.load_weights:
            self.image_model.load_weights("Models/image_model.h5")
            self.caption_model.load_weights("Models/caption_model.h5")

        self.number_of_training_examples = 0
        self.Ones_batch = 0
        self.Ones_remaining = 0

    def train_model(self):
        data_preprocessing = DataPreprocessing.DataProcessor(self.bow, self.image_features, which_sentence=0)
        images_train, _, captions_train, _, S_train = data_preprocessing.return_train_data()
        self.number_of_training_examples = len(S_train)
        self.Ones_batch = tf.constant(np.ones([self.batch_size, 1], 'float32'))
        self.Ones_remaining = tf.constant(np.ones([self.number_of_training_examples - self.batch_size, 1], 'float32'))

        self.F = np.random.randn(self.c, self.number_of_training_examples).astype("float32")
        self.G = np.random.randn(self.c, self.number_of_training_examples).astype("float32")
        self.F = np.transpose(np.array(self.image_model(np.array(images_train, dtype="float32"))))
        self.G = np.transpose(np.array(self.caption_model(np.array(captions_train))))
        self.B = np.sign(self.F + self.G)

        images_train = np.array(images_train,  dtype="float32")
        captions_train = np.array(captions_train, dtype="float32")
        print("Training has started... ")

        for epoch in range(self.nb_epochs):

            self.train_image_model(images_train, S_train)
            self.train_caption_model(captions_train, S_train)
            self.update_B()
            loss = self.loss_function(self.B, self.F, self.G, S_train)

            print("Epoch: " + str(epoch+1) + ", loss: " + str(loss))
            self.image_model.save_weights("Models/image_model.h5")
            self.caption_model.save_weights("Models/caption_model.h5")

    def train_image_model(self, images_train, S_train):

        for iteration in range(int(self.number_of_training_examples / self.batch_size)):
            random_image_indices = np.random.permutation(self.number_of_training_examples)[:self.batch_size]
            remaining_indices = np.setdiff1d(range(self.number_of_training_examples), random_image_indices)
            input_images = images_train[random_image_indices,:]
            B_batch = self.B[:, random_image_indices]
            S_batch = S_train[random_image_indices, :]

            with tf.GradientTape() as tape:
                F_updated = tf.transpose(self.image_model(input_images))
                self.F[:, random_image_indices] = F_updated
                F_remaining = self.F[:, remaining_indices]

                # Depending on the loss minimization technique used (see below), switch to appropriate loss function
                L = self.image_loss_function(self.G, F_remaining, B_batch, S_batch, F_updated)

                # One of 2 techniques can be used:
                # 1) Minimizing the loss function J directly (uncomment the first line)
                # 2) Computing dJ/dF, and then computing dJ/dTheta_x using the chain rule (uncomment the second line)
                # Note: the appropriate loss function needs to be selected
                #gradients = tape.gradient(L, self.image_model.trainable_variables)
                gradients = tape.gradient(F_updated, self.image_model.trainable_variables, output_gradients=L)
                #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            self.optimizer.apply_gradients(zip(gradients, self.image_model.trainable_variables))

    def train_caption_model(self, captions_train, S_train):
        for iteration in range(int(self.number_of_training_examples / self.batch_size)):
            random_caption_indices = np.random.permutation(self.number_of_training_examples)[:self.batch_size]
            remaining_indices = np.setdiff1d(range(self.number_of_training_examples), random_caption_indices)
            input_captions = captions_train[random_caption_indices, :]
            B_batch = self.B[:, random_caption_indices]
            S_batch = S_train[:, random_caption_indices]
            with tf.GradientTape() as tape:
                G_updated = tf.transpose(self.caption_model(input_captions))
                self.G[:, random_caption_indices] = G_updated
                G_remaining = self.G[:, remaining_indices]

                # Depending on the loss minimization technique used (see below), switch to appropriate loss function
                L = self.caption_loss_function(self.F, G_remaining, B_batch, S_batch, G_updated)

                # One of 2 techniques can be used:
                # 1) Minimizing the loss function J directly (uncomment the first line)
                # 2) Computing dJ/dG, and then computing dJ/dTheta_y using the chain rule (uncomment the second line)
                # Note: the appropriate loss function needs to be selected
                #gradients = tape.gradient(L, self.caption_model.trainable_variables)
                gradients = tape.gradient(G_updated, self.caption_model.trainable_variables, output_gradients=L)
                #gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
            self.optimizer.apply_gradients(zip(gradients, self.caption_model.trainable_variables))

    def image_loss_function(self, G, F_remaining, B_batch, S_batch, F_updated):
        Ones = tf.constant(np.ones([self.number_of_training_examples, 1], 'float32'))
        theta = 0.5 * tf.matmul(tf.transpose(G), F_updated)
        sigma = tf.divide(1.0, 1.0 + tf.math.exp(-theta))
        F1 = tf.matmul(self.F, Ones)
        cross_modal_loss = self.alpha * 0.5 * tf.matmul(G,(sigma - tf.transpose(np.array(S_batch, dtype="float32"))))
        cross_modal_hashing_loss = 2 * self.gamma * (F_updated - B_batch)
        repF1 = tf.tile(F1, tf.constant([1, self.batch_size], tf.int32))
        balance_loss = 2 * self.eta * repF1
        dJdFb = cross_modal_loss + cross_modal_hashing_loss + balance_loss
        loss = tf.divide(dJdFb, float(self.number_of_training_examples * self.batch_size))

        return loss

    def caption_loss_function(self, F, G_remaining, B_batch, S_batch, G_updated):
        Ones = tf.constant(np.ones([self.number_of_training_examples, 1], 'float32'))
        theta = 0.5 * tf.matmul(tf.transpose(F), G_updated)
        sigma = tf.divide(1.0, 1.0 + tf.math.exp(-theta))
        G1 = tf.matmul(self.G, Ones)
        cross_modal_loss = self.alpha * 0.5 * tf.matmul(F, (sigma - tf.constant(np.array(S_batch, dtype="float32"))))
        cross_modal_hashing_loss = 2 * self.gamma * (G_updated - B_batch)
        repG1 = tf.tile(G1, tf.constant([1, self.batch_size], tf.int32))
        balance_loss = self.eta * repG1
        dJdGb = cross_modal_loss + cross_modal_hashing_loss + balance_loss
        loss = tf.divide(dJdGb, float(self.number_of_training_examples * self.batch_size))
        return loss

    def image_loss_function_second_method(self, G, F_remaining, B_batch, S_batch, F_updated):
        theta = 1.0 / 2 * tf.matmul(tf.transpose(F_updated), G)
        cross_modal_loss = - self.alpha * tf.reduce_sum(tf.multiply(S_batch, theta) - tf.math.log(1.0 + tf.math.exp(theta)))
        cross_modal_hashing_loss = self.gamma * tf.reduce_sum(tf.pow((B_batch - F_updated), 2))
        balance_loss = self.eta * tf.reduce_sum(tf.pow(tf.matmul(F_updated, self.Ones_batch) + tf.matmul(F_remaining, self.Ones_remaining), 2))
        loss = tf.divide(cross_modal_loss + cross_modal_hashing_loss + balance_loss, float(self.number_of_training_examples * self.batch_size))
        return loss

    def caption_loss_function_second_method(self, F, G_remaining, B_batch, S_batch, G_updated):
        theta = 1.0 / 2 * tf.matmul(tf.transpose(F), G_updated)
        cross_modal_loss = - self.alpha * tf.reduce_sum(tf.multiply(S_batch, theta) - tf.math.log(1.0 + tf.math.exp(theta)))
        cross_modal_hashing_loss = self.gamma * tf.reduce_sum(tf.pow((B_batch - G_updated), 2))
        balance_loss = self.eta * tf.reduce_sum(tf.pow(tf.matmul(G_updated, self.Ones_batch) + tf.matmul(G_remaining, self.Ones_remaining), 2))
        loss = tf.divide(cross_modal_loss + cross_modal_hashing_loss + balance_loss, float(self.number_of_training_examples * self.batch_size))
        return loss

    def loss_function(self, B, F, G, S):
        theta = np.matmul(np.transpose(F), G)/2
        cross_modal_loss = self.alpha * np.sum(np.log(1+np.exp(theta)) - S * theta)
        cross_modal_hashing_loss = self.gamma * np.sum(np.power((B-F), 2) + np.power(B-G,2))
        balance_loss = self.eta * ( np.sum(np.power(np.matmul(F, np.ones((F.shape[1],1))),2)) + np.sum(np.power(np.matmul(G, np.ones((F.shape[1],1))),2)) )
        loss = cross_modal_loss + cross_modal_hashing_loss + balance_loss
        return loss

    def update_B(self):
        self.B = np.sign(self.F + self.G)

    def load_models_weights(self):
        self.image_model.load_weights("Models/image_model.h5")
        self.caption_model.load_weights("Models/caption_model.h5")

    def save_test_image_embeddings(self):
        self.test_feature_embeddings = []
        for f in self.image_features.test_features:
            feature_embedding = np.sign(self.image_model(np.array([f[1], ], dtype="float32")))
            self.test_feature_embeddings.append((f[0], feature_embedding))
        with open('ImageEmbeddings/test.embeddings', 'wb') as file:
            pickle.dump(self.test_feature_embeddings, file)

    def load_test_image_embeddings(self):
        with open('ImageEmbeddings/test.embeddings', 'rb') as file:
            self.test_feature_embeddings = pickle.load(file)

    @staticmethod
    def dot(x, y):
        sum = 0
        for i in range(len(x)):
            sum += x[i] * y[i]
        return sum

    def return_matching_images(self, query):
        matches = []
        query = self.bow.create_bag_of_word_encoding(query)
        query_embedding = np.sign(self.caption_model(np.array([query, ])))

        test_sentences = self.bow.matrix_sentences_test

        for f in self.test_feature_embeddings:
            distance = hamming(query_embedding[0], f[1][0])
            matches.append((distance, f[0]))
        #print(sorted(matches, key=lambda tup: tup[0], reverse=False))
        matches = sorted(matches, key=lambda tup: tup[0], reverse=False)[:10]

        precision_denominator2 = sum(query)
        average_precision = 0
        if precision_denominator2 > 0:
            for m in matches:
                for test_sentence in test_sentences:
                    if test_sentence[0] == m[1]:
                        word_presence = [sum(x) for x in zip(test_sentence[1][0], test_sentence[1][1], test_sentence[1][2], test_sentence[1][3], test_sentence[1][4])]
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


def main():
    bow = BagOfWords.BagOfWords("./flickr30kentities/")
    bow.load_matrix_sentences_train()
    bow.load_matrix_sentences_val()
    bow.load_matrix_sentences_test()
    image_features = ImageFeatures.ImageFeatures("./flickr30kentities/", "./flickr30kimagefeatures/", nb_train=1000, nb_val=1000, nb_test=1000)
    image_features.load_image_features()
    model = ImageTextModel(bow, image_features, load_weights=True, nb_epochs=50)
    model.train_model()


if __name__ == "__main__":
    main()
