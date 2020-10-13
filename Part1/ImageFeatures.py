import csv


class ImageFeatures:
    def __init__(self, entities_path, image_feature_path, nb_train=15000, nb_val=1000, nb_test=1000):
        self.entities_path = entities_path
        self.image_feature_path = image_feature_path
        self.image_features = []
        self.train_features = []
        self.val_features = []
        self.test_features = []
        self.nb_train = nb_train
        self.nb_val = nb_val
        self.nb_test = nb_test


    def load_image_features(self):
        self.load_train_file()
        self.load_val_file()
        self.load_test_file()

    def load_train_file(self):
        feature_reader = csv.reader(open('ImageFeatureFiles/image_features_train.csv', newline=''), delimiter=' ', quotechar='|')
        for index, row in enumerate(feature_reader):
            if index == self.nb_train * 2:
                break
            if len(row) > 0:
                self.train_features.append((row[0].split(".")[0], row[1:]))

    def load_val_file(self):
        feature_reader = csv.reader(open('ImageFeatureFiles/image_features_val.csv', newline=''), delimiter=' ', quotechar='|')
        for index, row in enumerate(feature_reader):
            if index == self.nb_val * 2:
                break
            if len(row) > 0:
                self.val_features.append((row[0].split(".")[0], row[1:]))

    def load_test_file(self):
        feature_reader = csv.reader(open('ImageFeatureFiles/image_features_test.csv', newline=''), delimiter=' ', quotechar='|')
        for index, row in enumerate(feature_reader):
            if index == self.nb_test * 2:
                break
            if len(row) > 0:
                self.test_features.append((row[0].split(".")[0], row[1:]))

    def create_feature_files(self):
        train_file = open(self.entities_path + "train.txt")
        train_lines = train_file.readlines()

        val_file = open(self.entities_path + "val.txt")
        val_lines = val_file.readlines()

        test_file = open(self.entities_path + "test.txt")
        test_lines = test_file.readlines()

        train_writer = csv.writer(open("ImageFeatureFiles/image_features_train.csv", "w"), delimiter=' ')
        val_writer = csv.writer(open("ImageFeatureFiles/image_features_val.csv", "w"), delimiter=' ')
        test_writer = csv.writer(open("ImageFeatureFiles/image_features_test.csv", "w"), delimiter=' ')

        feature_reader = csv.reader(open(self.image_feature_path + 'image_features.csv', newline=''), delimiter=' ', quotechar='|')
        for index, row in enumerate(feature_reader):
            if len(row) > 0 and row[0].split(".")[0]+"\n" in train_lines:
                train_writer.writerow(row)
            elif len(row) > 0 and row[0].split(".")[0]+"\n" in val_lines:
                val_writer.writerow(row)
            elif len(row) > 0 and row[0].split(".")[0]+"\n" in test_lines:
                test_writer.writerow(row)


def main():
    image_features = ImageFeatures("./flickr30kentities/", "./flickr30kimagefeatures/")
    image_features.create_feature_files()


if __name__ == "__main__":
    main()
