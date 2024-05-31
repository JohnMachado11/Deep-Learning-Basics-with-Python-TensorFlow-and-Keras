import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import pickle

# Cats and Dogs dataset
# https://www.microsoft.com/en-us/download/details.aspx?id=54765

# Directory of Pet Images
DATADIR = ""
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50

# 0 is a dog
# 1 is a cat
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs directory
        class_num = CATEGORIES.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

# print(len(training_data))
random.shuffle(training_data)

# for sample in training_data:
    # print(sample[1])

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# pickle_x_in = open("X.pickle", "rb")
# X = pickle.load(pickle_x_in)

# pickle_y_in = open("y.pickle", "rb")
# y = pickle.load(pickle_y_in)

