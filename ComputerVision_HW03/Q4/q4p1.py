import numpy as np
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import os

categories = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'Inside_City', 'Kitchen', 'Livingroom', 'Mountain',
              'Office', 'Open_Country', 'Store', 'Street', 'Suburb', 'Tall_Building']


def read_data(root):
    data, labels = [], []
    for path, subdirs, files in os.walk(root):
        for file in files:
            img = cv.imread(path + '/' + file)
            data.append(img)
            label = categories.index(path.split('/')[2])
            labels.append(label)
    return data, np.array(labels)


def knn(train_data, train_labels, test_data, test_labels, metric, n, s):
    train_resized = np.array([cv.resize(img, (s, s), interpolation=cv.INTER_AREA).ravel() for img in train_data])
    test_resized = np.array([cv.resize(img, (s, s), interpolation=cv.INTER_AREA).ravel() for img in test_data])
    model = KNeighborsClassifier(n_neighbors=n, metric=metric)
    scores = cross_validate(model, train_resized, train_labels)
    model = KNeighborsClassifier(n_neighbors=n, metric=metric)
    model.fit(train_resized, train_labels)
    acc = model.score(test_resized, test_labels)
    return np.average(scores['test_score']), acc


train_data, train_labels = read_data("Data/Train/")
test_data, test_labels = read_data("Data/Test/")

train_acc, test_acc = knn(train_data, train_labels, test_data, test_labels, 'manhattan', n=6, s=8)
print("train_acc, test_acc", train_acc, test_acc)

sizes = [4, 6, 8, 10, 12, 14, 16]
neighbours = [1, 2, 3, 4, 5, 6, 7]
accuracies = []
train_accuracies = []
for s in sizes:
    for n in neighbours:
        train_acc, test_acc = knn(train_data, train_labels, test_data, test_labels, 'manhattan', n=n, s=s)
        accuracies.append((s, n, train_acc, test_acc))
        train_accuracies.append(train_acc)
print(accuracies)
print(train_accuracies)
print(np.max(train_accuracies))

