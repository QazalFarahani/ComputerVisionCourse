import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import os

categories = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial', 'Inside_City', 'Kitchen', 'Livingroom', 'Mountain',
              'Office', 'Open_Country', 'Store', 'Street', 'Suburb', 'Tall_Building']


def read_data(root):
    data, labels = [], []
    for path, subdirs, files in os.walk(root):
        for file in files:
            img = cv2.imread(path + '/' + file)
            data.append(img)
            label = categories.index(path.split('/')[2])
            labels.append(label)
    return data, np.array(labels)


def knn(train_data, test_data, train_labels, test_labels, metric, n):
    model = KNeighborsClassifier(n_neighbors=n, metric=metric)
    scores = cross_validate(model, train_data, train_labels)
    model = KNeighborsClassifier(n_neighbors=n, metric=metric)
    model.fit(train_data, train_labels)
    acc = model.score(test_data, test_labels)
    return np.average(scores['test_score']), acc


def get_features(num, data):
    sift = cv2.SIFT_create()
    all_features = []
    features = []
    for img in data:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        all_features.append(desc)
        features.extend(desc[:num])
    return all_features, features


def get_histograms(km, all_features):
    histograms = []
    for i in range(len(all_features)):
        words = km.predict(all_features[i]).tolist()
        hist = []
        for index in range(km.n_clusters):
            hist.append(words.count(index))
        hist = np.array(hist)
        hist = hist / hist.sum()
        histograms.append(hist)
    return np.array(histograms)


def process():
    all_features_train, features_train = get_features(15, train_data)
    all_features_test, features_test = get_features(15, test_data)
    km = KMeans(n_clusters=75, random_state=0).fit(np.array(features_train))
    train_histograms = get_histograms(km, all_features_train)
    test_histograms = get_histograms(km, all_features_test)

    train_acc, test_acc = knn(train_histograms, test_histograms, train_labels, test_labels, 'manhattan', n=2)
    print(train_acc, test_acc)
    train_acc, test_acc = knn(train_histograms, test_histograms, train_labels, test_labels, 'manhattan', n=4)
    print(train_acc, test_acc)
    train_acc, test_acc = knn(train_histograms, test_histograms, train_labels, test_labels, 'manhattan', n=6)
    print(train_acc, test_acc)
    train_acc, test_acc = knn(train_histograms, test_histograms, train_labels, test_labels, 'manhattan', n=8)
    print(train_acc, test_acc)
    train_acc, test_acc = knn(train_histograms, test_histograms, train_labels, test_labels, 'manhattan', n=10)
    print(train_acc, test_acc)
    train_acc, test_acc = knn(train_histograms, test_histograms, train_labels, test_labels, 'manhattan', n=12)
    print(train_acc, test_acc)
    train_acc, test_acc = knn(train_histograms, test_histograms, train_labels, test_labels, 'manhattan', n=14)
    print(train_acc, test_acc)
    train_acc, test_acc = knn(train_histograms, test_histograms, train_labels, test_labels, 'manhattan', n=16)
    print(train_acc, test_acc)


train_data, train_labels = read_data("Data/Train/")
test_data, test_labels = read_data("Data/Test/")
process()
