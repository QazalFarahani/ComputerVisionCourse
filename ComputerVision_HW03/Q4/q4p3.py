import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import os

from sklearn.svm import SVC

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


def svc(train_data, test_data, train_labels, test_labels):
    model = SVC(C=1.2, kernel='rbf')
    scores = cross_validate(model, train_data, train_labels)
    model = SVC(C=1.2, kernel='rbf')
    model.fit(train_data, train_labels)
    acc = model.score(test_data, test_labels)

    cm = confusion_matrix(test_labels, model.predict(test_data))

    return np.average(scores['test_score']), acc, cm


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

    train_acc, test_acc, cm = svc(train_histograms, test_histograms, train_labels, test_labels)
    print(train_acc, test_acc)
    plt.matshow(cm)
    plt.savefig('res09.jpg')


train_data, train_labels = read_data("Data/Train/")
test_data, test_labels = read_data("Data/Test/")
process()
