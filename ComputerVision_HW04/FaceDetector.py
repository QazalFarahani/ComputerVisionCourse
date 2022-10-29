import cv2 as cv
import numpy as np
import os
import pickle
from skimage import feature
from sklearn import utils
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def split_data(root, hog):
    data = []
    for path, subdirs, files in os.walk(root):
        for file in files:
            img = cv.imread(path + '/' + file)
            img = img[25:-25, 25:-25, :]
            img = cv.resize(img, (128, 128), interpolation=cv.INTER_AREA)
            data.append(hog.compute(img, (32, 32)).ravel())
    data = utils.shuffle(data)[:12000]
    return data[:10000], data[10000:11000], data[11000: 12000]


def svm(x, y):
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
    clf.fit(np.array(x), y.ravel())
    return clf


def evaluate_model(clf, x, y):
    predictions = np.zeros(y.shape)
    scores = np.zeros(y.shape)
    for i, z in enumerate(x):
        predictions[i] = clf.predict([z])
        scores[i] = clf.decision_function([z])
    return predictions, scores


def save_file(file, name):
    f = open(name, 'wb')
    pickle.dump(file, f)
    f.close()


def load_file(name):
    f = open(name, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable


def draw_rectangles(peaks, result, scales):
    for peak in peaks:
        pt1 = tuple(peak)
        scale = scales[pt1]
        pt2 = (np.array([pt1[0] * scale + 128, pt1[1] * scale + 128]) / scale)
        pt2 = tuple(pt2.astype(int)[::-1])
        print(pt1, scale)
        cv.rectangle(result, tuple(peak)[::-1], pt2, (255, 0, 0), 5)
    return result


def face_detection(clf, img, hog, scale_range, scale_step, min_dist, threshold, s):
    bbox_size = 128
    step = 10

    h, w, _ = img.shape
    result = np.zeros((h + s, w + s, 3)).astype(np.uint8)
    result[s // 2:h + s // 2, s // 2:w + s // 2, :] = img

    frame_scores = np.zeros(result.shape[:2])
    scales = np.zeros(result.shape[:2])

    for scale in np.arange(scale_range[0], scale_range[1], scale_step):
        img = cv.resize(result, (0, 0), None, scale, scale, cv.INTER_AREA)
        h, w, _ = img.shape
        for i in np.arange(0, h - bbox_size, step):
            for j in np.arange(0, w - bbox_size, step):
                patch = img[i:i + bbox_size, j:j + bbox_size, :]
                vector = hog.compute(patch, (32, 32))
                score = clf.decision_function([vector])[0]
                if score > -1:
                    pt1 = (np.array([i, j]) / scale).astype(int)
                    if frame_scores[pt1[0], pt1[1]] < score:
                        frame_scores[pt1[0], pt1[1]] = score
                        scales[pt1[0], pt1[1]] = scale

    peaks = feature.peak_local_max(frame_scores, min_distance=min_dist, threshold_abs=threshold)
    draw_rectangles(peaks, result, scales)
    return result


def learn_params():
    hog = cv.HOGDescriptor((128, 128), (16, 16), (16, 16), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

    train_pos, validation_pos, test_pos = split_data('lfw', hog)
    train_neg, validation_neg, test_neg = split_data('256_ObjectCategories', hog)

    train_data = train_pos + train_neg
    test_data = test_pos + test_neg
    validation_data = validation_pos + validation_neg

    train_label = np.vstack((np.ones((10000, 1)), np.zeros((10000, 1))))
    validation_label = np.vstack((np.ones((1000, 1)), np.zeros((1000, 1))))
    test_label = np.vstack((np.ones((1000, 1)), np.zeros((1000, 1))))

    clf = svm(train_data, train_label)

    prediction, score = evaluate_model(clf, np.array(validation_data), validation_label)
    validation_accuracy = metrics.accuracy_score(np.array(validation_label), prediction)
    print(f'validation accuracy = {validation_accuracy}')

    prediction, score = evaluate_model(clf, test_data, test_label)
    test_accuracy = metrics.accuracy_score(test_label, prediction)
    test_roc = metrics.roc_curve(test_label, score)
    precision_recall_test = metrics.precision_recall_curve(test_label, score)
    ap = metrics.average_precision_score(test_label, score)

    plt.plot(test_roc[0], test_roc[1])
    plt.title('ROC')
    plt.savefig("res1.jpg")
    plt.clf()

    plt.plot(precision_recall_test[0], precision_recall_test[1])
    plt.title(f'Precision-Recall, AP={ap}')
    plt.savefig("res2.jpg")
    print(f'test accuracy: = {test_accuracy}')

    # save_file(clf, "clf.pckl")
    return clf, hog


clf, hog = learn_params()

# clf = load_file("clf.pckl")

img = cv.imread('Melli.jpg')
img1_det = face_detection(clf, img, hog, (0.5, 1.2), 0.2, 90, 0.48, 100)
cv.imwrite("res4.jpg", img1_det)

img = cv.imread('Esteghlal.jpg')
img1_det = face_detection(clf, img, hog, (0.5, 0.6), 0.2, 40, 0.2, 200)
cv.imwrite("res6.jpg", img1_det)

img = cv.imread('Persepolis.jpg')
img1_det = face_detection(clf, img, hog, (0.5, 1.3), 0.2, 60, 0.45, 200)
cv.imwrite("res5.jpg", img1_det)



