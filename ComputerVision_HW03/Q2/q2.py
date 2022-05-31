import cv2
import numpy as np
import random

from matplotlib import pyplot as plt


def draw_points(points, img, color):
    for p in points:
        cv2.circle(img, (int(p[0]), int(p[1])), 8, color, -1)
    return img


def process(img1, img2):
    sift = cv2.SIFT_create()
    key_points_1, des1 = sift.detectAndCompute(img1, None)
    key_points_2, des2 = sift.detectAndCompute(img2, None)
    kp1, kp2 = find_matches(des1, des2, key_points_1, key_points_2, 0.6, 2)
    F, mask = cv2.findFundamentalMat(kp1, kp2, cv2.FM_RANSAC, 2, 0.99)
    print(F)

    interests1 = draw_points(kp1, img1.copy(), red)
    interests2 = draw_points(kp2, img2.copy(), red)
    mask = [i[0] for i in mask]
    kp1p, kp2p = [], []
    for i in range(len(kp1)):
        if mask[i]:
            kp1p.append(kp1[i])
            kp2p.append(kp2[i])
    interests1 = draw_points(kp1p, interests1, green)
    interests2 = draw_points(kp2p, interests2, green)
    cv2.imwrite('res05.jpg', cv2.drawMatches(interests1, None, interests2, None, None, None))

    e1 = get_epipole(F)
    e2 = get_epipole(F.transpose())
    print("e:", e1)
    print("e prim:", e2)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).astype('uint8'))
    plt.scatter(e1[0], e1[1])
    fig.savefig("res06.jpg")

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB).astype('uint8'))
    plt.scatter(e2[0], e2[1])
    fig.savefig("res07.jpg")

    res1, res2 = get_lined_image(kp1p, F, img1, kp2p, np.transpose(F), img2)

    result = cv2.drawMatches(res1, None, res2, None, None, None)
    cv2.imwrite('res08.jpg', result)


def get_lined_image(kp1p, f, img2, kp2p, ft, img1):
    h, w, _ = img1.shape
    for i in range(11):
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

        p = kp1p[i]
        p = np.array([p[0], p[1], 1])
        a, bp, c = np.dot(f, p)
        x, y = 0, int(-c / bp)
        x1, y1 = img1.shape[1], int((-c - a * w) / bp)
        cv2.line(img1, (x, y), (x1, y1), (r, g, b), 4)

        p = kp2p[i]
        p = np.array([p[0], p[1], 1])
        a, bp, c = np.dot(ft, p)
        x, y = 0, int(-c / bp)
        x1, y1 = img1.shape[1], int((-c - a * w) / bp)
        cv2.line(img2, (x, y), (x1, y1), (r, g, b), 4)

    return img1, img2


def get_epipole(matrix):
    u, s, v = np.linalg.svd(matrix)
    return v[-1] / v[-1][2]


def find_matches(descriptors_1, descriptors_2, key_points_1, key_points_2, match_threshold, k):
    matched_points = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_1, descriptors_2, k)
    kp1, kp2 = [], []
    for i, (m, n) in enumerate(matched_points):
        if m.distance < match_threshold * n.distance:
            img2_idx = m.trainIdx
            img1_idx = m.queryIdx
            kp1.append(key_points_1[img1_idx].pt)
            kp2.append(key_points_2[img2_idx].pt)
    return np.float32(kp1), np.float32(kp2)


green = (100, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)

image1 = cv2.imread('01.jpg')
image2 = cv2.imread('02.jpg')

process(image1, image2)
