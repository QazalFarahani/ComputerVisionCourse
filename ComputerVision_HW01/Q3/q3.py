import random

import cv2
import numpy as np
from matplotlib import pyplot as plt


def find_matches(descriptors_1, descriptors_2, key_points_1, key_points_2, match_threshold, k):
    matched_points = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_1, descriptors_2, k=k)
    good_points = []
    correspondence_points = []
    kp1, kp2 = [], []
    for i, (m, n) in enumerate(matched_points):
        if m.distance / n.distance < match_threshold:
            good_points.append([m])
            correspondence_points.append(m)
            img2_idx = m.trainIdx
            img1_idx = m.queryIdx
            kp1.append(key_points_1[img1_idx])
            kp2.append(key_points_2[img2_idx])
    return good_points, correspondence_points, kp1, kp2


def find_homography(kp1, kp2):
    final_matches1 = np.float64([p.pt for p in kp1])
    final_matches2 = np.float64([p.pt for p in kp2])
    transition_matrix, mask = cv2.findHomography(final_matches2, final_matches1, cv2.RANSAC,
                                                 ransacReprojThreshold=5, maxIters=2000, confidence=0.999)
    mask = [i[0] for i in mask]
    return transition_matrix, mask, final_matches1, final_matches2


def get_corners(matrix, img):
    h, w, _ = img.shape
    points = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype='float64').reshape(-1, 1, 2)
    warped_points = np.array(cv2.perspectiveTransform(points, matrix), dtype=int)
    return warped_points


def process(img1, img2):
    sift = cv2.SIFT_create()
    key_points_1, descriptors_1 = sift.detectAndCompute(img1, None)
    key_points_2, descriptors_2 = sift.detectAndCompute(img2, None)

    green = (100, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)

    interest_points = cv2.drawMatches(img1, key_points_1, img2, key_points_2, None, None, singlePointColor=green)
    cv2.imwrite('res13_corners.jpg', interest_points)

    good_points, matched_points, kp1, kp2 = find_matches(descriptors_1, descriptors_2, key_points_1, key_points_2, 0.74,
                                                         2)
    img_correspondence_points = interest_points.copy()

    cv2.drawMatches(img1, kp1, img2, kp2, None, img_correspondence_points, singlePointColor=blue,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imwrite('res14_correspondence.jpg', img_correspondence_points)

    img_matched_points = interest_points.copy()
    cv2.drawMatchesKnn(img1, key_points_1, img2, key_points_2, good_points, img_matched_points, matchColor=blue,
                       singlePointColor=green)
    cv2.imwrite('res15_matches.jpg', img_matched_points)

    img_random_matches = interest_points.copy()
    rand_matched = matched_points.copy()
    random.shuffle(rand_matched)
    cv2.drawMatches(img1, key_points_1, img2, key_points_2, rand_matched[:20], img_random_matches, matchColor=blue,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS + cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imwrite('res16.jpg', img_random_matches)

    transition_matrix, mask, final_matches1, final_matches2 = find_homography(kp1, kp2)
    img_inliers = cv2.drawMatches(img1, None, img2, None, None, None)
    cv2.drawMatches(img1, key_points_1, img2, key_points_2, matched_points, img_inliers
                    , matchColor=blue, singlePointColor=blue,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.drawMatches(img1, key_points_1, img2, key_points_2, matched_points, img_inliers
                    , matchColor=red, singlePointColor=red, matchesMask=mask,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('res17.jpg', img_inliers)

    for i in range(len(final_matches1)):
        # print(i)
        if mask[i]:
            pass
            # plt.imshow(cv2.circle(img1.copy(), (int(final_matches1[i][0]), int(final_matches1[i][1])), 10, (255, 0, 0), 10))
            # plt.show()
            # plt.imshow(cv2.circle(img2.copy(), (int(final_matches2[i][0]), int(final_matches2[i][1])), 10, (255, 0, 0), 10))
            # plt.show()

    transition_matrix_inv = np.linalg.inv(transition_matrix)
    corners = get_corners(transition_matrix_inv, img1).reshape(-1, 1, 2)
    drawn_rect_img = cv2.polylines(img2, [corners], True, (0, 0, 255), thickness=3)
    img_1to2_warped = cv2.warpPerspective(img1, transition_matrix_inv, (img2.shape[1], img2.shape[0]))
    side_by_side_1tp2 = cv2.drawMatches(img_1to2_warped, None, drawn_rect_img, None, None, None)
    cv2.imwrite('res19.jpg', side_by_side_1tp2)

    corners = get_corners(transition_matrix, img2).reshape((4, 2))
    p_min, p_max = np.min(corners, axis=0), np.max(corners, axis=0)
    p_diff = (p_max - p_min).astype(int)
    transition_p = np.matmul(np.array([[1, 0, -p_min[0]], [0, 1, -p_min[1]], [0, 0, 1]]), transition_matrix)
    side_by_side = cv2.warpPerspective(img2, transition_p, tuple(p_diff))
    cv2.imwrite('res20.jpg', side_by_side)

    side_by_side = cv2.drawMatches(img1, None, side_by_side, None, None, None)
    cv2.imwrite('res21.jpg', side_by_side)


image_1 = cv2.imread('im03.jpg')
image_2 = cv2.imread('im04.jpg')

process(image_1, image_2)
