import random
import cv2
import numpy as np
from matplotlib import pyplot as plt


def ransac(src_points, dst_points, threshold, max_iterations, confidence):
    src_points = np.array(src_points, dtype=np.float64)
    dst_points = np.array(dst_points, dtype=np.float64)

    h, w = src_points.shape
    bss = 0
    for i in range(max_iterations):
        if bss / h >= confidence:
            break
        src_samples, dts_samples = get_samples(src_points, dst_points)
        model = get_homography(src_samples, dts_samples)
        x_prime = transform(src_points, model)
        error = np.linalg.norm(x_prime - dst_points, axis=1)
        size = np.sum(error < threshold)
        if size > bss:
            bss = size
            bs_inliers = error < threshold

    src_final_points = src_points[bs_inliers]
    dst_final_points = dst_points[bs_inliers]
    return get_homography(src_final_points, dst_final_points), bs_inliers


def get_samples(src_points, dst_points):
    h, w = src_points.shape
    src_samples = np.ndarray((4, 2))
    dts_samples = np.ndarray((4, 2))
    for j in range(4):
        index = random.randint(0, h - 1)
        src_samples[j, :] = src_points[index, :]
        dts_samples[j, :] = dst_points[index, :]
    return src_samples, dts_samples


def get_homography(src_points, dst_points):
    src_points, mat_1 = get_normalized(np.array(src_points, dtype=np.float64))
    dst_points, mat_2 = get_normalized(np.array(dst_points, dtype=np.float64))
    matrix = create_homo_mat(src_points, dst_points)
    return calc_h(matrix, mat_1, mat_2)


def create_homo_mat(src_points, dst_points):
    h, w = src_points.shape
    matrix = np.zeros((2 * h, 9))
    points = zip(src_points, dst_points)

    for i, (pt1, pt2) in enumerate(points):
        x, y = pt1[0], pt1[1]
        x_prime, y_prime = pt2[1], pt2[0]
        matrix[2 * i] = np.array([-x, -y, -1, 0, 0, 0, x * x_prime, y * y_prime, y_prime])
        matrix[(2 * i) + 1] = np.array([0, 0, 0, -x, -y, -1, x * x_prime, y * x_prime, x_prime])
    return matrix


def transform(src_points, mat):
    src_g = np.ndarray((src_points.shape[0], 3))
    src_g[:, :2] = src_points
    src_g[:, 2] = np.ones(src_points.shape[0], dtype=np.float64)
    dst_points = np.dot(mat, src_g.transpose()).transpose()
    dst_points[:, 2][dst_points[:, 2] == 0] = 0.000001
    dst_points[:, 0] = dst_points[:, 0] / dst_points[:, 2]
    dst_points[:, 1] = dst_points[:, 1] / dst_points[:, 2]
    return dst_points[:, :2]


def get_normalized(points):
    std_p = np.sqrt(2)
    mean, std = np.mean(points, axis=0), np.std(points, axis=0)
    std[std < 0.00001] = 0.00001
    normalize_mat = np.array(
        [[1 / std[0] * std_p, 0, -mean[0] / std[0] * std_p], [0, 1 / std[1] * std_p, -mean[1] / std[1] * std_p],
         [0, 0, 1]])
    return transform(points, normalize_mat), normalize_mat


def calc_h(mat, t1, t2):
    U, S, V = np.linalg.svd(mat)
    h = V[-1, :]
    h = h.reshape((3, 3))
    return np.dot(np.linalg.inv(t2), np.dot(h, t1))


def find_matches(descriptors_1, descriptors_2, key_points_1, key_points_2, match_threshold, k):
    matched_points = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptors_1, descriptors_2, k)
    good_points = []
    correspondence_points = []
    kp1, kp2 = [], []
    kp1_p, kp2_p = [], []
    for i, (m, n) in enumerate(matched_points):
        if m.distance < match_threshold * n.distance:
            good_points.append([m])
            correspondence_points.append(m)
            img2_idx = m.trainIdx
            img1_idx = m.queryIdx
            kp1.append(key_points_1[img1_idx].pt)
            kp2.append(key_points_2[img2_idx].pt)
            kp1_p.append(key_points_1[img1_idx])
            kp2_p.append(key_points_2[img2_idx])
    return good_points, correspondence_points, kp1, kp2, kp1_p, kp2_p


def get_corners(matrix, img):
    h, w, _ = img.shape
    points = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype='float64').reshape(-1, 1, 2)
    warped_points = np.array(cv2.perspectiveTransform(points, matrix), dtype=int)
    return warped_points


def process(img1, img2):
    green = (0, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)

    sift = cv2.SIFT_create()
    key_point_1, descriptors_1 = sift.detectAndCompute(img1, None)
    key_point_2, descriptors_2 = sift.detectAndCompute(img2, None)

    interest_points = cv2.drawMatches(img1, key_point_1, img2, key_point_2, None, None, singlePointColor=green)
    cv2.imwrite('res22_corners.jpg', interest_points)

    match_ratio_threshold = 0.7
    k = 2
    good_points, matches, kp1, kp2, kp1_p, kp2_p = find_matches(descriptors_1, descriptors_2, key_point_1, key_point_2,
                                                  match_ratio_threshold, k)

    img_correspondence_points = interest_points.copy()
    cv2.drawMatches(img1, kp1_p, img2, kp2_p, None, img_correspondence_points, singlePointColor=blue,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imwrite('res23_correspondence.jpg', img_correspondence_points)

    img_matched_points = interest_points.copy()
    cv2.drawMatchesKnn(img1, key_point_1, img2, key_point_2, good_points, img_matched_points, matchColor=blue,
                       singlePointColor=green)
    cv2.imwrite('res24_matches.jpg', img_matched_points)

    img_random_matches = interest_points.copy()
    random.shuffle(matches)
    cv2.drawMatches(img1, key_point_1, img2, key_point_2, matches[:20], img_random_matches, matchColor=blue,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS + cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
    cv2.imwrite('res25.jpg', img_random_matches)

    transition_matrix, mask = ransac(kp2, kp1, 5, 7000, 0.9)

    img_inliers = cv2.drawMatches(img1, None, img2, None, None, None)
    cv2.drawMatches(img1, key_point_1, img2, key_point_2, matches, img_inliers
                    , matchColor=blue, singlePointColor=blue,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.drawMatches(img1, key_point_1, img2, key_point_2, matches, img_inliers
                    , matchColor=red, singlePointColor=red, matchesMask=np.multiply(mask, 1),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG + cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('res26.jpg', img_inliers)

    transition_matrix_inv = np.linalg.inv(transition_matrix)

    corners = get_corners(transition_matrix_inv, img1).reshape(-1, 1, 2)
    drawn_rect_img = cv2.polylines(img2, [corners], True, (0, 0, 255), thickness=3)
    img_1to2_warped = cv2.warpPerspective(img1, transition_matrix_inv, (img2.shape[1], img2.shape[0]))
    side_by_side_1tp2 = cv2.drawMatches(img_1to2_warped, None, drawn_rect_img, None, None, None)
    cv2.imwrite('res28.jpg', side_by_side_1tp2)

    # for i in range(len(kp1)):
    #     print(i)
    #     if mask[i]:
            # plt.imshow(cv2.circle(img1.copy(), (int(kp1[i][0]), int(kp1[i][1])), 10, (255, 0, 0), 10))
            # plt.show()
            # plt.imshow(cv2.circle(img2.copy(), (int(kp2[i][0]), int(kp2[i][1])), 10, (255, 0, 0), 10))
            # plt.show()

    corners = get_corners(transition_matrix, img2).reshape((4, 2))
    p_min, p_max = np.min(corners, axis=0), np.max(corners, axis=0)
    p_diff = tuple((p_max - p_min).astype(int))
    transition_p = np.matmul(np.array([[1, 0, -p_min[0]], [0, 1, -p_min[1]], [0, 0, 1]]), transition_matrix)
    side_by_side = cv2.warpPerspective(img2, transition_p, p_diff)
    cv2.imwrite('res29.jpg', side_by_side)

    side_by_side = cv2.drawMatches(img1, None, side_by_side, None, None, None)
    cv2.imwrite('res30.jpg', side_by_side)


image_1 = cv2.imread('im03.jpg')
image_2 = cv2.imread('im04.jpg')

process(image_1, image_2)
