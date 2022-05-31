import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from skimage import util, color, draw
from scipy.ndimage import gaussian_filter1d
from heapq import heapify, heappush, heappop

red = (0, 0, 255)


def read_all_frames(vc):
    all_frames = []
    while vc.isOpened() and len(all_frames) < 900:
        ret, frame = vc.read()
        if not ret:
            break
        all_frames.append(frame)
    vc.release()
    return all_frames


def save_file(file, name):
    f = open(name, 'wb')
    pickle.dump(file, f)
    f.close()


def load_file(name):
    f = open(name, 'rb')
    variable = pickle.load(f)
    f.close()
    return variable


def find_homography(img_1, img_2):
    down_scale = 0.5
    img_1 = cv2.resize(img_1, (0, 0), img_1, down_scale, down_scale)
    img_2 = cv2.resize(img_2, (0, 0), img_2, down_scale, down_scale)
    algo = cv2.SIFT_create()
    kp_1, desc_1 = algo.detectAndCompute(img_1, None)
    kp_2, desc_2 = algo.detectAndCompute(img_2, None)
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_1, desc_2, 2)
    matches = [i for i, j in matches if i.distance / j.distance < 0.8]
    final_points_1 = np.float32([kp_1[m.queryIdx].pt for m in matches])
    final_points_2 = np.float32([kp_2[m.trainIdx].pt for m in matches])
    h, mask = cv2.findHomography(final_points_1, final_points_2, cv2.RANSAC,
                                 maxIters=5000,
                                 confidence=0.99)
    cnv = np.array([[down_scale, 0, 0], [0, down_scale, 0], [0, 0, 1]])
    cnvi = np.array([[1 / down_scale, 0, 0], [0, 1 / down_scale, 0], [0, 0, 1]])
    return np.linalg.multi_dot((cnvi, h, cnv))


def warp_perspective(src, mat, sz):
    res = cv2.warpPerspective(src, mat, sz)
    return util.img_as_float64(res)


def find_frame(hs, h, w):
    src_points = np.array([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst_points = [tuple(cv2.perspectiveTransform(src_points.astype(float), h).reshape((4, 2))) for h in hs]
    dst_points = np.concatenate(dst_points)
    p_min, p_max = np.min(dst_points, axis=0), np.max(dst_points, axis=0)
    p_diff = tuple((p_max - p_min).astype(int))
    ht = np.array([[1, 0, -p_min[0]],
                   [0, 1, -p_min[1]],
                   [0, 0, 1]])
    return p_diff, ht


def section_one(main_frames):
    height, width, _ = main_frames[0].shape

    h = find_homography(main_frames[1], main_frames[2])
    h2 = find_homography(main_frames[2], main_frames[2])
    rect1 = np.array([[500, 500], [1000, 500], [1000, 1000], [500, 1000]]).reshape((-1, 1, 2))
    rect2 = cv2.perspectiveTransform(rect1.astype(float), np.linalg.inv(h)).astype(int)
    rect_450 = cv2.polylines(main_frames[2].copy(), [rect1], True, red, thickness=3)
    rect_270 = cv2.polylines(main_frames[1].copy(), [rect2], True, red, thickness=3)
    cv2.imwrite("res01-450-rect.jpg", rect_450)
    cv2.imwrite("res02-270-rect.jpg", rect_270)

    size, ht = find_frame([h, h2], height, width)
    size = tuple(size)
    warped_270 = warp_perspective(main_frames[1], np.matmul(ht, h), size)
    warped_450 = warp_perspective(main_frames[2], ht, size)
    mask_450 = np.where(warped_450 > 0)
    warped_270[mask_450] = warped_450[mask_450]
    warped_270 = warped_270 / np.max(warped_270) * 255
    cv2.imwrite('res03-270-450-panorama.jpg', warped_270)


def find_all_homography_mats(frames):
    homography_mats = [None] * 900

    homography_mats[450] = np.eye(3)
    homography_mats[270] = find_homography(frames[270], frames[450])
    homography_mats[630] = find_homography(frames[630], frames[450])
    homography_mats[90] = np.matmul(find_homography(frames[90], frames[270]), homography_mats[270])
    homography_mats[810] = np.matmul(find_homography(frames[810], frames[630]), homography_mats[630])

    for i in range(-450, 450):
        index = i + 450
        ref = index - ((i + 90) % 180 - 90)
        h = find_homography(frames[index], frames[ref])
        homography_mats[index] = np.matmul(h, homography_mats[ref])

    return homography_mats


def laplacian_pyramid(img, iterations):
    pyramid = [img.copy()]
    for i in range(iterations):
        blurred = cv2.GaussianBlur(pyramid[i], (17, 17), 0)
        pyramid[i] -= blurred
        pyramid.append(blurred)
    return pyramid


def blend(src, tar, msk, bandwidth):
    msk = cv2.GaussianBlur(msk, (bandwidth, bandwidth), 0)
    msk = cv2.merge((msk, msk, msk))
    return src * msk + tar * (1 - msk)


def multi_band_blend(src, tar, msk, iterations, ws1, ws2):
    src = src.astype(float)
    tar = tar.astype(float)
    msk = color.rgb2gray(msk)

    src_laplacian = laplacian_pyramid(src, iterations)
    tar_laplacian = laplacian_pyramid(tar, iterations)

    src_laplacian[iterations] = blend(src_laplacian[iterations], tar_laplacian[iterations], msk, ws1)
    for i in range(iterations - 1, -1, -1):
        src_laplacian[i] = blend(src_laplacian[i], tar_laplacian[i], msk, ws2)
        src_laplacian[i] += src_laplacian[i + 1]
    return src_laplacian[0]


def get_min_cut_mask(diff):
    cut = np.zeros(diff.shape, dtype=bool)
    path = get_min_cut_path(diff, diff.shape)
    index = 0
    for i in path:
        cut[index, :i] = True
        index += 1
    return 1 - cut


def get_min_cut_path(diff, shape):
    h, w = shape

    paths = []
    for i in range(len(diff[0])):
        paths.append([diff[0][i], [i]])
    heapify(paths)

    seen = set()
    while True:
        top = heappop(paths)
        error, path = top[0], top[1]
        length = len(path)
        index = path[length - 1]
        if length >= h:
            best_path = path
            break
        for next_index in [index - 1, index, index + 1]:
            if 0 <= next_index < w:
                if (length, next_index) not in seen:
                    updated_err = error + diff[length, next_index]
                    updated_path = path + [next_index]
                    heappush(paths, [updated_err, updated_path])
                    seen.add((length, next_index))
    return best_path


def find_mask(img_1, img_2):
    img_1, img_2 = img_1.astype(float), img_2.astype(float)
    img1_mask, img2_mask = (img_1 > 0.001).astype(int), (img_2 > 0.001).astype(int)
    ag = np.argwhere(np.logical_and(img_1 > 0, (img_2 > 0)).astype(int))
    min_p, max_p = np.min(ag, axis=0), np.max(ag, axis=0)
    diff = np.sum((img_1 - img_2) ** 2, axis=2)
    min_cut_mask = get_min_cut_mask(diff[:, min_p[1]:max_p[1]]).astype(int)
    final_mask = 1.0 * img2_mask
    final_mask[:, min_p[1]:max_p[1]] *= min_cut_mask[..., None]
    overlap = (1.0 * (img_1 > 0.001)) * (1.0 * (img_2 > 0.001))
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    overlap = cv2.morphologyEx(overlap, cv2.MORPH_CLOSE, kernel)
    img1_mask = cv2.morphologyEx(img1_mask.astype(float), cv2.MORPH_CLOSE, kernel)
    img2_mask = cv2.morphologyEx(img2_mask.astype(float), cv2.MORPH_CLOSE, kernel)
    return final_mask, overlap, img1_mask, img2_mask


def section_two(main_frames):
    h_450 = find_homography(main_frames[2], main_frames[2])
    h_270 = find_homography(main_frames[1], main_frames[2])
    h_630 = find_homography(main_frames[3], main_frames[2])
    h_90 = np.matmul(find_homography(main_frames[0], main_frames[1]), h_270)
    h_810 = np.matmul(find_homography(main_frames[4], main_frames[3]), h_630)

    h, w, _ = main_frames[0].shape
    mains = [h_90, h_270, h_450, h_630, h_810]
    p_diff, M_t = find_frame(mains, h, w)

    res04 = warp_perspective(main_frames[0], np.matmul(M_t, h_90), p_diff)

    key_frame_idx = [90, 270, 450, 630, 810]

    for i, idx in enumerate(key_frame_idx):
        if i == 0:
            continue
        tmp = warp_perspective(main_frames[i], np.matmul(M_t, mains[i]), p_diff)
        mask_m, mask_overlap, img1_mask, img2_mask = find_mask(res04, tmp)
        if i in [1, 2]:
            img1_mask = cv2.erode(img1_mask, np.ones((10, 10), np.uint8), iterations=2)
            img1_mask = cv2.blur(img1_mask.astype(float), (10, 10))
            res04 = multi_band_blend(tmp, res04, mask_m, 5, 9, 19) * img1_mask + tmp * (1 - img1_mask)
        else:
            mask_m = cv2.erode(mask_m, np.ones((25, 25), np.uint8), iterations=2)
            mask_m = cv2.blur(mask_m.astype(float), (25, 25))
            img1_mask = cv2.erode(img1_mask, np.ones((10, 10), np.uint8), iterations=2)
            img1_mask = cv2.blur(img1_mask.astype(float), (25, 25))
            res04 = multi_band_blend(tmp, res04, mask_m, 5, 9, 19) * img1_mask + tmp * (1 - img1_mask)

        del tmp
        del mask_m
        del mask_overlap
    cv2.imwrite('res04-key-frames-panorama.jpg', res04 * 255)


def section_three(frames, homography_mats):
    h, w, _ = frames[0].shape
    size, m = find_frame(homography_mats, h, w)
    vw = cv2.VideoWriter('res05-reference-plane.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

    for i in range(900):
        vw.write(np.uint8(warp_perspective(frames[i], np.matmul(m, homography_mats[i]), size) * 255))
    vw.release()


def section_four():
    panoramas = []
    frame_h, frame_w = 807, 1966
    # l = 200
    wws = [(0, frame_w), (frame_w, 2 * frame_w), (2 * frame_w, 3 * frame_w),
           (0, frame_w), (frame_w, 2 * frame_w), (2 * frame_w, 3 * frame_w),
           (0, frame_w), (frame_w, 2 * frame_w), (2 * frame_w, 3 * frame_w)]
    hws = [(0, frame_h), (0, frame_h), (0, frame_h),
           (frame_h, 2 * frame_h), (frame_h, 2 * frame_h), (frame_h, 2 * frame_h),
           (2 * frame_h, 3 * frame_h), (2 * frame_h, 3 * frame_h), (2 * frame_h, 3 * frame_h)]
    for i in range(9):
        m = create_one_section_of_panorama(i, hws[i], wws[i])
        panoramas.append(m)
        cv2.imwrite('part' + str(i) + '.jpg', m)

    # for i in range(9):
    #     panoramas.append(cv2.imread('part' + str(i) + '.jpg'))

    res06 = np.vstack((np.hstack((panoramas[0], panoramas[1], panoramas[2])),
                       np.hstack((panoramas[3], panoramas[4], panoramas[5])),
                       np.hstack((panoramas[6], panoramas[7], panoramas[8]))))

    cv2.imwrite('res06-background-panorama.jpg', res06)


def create_one_section_of_panorama(part, hw, ww):
    vc = cv2.VideoCapture('res05-reference-plane.mp4')
    frames = []
    current_frame = 0
    end_frame_num = 700

    if part % 3 == 1:
        end_frame_num = 900
        while current_frame < 1:
            vc.read()
            current_frame += 1
    elif part % 3 == 2:
        end_frame_num = 900
        while current_frame < 200:
            vc.read()
            current_frame += 1

    while True:
        if current_frame % 6 != 0:
            current_frame += 1
            vc.read()
        else:
            ret, frame = vc.read()
            frame = frame.astype('uint8')
            frames.append(frame[hw[0]:hw[1], ww[0]:ww[1], :])
            current_frame += 1
            del frame
        if end_frame_num <= current_frame:
            break

    print(current_frame)
    frames = np.ma.masked_equal(frames, 0)
    return np.ma.median(frames, axis=0).filled(0)


def section_five(homography_mats, ):
    res06 = cv2.imread('res06-background-panorama.jpg')
    h, w = 1080, 1920

    _, m = find_frame(homography_mats, h, w)
    vw = cv2.VideoWriter('res07-background-video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (w, h))
    for i, mat in enumerate(homography_mats):
        vw.write(np.uint8(warp_perspective(res06, np.linalg.pinv(np.dot(m, mat)), (w, h)) * 255))
    vw.release()


def find_foreground_mask(frame1, frame2):
    thresh = 20
    mask1 = np.where(np.abs(frame2[:, :, 0] - frame1[:, :, 0]) > thresh, 1, 0)
    mask2 = np.where(np.abs(frame2[:, :, 1] - frame1[:, :, 1]) > thresh, 1, 0)
    mask3 = np.where(np.abs(frame2[:, :, 2] - frame1[:, :, 2]) > thresh, 1, 0)

    return np.where((mask1 == 1) | (mask2 == 1) | (mask3 == 1), 255, 0)


def get_final_mask(frame1, frame2, blur1, blur2, close, open):
    frame1 = cv2.GaussianBlur(frame1, (blur1, blur1), 0)
    frame2 = cv2.GaussianBlur(frame2, (blur1, blur1), 0)

    diff_mag = np.linalg.norm(frame1.astype(np.int16) - frame2.astype(np.int16), axis=2)
    mask = ((diff_mag > 80) & (np.linalg.norm(frame2, axis=2) > 1))

    mask2 = find_foreground_mask(frame1, frame2)
    mask = np.logical_and(mask, mask2).astype(np.float64)
    # plt.imshow(mask)
    # plt.show()
    mask = cv2.GaussianBlur(mask.astype(float), (blur2, blur2), 0)
    # plt.imshow(mask)
    # plt.show()
    mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_CLOSE, np.ones((close, close), dtype=np.uint8))
    # plt.imshow(mask)
    # plt.show()
    mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_OPEN, np.ones((open, open), dtype=np.uint8)).astype(float)
    # plt.imshow(mask)
    # plt.show()
    mask = cv2.GaussianBlur(mask.astype(float), (9, 9), 0).astype(np.float64)
    # plt.imshow(mask)
    # plt.show()

    return mask


def section_six():
    vc1 = cv2.VideoCapture('res07-background-video.mp4')
    vc2 = cv2.VideoCapture('video.mp4')
    count = 0
    vw = cv2.VideoWriter('res08-foreground-video.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1920, 1080))
    while vc1.isOpened() and count < 900:
        ret, frame1 = vc1.read()
        ret, frame2 = vc2.read()
        if not ret:
            break

        if count < 150:
            mask = get_final_mask(frame1, frame2, 3, 3, 5, 7)
        else:
            mask = get_final_mask(frame1, frame2, 3, 3, 11, 3)

        frame2 = frame2.copy().astype(np.float64)
        frame2[:, :, 2] = frame2[:, :, 2] + np.ones_like(frame2[:, :, 2], dtype=np.float64) * 100 * mask
        frame2[:, :, 2][frame2[:, :, 2] > 255] = 255

        vw.write(frame2.astype('uint8'))

        count += 1
        print(count)
    vc1.release()
    vc2.release()
    vw.release()


def section_seven(homography_mats):
    res06 = cv2.imread('res06-background-panorama.jpg')
    h, w = 1080, 1920

    _, m = find_frame(homography_mats, h, w)
    size = (int(w * 1.5), h)

    vw = cv2.VideoWriter('res09-background-video-wider.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, size)
    for i, mat in enumerate(homography_mats):
        vw.write(np.uint8(warp_perspective(res06, np.linalg.inv(np.matmul(m, mat)), size) * 255))
    vw.release()


def section_eight(homography_mats):
    vc = cv2.VideoCapture('video.mp4')
    homography_mats = np.vstack(homography_mats).reshape(900, 3, 3)
    homography_matsp = np.vstack(homography_mats).reshape(900, 3, 3)
    h, w = 1080, 1920

    for i in range(3):
        for j in range(3):
            homography_matsp[:, i, j] = gaussian_filter1d(homography_matsp[:, i, j], 5)

    count = 0
    vw = cv2.VideoWriter('res10-video-shakeless.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (w, h))
    while vc.isOpened() and count < 900:
        ret, frame = vc.read()
        if not ret:
            break
        warped = warp_perspective(frame, np.linalg.inv(homography_matsp[count, :, :]) @ homography_mats[count, :, :],
                                  (w, h)) * 255
        warped[warped < 0.1] = frame[warped < 0.1]
        vw.write(np.uint8(warped))
        count += 1
    vw.release()


def process(vc):
    frames = read_all_frames(vc)
    # main_frames = [frames[90], frames[270], frames[450], frames[630], frames[810]]
    homography_mats = find_all_homography_mats(frames)
    # save_file(homography_mats, "homography_mats.pckl")
    homography_mats = load_file("homography_mats.pckl")

    # section_one(main_frames)
    # print('1')
    # section_two(main_frames)
    # print('2')
    # section_three(frames, homography_mats)
    # print('3')
    section_four()
    print('4')
    # section_five(homography_mats)
    # print('5')
    # section_six()
    # print('6')
    # section_seven(homography_mats)
    # print('7')
    # section_eight(homography_mats)
    # print('8')


video_cap = cv2.VideoCapture('video.mp4')
process(video_cap)
