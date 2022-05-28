import cv2
import numpy as np
import math


def process(img):
    distance = 40
    height = 25
    f = 500
    img_size1, img_size2 = (1000, 1700), (256, 256)

    k_1 = np.array([[f, 0, img_size1[0] // 2], [0, f, img_size1[1] // 2], [0, 0, 1]])
    k_2 = np.array([[f, 0, img_size2[0] // 2], [0, f, img_size2[1] // 2], [0, 0, 1]])

    r_1 = np.identity(3)
    t_1 = [0, 0, 0]
    theta = math.atan(distance / height)
    r_2 = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    t_2 = [0, distance, 0]

    rt_1 = np.zeros((3, 4))
    rt_2 = np.zeros((3, 4))
    rt_1[:3, :3] = r_1
    rt_1[:, -1] = t_1
    rt_2[:3, :3] = r_2
    rt_2[:, -1] = t_2

    points = np.array([[0, 0, height, 1], [10, 20, height, 1], [10, 10, height, 1], [20, 10, height, 1]]).transpose()
    x = np.dot(np.linalg.inv(np.vstack((rt_2, np.array([[0, 0, 0, 1]])))), points)

    x_1 = np.linalg.multi_dot([k_2, rt_1, x])
    x_2 = np.linalg.multi_dot([k_1, rt_2, x])

    x_1 = x_1 / x_1[2, :]
    x_2 = x_2 / x_2[2, :]
    x_1 = np.float32(x_1[:-1, :]).transpose()
    x_2 = np.float32(x_2[:-1, :]).transpose()

    H = cv2.getPerspectiveTransform(x_1, x_2)
    logo_warped = cv2.warpPerspective(img, H, img_size1)
    cv2.imwrite("res12.jpg", logo_warped)


image = cv2.imread("logo.png")
process(image)
