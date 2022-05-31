import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def get_intersection(lines):
    A = np.ones((len(lines), 3))
    for i in range(len(lines)):
        r, theta = lines[i][0]
        A[i] = [np.cos(theta), np.sin(theta), -r]
    u, s, v = np.linalg.svd(A)
    v = v[-1]
    return v / v[2]


def process(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 1)
    gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    edges = cv2.Canny(gray_image, 150, 200, None, 3)

    x_lines = cv2.HoughLines(edges, 1, np.pi / 150, 300, min_theta=1.64, max_theta=1.75)
    y_lines = cv2.HoughLines(edges, 1, np.pi / 150, 300, min_theta=1, max_theta=1.5)
    z_lines = cv2.HoughLines(edges, 1, np.pi / 150, 450, min_theta=-0.2, max_theta=0)
    x_lines = np.delete(x_lines, [0, 3, 4, 5, 6, 8], 0)
    y_lines = np.delete(y_lines, [3, 4, 5, 6, 7, 8, 9, 10, 13, 14], 0)
    z_lines = np.delete(z_lines, [3, 4, 5, 6, 7, 8, 9, 11, 12, 13], 0)

    v_x = np.round(get_intersection(x_lines)).astype(int)
    v_y = np.round(get_intersection(y_lines)).astype(int)
    v_z = np.round(get_intersection(z_lines)).astype(int)
    print("vx, vy, vz: ", v_x, v_y, v_z)

    h = np.cross(v_x, v_y)
    height, width, _ = img.shape
    res01 = (np.ones((height + 300, width, 3)) * 255).astype('uint8')

    res01[:height, :, :] = img
    res01 = cv2.line(res01, (int(v_x[0]), int(v_x[1])), (int(v_y[0]), int(v_y[1])), (0, 0, 255), 8, lineType=8)
    cv2.imwrite("res01.jpg", res01)

    normalized_h = h / np.linalg.norm(h[0:2])
    print("normalized h: ", normalized_h)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rgb_img.astype('uint8'))
    ax.scatter(v_x[0], v_x[1], color=(1, 0, 0))
    ax.scatter(v_y[0], v_y[1], color=(0, 1, 0))
    ax.scatter(v_z[0], v_z[1], color=(0, 0, 1))
    ax.plot([v_x[0], v_y[0]], [v_x[1], v_y[1]], linewidth=1, color=(1, 0, 0))
    fig.set_size_inches(fig.get_size_inches() * 3)
    fig.savefig("res02.jpg")


image = cv2.imread('vns.jpg')
process(image)
