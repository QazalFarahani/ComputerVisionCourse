import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_intersection(lines):
    A = np.ones((len(lines), 3))
    for i in range(len(lines)):
        r, theta = lines[i][0]
        A[i] = [np.cos(theta), np.sin(theta), -r]
    u, s, v = np.linalg.svd(A)
    v = v[-1]
    return v / v[2]


def find_focal_principal(v_x, v_y, v_z):
    A = np.array([[v_x[0] - v_z[0], v_x[1] - v_z[1]],
                  [v_y[0] - v_z[0], v_y[1] - v_z[1]]])
    b = np.array([v_y[0] * (v_x[0] - v_z[0]) + v_y[1] * (v_x[1] - v_z[1]),
                  v_x[0] * (v_y[0] - v_z[0]) + v_x[1] * (v_y[1] - v_z[1])])
    p = np.linalg.solve(A, b)
    f = np.sqrt(-p[0] ** 2 - p[1] ** 2 + (v_x[0] + v_y[0]) * p[0] + (v_x[1] + v_y[1]) * p[1] - (
            v_x[0] * v_y[0] + v_x[1] * v_y[1]))
    return p, f


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

    p, f = find_focal_principal(v_x, v_y, v_z)
    print("p: ", p, "f: ", f)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(rgb_img.copy().astype('uint8'))
    ax.scatter(p[0], p[1], color=(1, 0, 0))
    plt.title(f"f = {f}")
    fig.savefig("res03.jpg")

    K = np.array([[f, 0, p[0]],
                  [0, f, p[1]],
                  [0, 0, 1]])
    Ki = np.linalg.inv(K)

    print("K: ", K)

    z_rotation = np.arctan(float(v_y[1] - v_x[1]) / float(v_y[0] - v_x[0]))
    camera_z = Ki.dot(v_z)
    camera_z = camera_z / np.linalg.norm(camera_z)
    main_z = [0, 0, 1]
    main_z = main_z / np.linalg.norm(main_z)
    x_rotation = np.pi / 2 - np.arccos(np.dot(camera_z, main_z))
    print("z theta:", -np.rad2deg(z_rotation), "x theta:", np.rad2deg(x_rotation))


image = cv2.imread('vns.jpg')
process(image)
