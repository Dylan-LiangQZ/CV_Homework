import cv2
import numpy as np
import matplotlib
import random

matplotlib.use('TkAgg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# Faster R-CNN
# ratio = [0.5,1,2]
# scale = [8,16,32]

# w*h = area
# w/h = ratio
# w = math.sqrt(scale*area*ratio)
# h = math.sqrt(area/scale*ratio)

def _anchor2whc(anchor):
    if anchor.size != 4:
        return

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + (w - 1) / 2
    y_ctr = anchor[1] + (h - 1) / 2
    return w, h, x_ctr, y_ctr


def _whc2anchors(ws, hs, x_ctr, y_ctr):
    # np.newaxis 增加维度
    ws = np.array(ws)[:, np.newaxis]
    hs = np.array(hs)[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _enum_ratio(anchor, ratios):
    w, h, x_ctr, y_ctr = _anchor2whc(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _whc2anchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _enum_scale(anchor, scales):
    w, h, x_ctr, y_ctr = _anchor2whc(anchor)
    ws = [w * i for i in scales]
    hs = [h * i for i in scales]
    anchors = _whc2anchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    """
    :return: anchor boxes: format (x1, y1, x2, y2).
    """
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1])
    ratio_anchors = _enum_ratio(base_anchor, ratios)
    anchors = np.vstack([_enum_scale(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def plot_box_s(box_s, c='k'):
    ax_min = box_s.min()
    ax_max = box_s.max()
    rect_color = ['b', 'g', 'r']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, box in enumerate(box_s):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        rect = patches.Rectangle(xy=(x1, y1), width=w, height=h, color=rect_color[i % 3], fill=False)
        ax.add_patch(rect)

    plt.xlim((ax_min-50, ax_max+50))
    plt.ylim((ax_min-50, ax_max+50))
    plt.show()


def map_anchor2image(image, feature_point, base_anchors):
    img_w = img.shape[1]
    img_h = img.shape[0]
    rect_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] #rect_color[i % 3]

    for point in feature_points:
        for i, anchor in enumerate(base_anchors):
            w = anchor[2] - anchor[0] + 1
            h = anchor[3] - anchor[1] + 1
            x_1 = max(0, point[0] - w / 2)
            y_1 = max(0, point[1] - h / 2)
            x_2 = min(img_w, point[0] + w / 2)
            y_2 = min(img_h, point[1] + h / 2)
            start_point = (x_1, y_1)
            end_point = (x_2, y_2)
            color = rect_color[i % 3]
            image = cv2.rectangle(image, start_point, end_point, color)

    cv2.imshow("anchors", image)


if __name__ == '__main__':
    # generate 9 anchors
    result = generate_anchors()
    plot_box_s(result)

    """""
    # map anchors in original image
    img = cv2.imread("dogs.jpg")
    # generate feature points
    img_w = img.shape[1]
    img_h = img.shape[0]
    feature_points = []
    for i in range(4):
        x = random.randint(1, img_w-1)
        y = random.randint(1, img_h-1)
        feature_points.append((x, y))
    map_anchor2image(img, feature_points, result)
    """""



