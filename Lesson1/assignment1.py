import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# lists is a list. lists[0:4]: x1, x2, y1, y2; lists[4]: score
def NMS(lists, thre):

    # parse
    x1 = lists[0]
    x2 = lists[1]
    y1 = lists[2]
    y2 = lists[3]
    areas = (y2-y1+1) * (x2-x1+1)
    scores = lists[4, :]
    result = [[]] * 5

    # sort
    index = np.argsort(-scores)

    # suppression
    while index.size > 0:

        # the first one has highest score, keep it
        i = index[0]
        result = np.column_stack((result, lists[:, i]))

        # calculate the area of overlap
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        overlaps = w*h

        # calculate the iou, intersection / union
        IOUs = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        keep_index = np.where(IOUs <= thre)[0]
        index = index[keep_index+1]

    return result


def plot_box_s(box_s, c='k'):
    x1 = box_s[0]
    x2 = box_s[1]
    y1 = box_s[2]
    y2 = box_s[3]

    plt.plot([x1, x2], [y1, y1], c)
    plt.plot([x1, x1], [y1, y2], c)
    plt.plot([x1, x2], [y2, y2], c)
    plt.plot([x2, x2], [y1, y2], c)


if __name__ == '__main__':
    boxes = np.array([[100, 250, 220, 100, 230, 220],
                      [210, 420, 320, 210, 325, 315],
                      [100, 250, 220, 100, 240, 230],
                      [210, 420, 330, 210, 330, 340],
                      [0.72, 0.8, 0.92, 0.72, 0.81, 0.9]])

    plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax1)
    plot_box_s(boxes, 'k')

    lists = NMS(boxes, 0.7)
    plt.sca(ax2)
    plot_box_s(lists, 'r')
    plt.show()
