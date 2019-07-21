import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def image_crop(image):
    row_center = image.shape[0]/2
    col_center = image.shape[1]/2
    img_crop = image[0:row_center, 0:col_center]
    return  img_crop;

def random_image_color_sift(img):
    dst_img = img.clone()
    B, G, R = cv2.split(dst_img)

    for i in (B,G,R):
        rand = random.randint(-50,50)
        if rand == 0:
            pass
        elif rand > 0:
            lim = 255-rand
            i[i > lim] = 255
            i[i <= lim] = (rand + i[i <= lim]).astype(img.dtype)
        elif rand < 0:
            lim = 0-rand
            i[i < lim] = 0
            i[i >= lim] = (rand + i[i >= lim]).astype(img.dtype)
    return dst_img

def image_rotation(image):
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 30, 1)  # center, angle, scale
    img_rotate = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return imh_rotate

def random_image_warp(img):
    height, width, channels = img.shape

    # warp:
    random_margin = 60
    x1 = random.randint(-random_margin, random_margin)
    y1 = random.randint(-random_margin, random_margin)
    x2 = random.randint(width - random_margin - 1, width - 1)
    y2 = random.randint(-random_margin, random_margin)
    x3 = random.randint(width - random_margin - 1, width - 1)
    y3 = random.randint(height - random_margin - 1, height - 1)
    x4 = random.randint(-random_margin, random_margin)
    y4 = random.randint(height - random_margin - 1, height - 1)

    dx1 = random.randint(-random_margin, random_margin)
    dy1 = random.randint(-random_margin, random_margin)
    dx2 = random.randint(width - random_margin - 1, width - 1)
    dy2 = random.randint(-random_margin, random_margin)
    dx3 = random.randint(width - random_margin - 1, width - 1)
    dy3 = random.randint(height - random_margin - 1, height - 1)
    dx4 = random.randint(-random_margin, random_margin)
    dy4 = random.randint(height - random_margin - 1, height - 1)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return img_warp

if __name__ == "__main__":
    img = read("lenna.jpg")
    transforms = [image_crop, random_image_color_sift, image_rotation, random_image_warp]
    print("1.image_crop")
    print("2.random_image_color_sift")
    print("3.image_rotation")
    print("4.random_image_warp")
    n = input("选择要执行的函数：")
    if n > 4 or n < 1:
        pass
    transforms[n](img)
