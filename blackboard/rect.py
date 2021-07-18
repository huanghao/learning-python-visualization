import sys
import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# https://segmentfault.com/a/1190000013925648
# https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/
# https://www.cnblogs.com/jclian91/p/9728488.html


def process(fname):
    origin = cv.imread(fname)

    small = origin.copy()
    while small.shape[0] > 1000:
        small = cv.pyrDown(small)

    binary = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
    hist1 = cv.calcHist([binary], [0], None, [256], [0, 256])

    blur = cv.medianBlur(binary, 3)
    norm = np.zeros_like(blur)
    cv.normalize(blur, norm, 50, 200, cv.NORM_MINMAX, cv.CV_8U)

    hist2 = cv.calcHist([norm], [0], None, [256], [0, 256])
    white = 150
    ret, thresh = cv.threshold(blur, white, 255, cv.THRESH_BINARY)

    if 0:
        plt.plot(hist1, color='r')
        plt.plot(hist2, color='g')
        plt.xlim([0, 256])
        plt.show()

    # https://www.docs.opencv.org/master/da/d22/tutorial_py_canny.html
    #canny = cv.Canny(small, 100, 100)

    # https://www.cnblogs.com/bjxqmy/p/12347265.html

    # https://docs.opencv.org/master/d3/d05/tutorial_py_table_of_contents_contours.html
    #im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST,
                                               cv.CHAIN_APPROX_SIMPLE)
    #im2, contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    cnts = np.zeros_like(small)
    cv.drawContours(cnts, contours, -1, (255, 0, 0), 3)

    def angle_cos(p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

    squares = []
    for cnt in contours:
        cnt_len = cv.arcLength(cnt, True)
        cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
        if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(
                cnt):
            m = cv.moments(cnt)
            cx = int(m['m10'] / m['m00'])
            cy = int(m['m01'] / m['m00'])
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([
                angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4])
                for i in range(4)
            ])
            if max_cos < 0.1:
                squares.append(cnt)

    if not squares:
        #return
        for cnt in contours:
            cv.polylines(small, [cnt], True, (255, 0, 0), 2)
        while 1:
            cv.imshow('small', small)
            cv.imshow('thresh', thresh)
            cv.imshow('contours', cnts)
            ch = cv.waitKey(0)
            if ch in (27, 113):
                break
        return

    print(squares)
    points1 = squares[0]

    cv.polylines(small, [points1], True, (255, 0, 0), 2)
    for i, p in enumerate(points1):
        cv.putText(small, str(i), tuple(p), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 0), 2)

    wh, WH = small.shape[:2], origin.shape[:2]
    points1 = np.float32([(x / wh[0] * WH[0], y / wh[1] * WH[1])
                          for (x, y) in points1])

    x, y = points1[:, 0], points1[:, 1]
    w, h = x.max() - x.min(), y.max() - y.min()
    dx, dy = x[1] - x[0], y[1] - y[0]
    if abs(dx) > abs(dy):
        if dx < 0:
            points2 = np.float32([(w, 0), (0, 0), (0, h), (w, h)])
        else:
            points2 = np.float32([(0, h), (w, h), (w, 0), (0, 0)])
    else:
        if dy < 0:
            points2 = np.float32([(w, h), (w, 0), (0, 0), (0, h)])
        else:
            points2 = np.float32([(0, 0), (0, h), (w, h), (w, 0)])

    # print(points1)
    # print(points2)

    m = cv.getPerspectiveTransform(points1, points2)
    processed = cv.warpPerspective(origin, m, (w, h))

    # https://blog.csdn.net/qq_40755643/article/details/84032773
    # im = cv.equalizeHist(processed)
    im = np.zeros_like(processed)
    cv.normalize(processed, im, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    im = cv.filter2D(im, -1, kernel)

    def imwrite(fname, cat, im):
        output_fname = f'{fname}.{cat}.jpeg'
        cv.imwrite(output_fname, im)
        return output_fname

    # out_fname = fname + '.my.jpeg'
    # cv.imwrite(out_fname, im)
    # return out_fname
    imwrite(fname, 'small', small)
    imwrite(fname, 'thresh', thresh)
    imwrite(fname, 'contours', cnts)
    imwrite(fname, 'result', im)

    while 1:
        cv.imshow('small', small)
        cv.imshow('thresh', thresh)
        cv.imshow('contours', cnts)
        cv.imshow('result', im)
        ch = cv.waitKey(0)
        if ch in (27, 113):
            break


def main():
    for fname in sys.argv[1:]:
        print(fname, process(fname))


if __name__ == '__main__':
    main()