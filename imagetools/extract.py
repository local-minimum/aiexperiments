from glob import glob
from pathlib import Path

import cv2
import numpy as np
from skimage import filters


def loader(files):
    for path in sorted(glob(str(files))):
        yield (cv2.imread(path), path)


def togray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def toedges(gray):
    return filters.sobel(gray.astype(float))


def get_first_crop_threshold(edges, bins, lim):
    _, xborders = np.histogram(np.mean(edges, axis=0), bins=bins)
    _, yborders = np.histogram(np.mean(edges, axis=1), bins=bins)
    return min(max(xborders[1], yborders[1]), lim)


def crop(image, edges, target, threshold=None):
    y1 = 0
    x1 = 0
    y2, x2 = edges.shape
    sy, sx = edges.shape
    targetx, targety = target
    if threshold is None:
        threshold = edges.max()
    while sx != targetx or sy != targety:
        s = edges[y1:y2, x1:x2]
        xbar = np.mean(s, axis=0)
        ybar = np.mean(s, axis=1)
        doney = sy == targety
        dox = sx > targetx
        x1v = xbar[0]
        x2v = xbar[-1]
        y1v = ybar[0]
        y2v = ybar[-1]
        if dox and x1v < x2v and (doney or x1v < y1v and x1v < y2v):
            if x1v > threshold:
                break
            x1 += 1
            sx -= 1
        elif dox and (doney or x2v < y1v and x2v < y2v):
            if x2v > threshold:
                break
            x2 -= 1
            sx -= 1
        elif y1v < y2v:
            if y1v > threshold:
                break
            y1 += 1
            sy -= 1
        else:
            if y2v > threshold:
                break
            y2 -= 1
            sy -= 1

    return image[y1: y2, x1: x2]


def scale(image, target):
    y, x = image.shape[:2]
    tx, ty = target
    s = max(ty / y, tx / x)
    return cv2.resize(image, (max(int(x * s), tx), max(int(y * s), ty)))


def crop_scale_crop(image, target=(128, 128), f1=50, f2=2):
    gray = togray(image)
    edges = toedges(gray)
    threshold = get_first_crop_threshold(gray, f1, f2)
    image = crop(image, edges, target, threshold=threshold)
    print('Cropped: {}'.format(image.shape[:2]))
    image = scale(image, target)
    print('Scaled: {}'.format(image.shape[:2]))
    gray = togray(image)
    edges = toedges(gray)
    image = crop(image, edges, target)
    print('Cropped: {}'.format(image.shape[:2]))
    return image


def extract(path, target=(128, 128), f1=50, f2=2, outprefix='crop'):
    i = 0
    for image, p in loader(path):
        print("\n** {}".format(p))
        if image is None or image.size == 0:
            print("Couldn't read")
            continue
        sy, sx = image.shape[:2]
        if sy < target[1] or sx < target[0]:
            print("Too small")
            continue
        small = crop_scale_crop(image, target=target, f1=f1, f2=f2)
        out = Path(p).parent / ('crop' + str(i).zfill(6) + '.jpg')
        print("Saved as {}".format(out))
        cv2.imwrite(str(out), small)
        i += 1
