from PIL import Image
import albumentations as A
import cv2 as cv
import numpy as np

import math
import matplotlib.pyplot as plt
from pathlib import Path
import os

# import pdb
# pdb.set_trace()

# --------------- Support Functions --------------------------------------------------------


def crop_base(height, width):
    transformations = [A.Crop(always_apply=True, p=1.0, x_min=0,
                              y_min=0, x_max=width, y_max=height - 70), ]
    return transformations


def neighbors(a, radius, rowNumber, columnNumber):
    return [[a[i][j] if i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else 0
             for j in range(columnNumber - radius, columnNumber + 1 + radius)]
            for i in range(rowNumber - radius, rowNumber + 1 + radius)]


def pilim(image):
    # In some computers CV.imwrite does not work
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image)
    return im_pil


# --------------- Main Function ----------------------------------------------------------
def cv_img(path, root_images, custom_threshold=100):
    name = os.path.splitext(path)[0]
    Path(name + "/").mkdir(parents=True, exist_ok=True)

    # read img
    img = cv.imread(root_images + "/" + path, 0)
    height, width = img.shape
    print(f" {name} image height: {height} and width: {width} and Threshold {custom_threshold}")
    # Crop to eliminate base
    transforms = crop_base(height=int(height), width=int(width))
    transforms = A.Compose(transforms)
    transformed = transforms(image=img)
    crop_image = transformed["image"]
    # -------------------------------------------------------------- #
    # Normalize
    crop_image = cv.normalize(crop_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)  # noqa: E501
    # Try median filter
    median_img = cv.medianBlur(crop_image, 5)  # Add median filter to image
    # pilim(median_img).save("median_filter.tif")
    cv.imwrite(name + "/" + "median_filter.jpg", median_img)

    # -------------------------------------------------------------- #
    # - (B) -  Binarized image
    # -------------------------------------------------------------- #
    # Otsu
    Threshold = custom_threshold
    ret, th1 = cv.threshold(median_img, Threshold, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)  # noqa: E501
    # try adaptive threshold
    th2 = cv.adaptiveThreshold(median_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)  # noqa: E501
    th3 = cv.adaptiveThreshold(median_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)  # noqa: E501

    titles = ['Original', "THRESH_BINARY",  'Adaptive Mean Thresh', 'Adaptive Gaussian Thresh']    # noqa: E501
    images = [crop_image, th1, th2, th3]
    for count, im in enumerate(images):
        cv.imwrite(name + "/" + titles[count] + ".jpg", im)
        # pilim(im).save(titles[count] + ".tif")

    # -------------------------------------------------------------- #
    # - (C) -- Overlay binarized onto original-----
    # -------------------------------------------------------------- #
    # setting alpha=1, beta=1, gamma=0 gives direct overlay of two images
    alpha = 1
    beta = 1
    gamma = 0
    overlay = cv.addWeighted(th1, alpha, median_img, beta, gamma)
    cv.imwrite(name + "/" + 'overlay.jpg', overlay)

    # -------------------------------------------------------------- #
    # -(D) -- Euclidean Distance from B -> encode into gray values -
    # -------------------------------------------------------------- #
    # invert
    inv_th1 = cv.bitwise_not(th1)
    # Perform the distance transform algorithm
    dist = cv.distanceTransform(inv_th1, cv.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    dist_transform = cv.normalize(dist, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)  # noqa: E501
    cv.imwrite(name + "/" + 'Distance Transform.jpg', dist_transform)

    # Euclidean distance is L2
    # -------------------------------------------------------------- #
    # (E) identify local maxima in (d) -> fir circles into (b)
    # -------------------------------------------------------------- #
    bin_circles = cv.cvtColor(th1, cv.COLOR_GRAY2RGB)
    rad_list = []
    # Can also use while to color everything
    for counts in range(2500):
        maxDT = np.unravel_index(dist_transform.argmax(), dist_transform.shape)  # noqa: E501

        center_coordinates = (maxDT[1], maxDT[0])

        for x in range(1, 50):
            value = neighbors(dist_transform, x, maxDT[0], maxDT[1])
            value = np.array(value)
            if value.min() == 0:
                # First black pixel found
                minval = np.unravel_index(value.argmin(), value.shape)  # noqa: E501
                # Center
                maxval = np.unravel_index(value.argmax(), value.shape)  # noqa: E501
                break

        # radius of circle is distance to point
        eDistance = int(math.dist(maxval, minval))
        # save radius of circles
        rad_list.append(int(eDistance))
        # Draw circles
        if eDistance > 15:
            color = (0, 0, 100)  # Red ing BGR
        elif eDistance > 10:
            color = (100, 0, 0)  # Blue
        else:
            color = (0, 100, 0)  # Green
        thickness = -1  # full circles
        # over the binary image
        bin_circles = cv.circle(bin_circles, center_coordinates, eDistance, color, thickness)  # noqa: E501
        # prevents from taking same location as before taking the distance of the TH with circles
        dist_transform = cv.circle(dist_transform, center_coordinates, eDistance, 0, thickness)  # noqa: E501
        dist_transform = cv.distanceTransform(dist_transform, cv.DIST_L2, 3)  # noqa: E501
        dist_transform = cv.normalize(dist_transform, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)  # noqa: E501

    print(f" {name} total number of circles created: {len(rad_list)}")
    with open(name + "/" + 'radius.txt', 'w') as f:
        for item in rad_list:
            f.write("%s\n" % item)
    cv.imwrite(name + "/" + 'bin_circles.jpg', bin_circles)
    cv.imwrite(name + "/" + 'support_bin_circles.jpg', dist_transform)

    # -------------------------------------------------------------- #
    # (F) plot the size distribution that is obtained from the diameters of the fitted circles in (e)  # noqa: E501
    # -------------------------------------------------------------- #
    plt.hist(rad_list, bins=23, range=(0, 25), density=False)
    plt.ylabel('counts')
    plt.xlabel('radius')
    plt.title("rad of cirlces")
    plt.savefig(name + "/" + "histo_rad_circles")

    # -------------------------------------------------------------- #
    # (G) shows a brightness histogram of (a).
    # -------------------------------------------------------------- #
    # density=False would make counts
    plt.hist(median_img.flat, bins=225, range=(0, 255), density=False)
    plt.axvline(Threshold, color='r', linewidth=2)
    plt.ylabel('Counts')
    plt.xlabel('Pixel intensity')
    plt.title("Histogram median filder")
    plt.savefig(name + "/" + "histogram median_filter")


if __name__ == "__main__":
    # folder with all raw images
    root_images = "./root_images"
    # includes all .tif files
    files = [f for f in os.listdir(root_images) if f.endswith('.tif')]
    print(files)
    # CHANGE Threshold based on Histogram_median_filter   <----- change
    # In alphabetical order [transparent sample, white sample]
    Threshold = [45, 100]
    for count, image in enumerate(files):
        # goes through all images in root folder
        cv_img(image, root_images, Threshold[count])
