from cv2 import *
import numpy as np
from os import listdir
from os.path import isfile, join


def adjust_contrast(img):
    """
    Adjust constrast by computing histogram equalization
    (https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
    :param img: input image
    :return: enhanced image
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img)
    cv2.imwrite("results/1_cv_equalized.bmp", equalized)

    return equalized


def compute_mser(img):
    # Create MSER object
    mser = cv2.MSER_create()

    # Convert to gray scale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # detect regions in gray scale image
    regions, _ = mser.detectRegions(img)

    mask = np.zeros(img.shape, dtype=np.uint8)
    for points in regions:
        for point in points:
            mask[point[1], point[0]] = 255

    cv2.imwrite("results/2_mser_mask.bmp", mask)

    return mask


def readImg(data_path, filename):
    # Retrieve all images in data_path folder
    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    if files.__len__() == 0:
        print("cannot find images in \'", data_path, "\' folder")
        return None

    # Pick only the first
    img = None
    for f in files:
        if f.lower().__contains__(filename):
            img = cv2.imread(join(data_path, f))
            break

    return img


def displayImg(img, title=""):
    cv2.namedWindow(title)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detectText(data_path, filename):
    print(" -*- Text Detection with edge-enhanced MSER -*-")

    img = readImg(data_path, filename)
    assert img is not None, 'Cannot read the image!'
    displayImg(img, "Original image")

    # Convert to grayscale
    grey = cvtColor(img, COLOR_BGR2GRAY)

    # Linear adjust image intensities
    adjusted_image = adjust_contrast(grey)
    print("... image contrast adjusted")
    displayImg(adjusted_image, "Image with adjusted contrast")

    # Compute MSER
    mser_mask = compute_mser(adjusted_image)
    print("... MSER Mask computed")

    # Compute canny
    edges = cv2.Canny(grey, 60, 60 * 3, L2gradient=True)
    imwrite("results/3_canny.bmp", edges)
    print("... Canny computed")
    displayImg(edges, "Canny")

    # Create the edge enhanced MSER region
    edge_mser_intersection = edges & mser_mask
    cv2.imwrite("results/4_intersetion.bmp", edge_mser_intersection)
    print("... MSER + Canny")
    displayImg(edge_mser_intersection, "Canny & MSER")

    # Grow edges along gradient direction (erode + dilate)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_enhanced_mser = cv2.morphologyEx(edge_mser_intersection, cv2.MORPH_GRADIENT, kernel)
    cv2.imwrite("results/5_edge_enhanced_mser.bmp", edge_enhanced_mser)
    print("... Grown edges")
    displayImg(edge_enhanced_mser, "Enhanced Edges")

    # Find connected components
    enhanced_wlabels = np.zeros(edge_enhanced_mser.shape, np.uint8)
    stats = None
    ncc, enhanced_wlabels, stats, _ = cv2.connectedComponentsWithStats(edge_enhanced_mser)
    imwrite("results/6_labels.bmp", enhanced_wlabels)
    print("... ", ncc, " Connected components")

    # Geometric filtering
    # TODO
    # filtered = np.zeros(enhanced_wlabels.shape, dtype=np.uint8)
    # for i in range(ncc):
    #     if stats[i, cv2.CC_STAT_AREA] < 75 or stats[i, cv2.CC_STAT_AREA] > 600:
    #         continue
    #
    #     ratio = stats[i, cv2.CC_STAT_WIDTH] / stats[i, cv2.CC_STAT_HEIGHT]
    #     if ratio < 0.001 or ratio > 2:
    #         continue
    #
    #     filtered = stats[i]
    # imwrite("results/filtered_cc.bmp", filtered)
