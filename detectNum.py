#**********************************************************************************************************************#
#                                     OPTICAL CHARACTER RECOGNITION : DETECT NUMBER                                    #
#**********************************************************************************************************************#

#************************************************# IMPORT PACKAGES #***************************************************#
import sys
import os
from collections import Counter
import numpy as np
import cv2
from skimage.feature import hog
from joblib import load
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
#**********************************************************************************************************************#

#**********************************************# GLOBAL PARAMETERS #***************************************************#
RATIO_CONTOURS_THRESHOLD = 0.4
PAPER_DETECTION_THRESHOLD = 0.35
DIGIT_IMG_SIZE = (28, 44)
CROPPING_SIZE_X, CROPPING_SIZE_Y = .4, .4
#**********************************************************************************************************************#

#****************************************************# MODELS #********************************************************#
    # KNN CLASSIFIER (DIGIT CLASSIFIER)
knn = load('../Models/knn_digit_classifier.joblib')

    # RETINANET (PAPER DETECTION)
retinaNet = models.load_model('../Models/retinanet_paper_detector.h5')
retinaNet = models.convert_model(retinaNet)
#**********************************************************************************************************************#


#************************************************# HELPER FUNCTIONS #**************************************************#
def extract_papers(img, boxes):
    """
    This function extracts rectangle from image.
    """
    papers = []
    for box in boxes:
        box = np.int32(box)
        paper = img[box[1]:box[3], box[0]:box[2], :]
        papers.append(paper)
    return papers

def crop_images(imgs, wp, hp):
    """
    This function crops image to keep only the centre for identification.
    Example (width(w),height(h))=100x75 with crop factor of wp=0.4 and hp=.33,
    results in image centre size of 40x25
    """
    cropped_imgs = []
    for img in imgs:
        h, w = img.shape[:2]
        startRow = int(h*(1-hp)/2)
        startCol = int(w*(1-wp)/2)
        endRow = startRow + int(h*hp)
        endCol = startCol + int(w*wp)
        img_cropped = img[startRow:endRow, startCol:endCol]
        cropped_imgs.append(img_cropped)
    return cropped_imgs

def get_digits_contours(contours):
    """
    This function identifies and separate contours that represent digits from  other
    noise contours in the image.
    """
    out = []    # List of tuples, where tuple = (volume of bounding rectangle of contour, contour)
    if contours == []:
        return []

    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        out.append((w*h, contour))

    if len(out) == 1:
        return [out[0][1]]

    out = sorted(out, key=lambda tup: tup[0], reverse=True) # Sort contours by bounding rectangle volume (highest --> lowest)
    if out[1][0] / out[0][0] >= RATIO_CONTOURS_THRESHOLD:   # compare bounding rectangle volumes of first two contours to infer if both are digits or just the first one
        return [out[0][1], out[1][1]]
    else:
        return [out[0][1]]
#**********************************************************************************************************************#


#**********************************************# GET NUMBER FUNCTION #*************************************************#
def get_number(img, multiple=False):
    """
    This function accepts an image as an input and returns the number on the paper held by the person.
    """
    # INPUT IMAGE PRE-PROCESSING
    pp_img = preprocess_image(img)
    pp_img, scale = resize_image(pp_img)

    # DETECT PAPER IN INPUT IMAGE
    boxes, scores, labels = retinaNet.predict_on_batch(np.expand_dims(pp_img, axis=0))
    if multiple:  # if image has more than one person, accept all detected boxes with score higher than threshold
        detected_boxes = boxes[0, scores[0] >= PAPER_DETECTION_THRESHOLD]/scale
    else:  # if not, keep only the box with the highest score (boxes are already ordered)
        detected_boxes = np.array([boxes[0, 0]/scale])
    # If no boxes detected, exit function with special return code
    if(detected_boxes.tolist() == []):
        print('\nNO PAPER DETECTED!\n')
        return [-1]

    # EXTRACT PAPER FROM INPUT IMAGE
    detected_papers = extract_papers(img, detected_boxes)

    # CROP PAPER (EXTRACT SMALLER BOX AROUND NUMBER)
    cropped_papers = crop_images(detected_papers, CROPPING_SIZE_X, CROPPING_SIZE_Y)

    # BINARIZE CROPPED PAPER (Gaussian Blur + OTSU Thresholding)
        # GRAY SCALING
    papers_bw = [cv2.cvtColor(cropped_paper, cv2.COLOR_RGB2GRAY) for cropped_paper in cropped_papers]
        # BLURRING (Gaussian)
    papers_bw_blur = [cv2.GaussianBlur(paper_bw, (3,3), 0) for paper_bw in papers_bw]
        # BINARIZING (OTSU THRESHOLDING)
    papers_binary = [cv2.threshold(paper_bw_blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] for paper_bw_blur in papers_bw_blur]

    # FIND EXTERNAL CONTOURS FOR EACH DIGIT
        # DETECT CANNY EDGES
    papers_edged = [cv2.Canny(paper_bw_blur, 30, 70) for paper_bw_blur in papers_bw_blur]
        # FIND CONTOURS FROM EDGES
    papers_contours = [cv2.findContours(paper_edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] for paper_edged in papers_edged]
        # FILTER OUT CONTOURS NOT REPRESENTING DIGITS
    papers_contours = [get_digits_contours(contours) for contours in papers_contours]

    # ISOLATE EACH DIGIT IMAGE (RESIZED TO STANDARDIZED SHAPE)
    papers_digits = []
    for ix, paper_contours in enumerate(papers_contours):
        paper_digits = {}  # key:value == x_position:digit_img
        for contour in paper_contours:
            x, y, w, h = cv2.boundingRect(contour)
            paper_digit = papers_binary[ix][y:y+h, x:x+w]
            paper_digit_resized = cv2.resize(paper_digit, DIGIT_IMG_SIZE)
            paper_digits[x] = paper_digit_resized
        papers_digits.append(paper_digits)

    if papers_digits == [{}]:
        print('\nNo digits detected !\n')
        return [-1]

    # ORDER DIGITS BY X POSITION (needed to compose the final number from the digits)
    papers_digits = [[digit for pos, digit in sorted(paper_digits.items())] for paper_digits in papers_digits]



    # PREDICT DIGITS FROM HOG FEATURES & RECONSTRUCT THE FINAL NUMBER
    numbers = []
    for paper_digits in papers_digits:
        # CREATE HOG FEATURES FOR EACH DIGIT
        list_hog_digits = [hog(digit, block_norm='L2-Hys') for digit in paper_digits]
        digits = knn.predict(list_hog_digits)
        number = sum([digit*pow(10, power) for power, digit in enumerate(digits[::-1])])
        numbers.append(number)

    return numbers
#**********************************************************************************************************************#



if __name__ == '__main__':

    if len(sys.argv) == 2:
        multiple = False
    elif len(sys.argv) == 3:
        multiple = sys.argv[2]
    else :
        raise ValueError("Arguments not found !")

    file_path = sys.argv[1]
    print('\nReading File', file_path)

    if file_path.endswith('.jpg') or file_path.endswith('.JPG'):
        file_type = 'image'
        print('Detected Image File!')
        input_image = cv2.imread(file_path)
        num = get_number(input_image, multiple=multiple)

    elif file_path.endswith('.MP4') or file_path.endswith('.mp4'):
        file_type = 'video'
        print('Detected Video File!')
        vidcap = cv2.VideoCapture(file_path)
        success, image = vidcap.read()
        count = 0
        success = True
        sample_images = [] # to extract some frames from video
        while success:
            success, image = vidcap.read()
            if file_path.endswith('.mov'): # issue with .mov videos, need to be rotated 90 degres clockwise
                image = cv2.rotate(image, 0)
            if ((count+1)%3 == 0) & success:
                sample_images.append(image)
            if len(sample_images) > 30: # if video is too long stop after sampling 30 frames
                break
            count += 1
        # GET NUMBER FOR EACH SAMPLED IMAGE
        nums = []
        for img in sample_images:
            nums.append(get_number(img))

        nums = [tuple(elm) for elm in nums]
        counter = Counter(nums)
        num = list(counter.most_common()[0][0])

    if num != [-1]:
        print('\n************ The number(s) found on the {} is(are) : {} ************\n'.format(file_type, num))
