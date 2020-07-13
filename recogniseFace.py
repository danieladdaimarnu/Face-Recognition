#**********************************************************************************************************************#
#                                                     RECOGNISE FACES                                                  #
#**********************************************************************************************************************#

#***************************************************# IMPORT PACKAGES #************************************************#
import numpy as np
import os
import sys
import pickle
import cv2
from scipy import spatial

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from skimage import feature
from joblib import load
from utilities import *

#***************************************************# LOAD MODELS #****************************************************#
    # LOAD YOLO FACE DETECTOR
NET_PATH = '../Models/YOLO_v3.weights'
NET_CONF_PATH = '../Models/YOLO_v3_Config.cfg'
net = cv2.dnn.readNetFromDarknet(NET_CONF_PATH, NET_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # LOAD LR SURF MODEL
LR_surf = load('../Models/lr_surf.joblib')
    # LOAD SVM SURF MODEL
SVM_surf = load('../Models/svm_surf.joblib')
    # LOAD RANDOM FOREST SURF MODEL
RF_surf = load('../Models/rf_surf.joblib')
    # LOAD LR SIFT-LBP MODEL
LR_sift = load('../Models/lr_sift.joblib')
    # LOAD SVM SIFT-LBP MODEL
SVM_sift = load('../Models/svm_sift.joblib')
    # LOAD RANDOM FOREST SIFT-LBP MODEL
RF_sift = load('../Models/rf_sift.joblib')

    # LOAD RESNET
RESNET_PATH = '../Models/ResNet50_adam_sam_weights.hdf5'
        # BUILD MODEL
base_model = keras.applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
# ADD 2 MORE DENSE LAYERS WITH DROPOUT
for i in range(2):
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(rate=0.5))
top_model.add(Dense(48, activation='softmax'))  # 48 is the number of classes
resnet = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        # LOAD WEIGHTS
resnet.load_weights(RESNET_PATH)
        # LOAD INDEX-LABEL MAPPING DICTIONARY
with open('../Resources/index_to_label.pickle', 'rb') as handle:
    ix_to_label = pickle.load(handle)

# LOADING CODEBOOKS
    # LOAD SURF CODEBOOK
vocab_surf = np.load('../Resources/vocab_surf_new_100.npy')
    # LOAD SIFT-LBP CODEBOOK
vocab_sift = np.load('../Resources/vocab_sift_new_100.npy')

#**********************************************************************************************************************#

#***************************************************# HELPER FUNCTIONS #***********************************************#
# SURF HELPER FUNCTION
def get_surf_bovw(img, vocab):
    """This function returns the SURF feature vector of an input image."""
    #bovw = np.zeros(800) # Intialize vector representation of visual words
    bovw = np.zeros(len(vocab))  # Initialise vector representation of visual words
    surf = cv2.xfeatures2d.SURF_create()
    kp, descriptors = surf.detectAndCompute(img, None)
    if descriptors is None:
        return bovw
    else:
        for feature_vector in descriptors:
            # Get Index of Nearest Neighbor of feature_vector from visual vocabulary
            fv_nn_ix = spatial.KDTree(vocab).query(feature_vector)[1]
            # Update Visual Words dictionary
            bovw[fv_nn_ix] += 1
        return bovw

# KEYPOINT NEIGHBORHOOD FUNCTION
def get_patch(img_gray, keyPoint, size=(7,7)):
    """Function to compute a patch from an image around a key point with a window size. 
    This basically returns a neighborhood around the keypoint"""
    h, w = img_gray.shape
    x0, y0 = (int(keyPoint[0]), int(keyPoint[1]))
    sx, sy = size
    hsx = int(sx/2)
    hsy = int(sy/2)
    if x0 + hsx > h - 1 : x0 = h - 1 - hsx
    elif x0 - hsx < 0 : x0 = hsx
    if y0 + hsy > w - 1 : y0 = w - 1 - hsy
    elif y0 - hsy < 0 : y0 = hsy
    return img_gray[x0-hsx:x0+hsx+1, y0-hsy:y0+hsy+1]

# SIFT HELPER FUNCTION
def get_sift_bovw(img, vocab):
    """This function returns the SURF feature vector of an input image."""
    #bovw = np.zeros(128)
    bovw = np.zeros(len(vocab))  # Initialise vector representation of visual words
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        return bovw
    else:
        for feature_vector in descriptors:
            # Get Index of Nearest Neighbor of feature_vector from visual vocabulary
            fv_nn_ix = spatial.KDTree(vocab).query(feature_vector)[1]
            # Update Visual Words dictionary
            bovw[fv_nn_ix] += 1
        return bovw

#**********************************************************************************************************************#

#*********************************************# RECOGNISE FACE FUNCTION #**********************************************#

def recogniseFace(I, featureType, classifierType, creativeMode):
    """
    This function performs the Face Recognition
    Feature_extractor : 'SURF', 'SIFT', 'none'
    Algorithm : 'LR', 'SVM', 'RF', 'RESNET', 'none'
    CreativeMode: 0, 1
    """

    # SET INPUT TO THE NETWORK
    blob = cv2.dnn.blobFromImage(I, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)  # IMG_WIDTH and IMG_HEIGHT are in the configuration file of YOLO
    net.setInput(blob)

    # RUN FORWARD PASS
    outs = net.forward(get_outputs_names(net))

    # REMOVE BOUNDING BOXES WITH LOW CONFIDENCE
    faces = post_process(I, outs, CONF_THRESHOLD, NMS_THRESHOLD)


    persons = np.zeros((len(faces), 3), dtype=int)  # matrix to store the detected individuals

    if (featureType == 'SURF') and (classifierType == 'LR') and (creativeMode == str(0)):
        # ITERATE OVER EACH DETECTED FACE, EXTRACT FEATURES IF APPROPRIATE, THEN PREDICT
        for num, face in enumerate(faces):
            x, y, w, h = face
            x = max(0, x)
            y = max(0, y)
            img_face = I[y:y + h, x:x + w]
            #print(num, img_face)
            img_face = cv2.resize(img_face, (80, 80))
            fv = get_surf_bovw(img_face, vocab_surf)
            pred = LR_surf.predict(fv.reshape(1, -1))
            persons[num] = np.array((pred[0], x + w / 2, y + h / 2))
        return persons

    elif (featureType == 'SIFT') and (classifierType == 'LR') and (creativeMode == str(0)):
        # ITERATE OVER EACH DETECTED FACE, EXTRACT FEATURES IF APPROPRIATE, THEN PREDICT
        for num, face in enumerate(faces):
            x, y, w, h = face
            x = max(0, x)
            y = max(0, y)
            img_face = I[y:y + h, x:x + w]
            # print(num, img_face)
            img_face = cv2.resize(img_face, (80, 80))
            fv = get_sift_bovw(img_face, vocab_sift)
            pred = LR_sift.predict(fv.reshape(1, -1))
            persons[num] = np.array((pred[0], x + w / 2, y + h / 2))
        return persons

    elif (featureType == 'SURF') and (classifierType == 'SVM') and (creativeMode == str(0)):
        # ITERATE OVER EACH DETECTED FACE, EXTRACT FEATURES IF APPROPRIATE, THEN PREDICT
        for num, face in enumerate(faces):
            x, y, w, h = face
            x = max(0, x)
            y = max(0, y)
            img_face = I[y:y + h, x:x + w]
            # print(num, img_face)
            img_face = cv2.resize(img_face, (80, 80))
            fv = get_surf_bovw(img_face, vocab_surf)
            pred = SVM_surf.predict(fv.reshape(1, -1))
            persons[num] = np.array((pred[0], x + w / 2, y + h / 2))
        return persons
        
    elif (featureType == 'SIFT') and (classifierType == 'SVM') and (creativeMode == str(0)):
        # ITERATE OVER EACH DETECTED FACE, EXTRACT FEATURES IF APPROPRIATE, THEN PREDICT
        for num, face in enumerate(faces):
            x, y, w, h = face
            x = max(0, x)
            y = max(0, y)
            img_face = I[y:y + h, x:x + w]
            # print(num, img_face)
            img_face = cv2.resize(img_face, (80, 80))
            fv = get_sift_bovw(img_face, vocab_sift)
            pred = SVM_sift.predict(fv.reshape(1, -1))
            persons[num] = np.array((pred[0], x + w / 2, y + h / 2))
        return persons
            
    elif (featureType == 'SURF') and (classifierType == 'RF') and (creativeMode == str(0)):
        # ITERATE OVER EACH DETECTED FACE, EXTRACT FEATURES IF APPROPRIATE, THEN PREDICT
        for num, face in enumerate(faces):
            x, y, w, h = face
            x = max(0, x)
            y = max(0, y)
            img_face = I[y:y + h, x:x + w]
            # print(num, img_face)
            img_face = cv2.resize(img_face, (80, 80))
            fv = get_surf_bovw(img_face, vocab_surf)
            pred = RF_surf.predict(fv.reshape(1, -1))
            persons[num] = np.array((pred[0], x + w / 2, y + h / 2))
        return persons
        
    elif (featureType == 'SIFT') and (classifierType == 'RF') and (creativeMode == str(0)):
        # ITERATE OVER EACH DETECTED FACE, EXTRACT FEATURES IF APPROPRIATE, THEN PREDICT
        for num, face in enumerate(faces):
            x, y, w, h = face
            x = max(0, x)
            y = max(0, y)
            img_face = I[y:y + h, x:x + w]
            # print(num, img_face)
            img_face = cv2.resize(img_face, (80, 80))
            fv = get_sift_bovw(img_face, vocab_sift)
            pred = RF_sift.predict(fv.reshape(1, -1))
            persons[num] = np.array((pred[0], x + w / 2, y + h / 2))
        return persons
        
    elif (featureType == 'none') and (classifierType == 'RESNET') and (creativeMode == str(0)):
        # ITERATE OVER EACH DETECTED FACE, EXTRACT FEATURES IF APPROPRIATE, THEN PREDICT
        for num, face in enumerate(faces):
            x, y, w, h = face
            x = max(0, x)
            y = max(0, y)
            img_face = I[y:y + h, x:x + w]
            #print(num, img_face)
            img_face = cv2.resize(img_face, (80, 80))
            img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
            img_face = cv2.resize(img_face, (224, 224))

            img_face = image.img_to_array(img_face)
            img_face = np.expand_dims(img_face, axis=0)
            img_face = preprocess_input(img_face)

            #img_face = img_face[None, :, :, :]/255
            pred_ix = resnet.predict(img_face).argmax()
            #print(str(pred_ix))
            pred = [int(ix_to_label[pred_ix])]
            persons[num] = np.array((pred[0], x + w / 2, y + h / 2))
        return persons

    elif (featureType == 'none') and (classifierType == 'none') and (creativeMode == str(1)):
        face_mask = cv2.imread('../Resources/mask.jpg', -1)
        face_mask = face_mask[:, :, 0:3]
        frame = I

        # Iterate over each detected face:
        for num, face in enumerate(faces):
            x, y, w, h = face

            if h > 0 and w > 0:
                x = int(x - 0.27*w)
                y = int(y - 0.0*h)
                w = int(1.4 * w)
                h = int(1.3 * h)

                frame_roi = frame[y:y + h, x:x + w]  # Extract the region of interest from the image
                face_mask_small = cv2.resize(face_mask, (w, h))

                # Convert color image to grayscale and threshold it
                gray_mask = cv2.cvtColor(face_mask_small, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(gray_mask, 230, 255, cv2.THRESH_BINARY_INV)

                # Create an inverse mask
                mask_inv = cv2.bitwise_not(mask)

                # Use the mask to extract the face mask region of interest
                masked_face = cv2.bitwise_and(face_mask_small, face_mask_small, mask=mask)

                # Use the inverse mask to get the remaining part of the image
                masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)

                # add the two images to get the final output
                frame[y:y + h, x:x + w] = cv2.add(masked_face, masked_frame)

        #new_frame = cv2.resize(frame, (960, 540))
        cv2.imshow('creativeMode', frame)
        # Press any key to exit:
        cv2.waitKey(0)


    else:
        raise ValueError("Wrong argument values!")
#**********************************************************************************************************************#


if __name__ == '__main__':

    if len(sys.argv) != 5:
        raise ValueError("Arguments not complete!")
    else:
        image_path = sys.argv[1]
        featureType = sys.argv[2]
        classifierType = sys.argv[3]
        creativeMode = sys.argv[4]
    

    print('\nReading Image', image_path)

    I = cv2.imread(image_path)
    persons = recogniseFace(I, featureType, classifierType, creativeMode)


    print('\n************ The list of detected people ************\n')
    print(persons)
    print('\n*****************************************************\n')
