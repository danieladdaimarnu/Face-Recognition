
##########################################################################################
			                 README
##########################################################################################



# Introduction
##############

This Readme file contains instructions and remarks concerning the coursework submission developed by Daniel ADDAI-MARNU. Please take a moment to read it before moving on.


# Submission Structure
########################

In the present folder, you will find :
	- Two folders : OCR and Face Recognition
	- Two .txt files : Readme.txt and requirements.txt
	- One .pdf file : Computer Vision - CVcoursework - (this is the report)
	- One .mp4 file : Recording on how to run the functions 


	----> OCR :

The OCR folder contains 3 directories : Models (which will be downloaded from google drive), Scripts and Samples. Models contains the trained models required to run the function (a KNN Classifier for digit classification and a ResNet detector for paper detection based on retinaNet). 
Scripts contains the detectNum.py script and Samples contains sample images on which the function can be tested.

	---> FACE RECOGNITION :

The Face Recognition folder contains 4 sub-folders : 
Models (which will be downloaded from google drive) where trained models are stored (This includes LR, SVM and RF for two types of feature extractors meaning 6 classifiers + a ResNet50 for face classification + Yolo v3 weights + YOLO v3 configuration file for Face Detection). 
The subfolder Resources contains the two visual codebooks used for facial recognition using feature extractors, an index_to_label.pikle file containing a dictionary of the mapping from the Resnet's internal indices of the classes to the class labels (assigned numbers) and the mask image for the creative mode. 
The Samples folder contains a couple of group images on which the recogniseFace function can be tested. 
Finally, the Scripts folder contains the recogniseFace.py script + a utilities file taken from https://github.com/sthanhng/yoloface that includes a few preprocessing functions that are necessary to run the Yolo Face detection.

*IMPORTANT*
The Models folder for both the OCR and Face Recognition can be download from my google drive using the link below:
https://drive.google.com/file/d/1Rje0XfNAEphU_krtOD3UaO9bEVUzyLJ6/view?usp=sharing

Please download the zip file, unzip it and you will get a folder called CV Models. This folder contains two folders OCR and Face Recognition. These two folders contain their respective models in a folder named Models in them. The steps are illustrated below:
For OCR:
CV Models -> OCR -> Models
For Face Recognition:
CV Models -> Face Recognition -> Models

Please take the whole Models folder for each function to their main folders in Final Submission folder downloaded from Moodle, ie, the Face Recognition folder and the OCR folder in the Final Submission folder. Thus the full function folders should have the following folders:
Final Submission -> Face Recognition -> Models (containing 9 files)
				     -> Resources (containing 4 files)
				     -> Samples (containing 3 files)
				     -> Scripts (containing 2 scripts)

		 -> OCR -> Models (containing 2 files)
		 	-> Samples (containing 3 files)
			-> Scripts (containing 1 script)


# Installing packages
########################

A requirements_face_recognition.txt and requirements_ocr.tx files are included and, in theory, covers all the necessary packages. I tried creating two new conda environments, activating them, running pip -r install path/to/requirements_face_recognition.txt and pip -r install path/to/requirements_ocr.txt  and managed to execute the OCR and Facial recognition functions in their respective  environments successfully. 

A few things to keep in mind. Two external GitHub repositories were used in this coursework. One of them for the Yolo Face Detector (https://github.com/sthanhng/yoloface), from which we only use a utilities script downloaded from the repo, as well as the weights of the model and its configuration file. The second repo, contains a Keras implementation of RetinaNet (https://github.com/fizyr/keras-retinanet) which I used and trained (transfer learning on my own data) for Paper Detection. There are a few modules from this repo which I require in my solution, so the requirements_ocr.txt clones and installs its content.

Also, the SURF and SIFT features are no longer present in the standard openCV package. In order to use them, openCV-contrib is required. It contains an important module  xfeatures2d which contains both the implementation of SIFT and SURF. For this reason, the requirements_face_recognition.txt file installs the version of opencv-contrib-python to install instead of the standard opencv. Hopefully you will not encounter any problems with this one.


# Running the scripts
########################

The two scripts recogniseFace.py and detectNum.py are not functions. They can be executed using command line and accept arguments to be passed to the functions inside.

	detectNum.py : Performs OCR on an image or video. It takes two arguments, the first one being the path to the test image, and the second one, which is optional and set to False by default, is a flag indicating whether or not to detect multiple numbers in the same image.

	EXAMPLE : This is how you can run this script. 
		- First, change directory (cd) to the Script folder in command line
		- Type : 
			python detectNum.py ../Samples/IMG_6855.JPG                 OR
			python detectNum.py ../Samples/IMG_trio.jpg True       (to tell the function you want to retrieve multiple potential numbers).
		- You can replace the path of the images with your own path.
	REMARK : Only JPG and MP4 formats are accepted by the function detectNum.py
	OUTPUT : The function will print any numbers detected as the output.

	recogniseFace.py : Performs Facial Recognition on an image. It takes four arguments : the FIRST is the path to the image, the SECOND is the FEATURE EXTRACTOR, which can be either 'SURF', 'SIFT' or 'none', the THIRD argument is the ALGORITHM which is either 'LR', 'SVM', 'RF','RESNET' or 'none' and the FOURTH argument is the CREATIVE MODE, which is either 0 or 1. Any other values will not be accepted by the function. And none has to be specified with RESNET. 

	EXAMPLE : This is how you can run this script.
		- First, change directory (cd) to the Script's folder.
		- Then type :
	1. For Face Recognition: 
		python recogniseFace.py ../Samples/frame.jpg SURF LR 0       
OR
		python recogniseFace.py ../Samples/IMG_6828.JPG none RESNET 0

	2. For Creative Mode: 
		python recogniseFace.py ../Samples/IMG_7049.JPG none none 1

		- You can replace the path with your own.

	OUTPUT : This script will print a matrix of numbers where the rows represent identified individuals and the 3 columns represent the PREDICTION, the X-COORDINATE (COLUMNS), and the Y-COORDINATE (ROWS) respectively for face recognition.
For creative mode it will output the image overlaid with a face mask.



		


