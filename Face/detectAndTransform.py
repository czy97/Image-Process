from PIL import Image
import numpy as np
from src.face_process import getFace_112x96
from src.detector import FaceDetector
import cv2
import os
import traceback

def bgr2rgb(image):
    # The image read by cv2 is in bgr format, this
    # function transform it to rgb
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process_bbox(bbox):
    # change the format to top_left_x, top_left_y, width, height
    top_left_x = bbox[0]
    top_left_y = bbox[1]
    width = bbox[2] - top_left_x
    height = bbox[3] - top_left_y
    return np.array([top_left_x, top_left_y, width, height])

def process_landmarks(coordinates_array):
    # change the format to l_eye_x, l_eye_y, r_eye_x, r_eye_y, nose_x, nose_y,
    # l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y
    format_coordinate = np.zeros(10)
    format_coordinate[::2] = coordinates_array[:5] # coordinate x
    format_coordinate[1::2] = coordinates_array[5:] # coordinate y
    return format_coordinate

def get_bbox(image, facedetector):
    # the image is read by Image
    bounding_boxes, _ = facedetector.detect(image)

    if (isinstance(bounding_boxes, list) or bounding_boxes.shape[0] == 0):
        return None  # means no face is detected

    return process_bbox(bounding_boxes[0]) # only return the first bounding box

def get_landmarks(image, facedetector):
    # the image is read by Image
    _, landmarks = facedetector.detect(image)

    if(isinstance(landmarks,list) or landmarks.shape[0] == 0):
        return None # means no face is detected

    return process_landmarks(landmarks[0]) # only return the first bounding box

def get_bb_landmarks(image, facedetector):
    # the image is read by Image
    bounding_boxes, landmarks = facedetector.detect(image)

    if (isinstance(bounding_boxes, list) or bounding_boxes.shape[0] == 0):
        return None  # means no face is detected

    return (process_bbox(bounding_boxes[0]),process_landmarks(landmarks[0]))

def landmarks_to_facetrans(array):
    # this function change the format to a 5x2 matrix which can be used in face transform
    return np.resize(array,(5,2))

def get_transformedFace(img, facedetector):
    # we assume the img is cv2 read img
    # detect image's landmarks and transform the face to 112x96
    img = bgr2rgb(img)

    landmarks = get_landmarks(Image.fromarray(img,'RGB'),facedetector)

    if landmarks is None: # no faces is detected
        # May be the color of the image is terrible
        # I transform it to grey color
        (r, g, b) = cv2.split(img)
        tmp_image = img.copy()
        tmp_image[:, :, 0] = g
        tmp_image[:, :, 1] = g
        tmp_image[:, :, 2] = g
        facedetector.thresholds = [0.4, 0.5, 0.6]
        landmarks = get_landmarks(Image.fromarray(tmp_image, 'RGB'), facedetector)
        facedetector.thresholds = [0.6, 0.7, 0.8] # set back

        if landmarks is None: return None
        landmarks = landmarks_to_facetrans(landmarks)
        return getFace_112x96(img, landmarks)

    else:
        landmarks = landmarks_to_facetrans(landmarks)
        return getFace_112x96(img,landmarks)

if __name__ == '__main__':
    imagePath = 'viewtmp/0084_01.jpg'
    img = cv2.imread(imagePath)
    # The FaceDetector is a class used to calculate a image's face bbox and landmarks
    facedetector = FaceDetector()


    #***************face bounding box and face landmarks detection************
    img = bgr2rgb(img)
    img = Image.fromarray(img,'RGB') # the class only accept Image image
    bbox, landmarks = facedetector.detect(img)
    print(bbox)
    print(landmarks)


    #*********************detect and transform face image****************
    imagePath = 'viewtmp/0084_01.jpg'
    img = cv2.imread(imagePath)
    dst = get_transformedFace(img, facedetector)
    dst = bgr2rgb(dst)
    cv2.imshow('image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








