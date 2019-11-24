"""
Author: Zhengyang Chen
Email: chenzhengyang117@gmail.com
Description: In this script, five landmarks(two eyes, nose and two mouth corners)
              are adopted to perform similarity transformation and face crop
"""

import cv2
import numpy as np

def estimateTransform(source_landmark,target_landmark,type = True):
    '''
    :source_landmark: Nx2 numpy array
    :target_landmark: Nx2 numpy array
    :param type: True: fullAffine False: similarity transform
    '''
    '''
    when fullAffine is set true, the estimateRigidTransform will
    compute a 6 freedom (full affine) transformation
    when fullAffine is set False, the estimateRigidTransform will
    compute a 4 freedom (similarity transform) transformation,
    which includes the rorate angle theta, the dilation ratio s and the translation along x and y
    '''
    M = cv2.estimateRigidTransform(source_landmark, target_landmark, type)
    return M

def estimate_using_K_points(source_landmark,target_landmark,K,trans_type = True):
    '''
    :param K: The number of landmarks we use to estimate the transform
    '''
    if(K==5):
        tmp_source = source_landmark
        tmp_target = target_landmark
    elif(K==4):
        tmp_source = np.vstack((source_landmark[:2],source_landmark[-2:]))
        tmp_target = np.vstack((target_landmark[:2], target_landmark[-2:]))
    else:
        # K == 3, we use two eyes, and the center of mouth as the three points
        tmp_source = np.vstack((source_landmark[:2], np.mean(source_landmark[-2:],0)))
        tmp_target = np.vstack((target_landmark[:2], np.mean(target_landmark[-2:],0)))

    return estimateTransform(tmp_source,tmp_target,trans_type)



def getFaceRegion(imgpath,detect_landmarks,standard_landmarks,return_imgsize):
    img = cv2.imread(imgpath)
    dst = None

    '''
    We will use the False type first, because False type corresponds to 
    similarity transform. But the estimation of similarity transform 
    always fails. If the estimation fails, we will estimate a affine
    transform
    '''
    for type in [False,True]:

        for landmark_num in [5,4,3]:
            M = estimate_using_K_points(detect_landmarks,standard_landmarks,landmark_num,type)
            if(M is not None):
                dst = cv2.warpAffine(img, M, return_imgsize)
                break
        if(M is not None):
            break

    return dst

def getFace_112x96(imgpath,detect_landmarks):
    imgsize = (96, 112)
    standard_landmarks = np.float32([[30.2946, 51.6963],
                                [65.5318, 51.5014],
                                [48.0252, 71.7366],
                                [33.5493, 92.3655],
                                [62.7299, 92.2041]])
    return getFaceRegion(imgpath, detect_landmarks, standard_landmarks, imgsize)

if __name__ == '__main__':
    from testTransform import landmarks_dic
    image_name = '0137_01'

    landmarks = landmarks_dic[image_name]
    source_points = np.float32([[landmarks[0],landmarks[1]],
                                [landmarks[2],landmarks[3]],
                                [landmarks[4],landmarks[5]],
                                [landmarks[6],landmarks[7]],
                                [landmarks[8],landmarks[9]]])

    dst = getFace_112x96(image_name+'.jpg',source_points)
    cv2.imshow('show',dst)
    cv2.waitKey(0)

