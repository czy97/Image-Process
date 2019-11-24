"""
Author: Zhengyang Chen
Email: chenzhengyang117@gmail.com
Description: In this script, five landmarks(two eyes, nose and two mouth corners)
              are adopted to perform similarity transformation and face crop
"""

import cv2
import numpy as np

def plotlandmarks(img,landmarks):
    cv2.circle(img, (int(landmarks[0][0]), int(landmarks[0][1])), 1, (255, 0, 0), -1)
    cv2.circle(img, (int(landmarks[1][0]), int(landmarks[1][1])), 1, (255, 0, 0), -1)
    cv2.circle(img, (int(landmarks[2][0]), int(landmarks[2][1])), 1, (255, 0, 0), -1)
    cv2.circle(img, (int(landmarks[3][0]), int(landmarks[3][1])), 1, (255, 0, 0), -1)
    cv2.circle(img, (int(landmarks[4][0]), int(landmarks[4][1])), 1, (255, 0, 0), -1)

def estimateTransform(source_landmark,target_landmark,type = True):
    '''
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
def estimate_using_K_points(source_landmark,target_landmark,K,trans_type = True,third_type = 'mouth'):
    '''
    :param third_type: choose from mouth or nose
    '''
    if(K==5):
        tmp_source = source_landmark
        tmp_target = target_landmark
    elif(K==4):
        tmp_source = np.vstack((source_landmark[:2],source_landmark[-2:]))
        tmp_target = np.vstack((target_landmark[:2], target_landmark[-2:]))
    elif(K==3 and third_type == 'mouth'):
        tmp_source = np.vstack((source_landmark[:2], np.mean(source_landmark[-2:],0)))
        tmp_target = np.vstack((target_landmark[:2], np.mean(target_landmark[-2:],0)))
    else:
        # third_type='nose'
        tmp_source = source_landmark[:3]
        tmp_target = target_landmark[:3]

    return estimateTransform(tmp_source,tmp_target,trans_type)

def transformAndStore(img,imagename,storeSize,source_landmark,target_landmark):

    num2type = {
        5:['norm'],
        4:['norm'],
        3:['nose','mouth']
    }

    for num_landmark in [3,4,5]:
        for value in num2type[num_landmark]:
            for type in [True,False]:
                M =  estimate_using_K_points(source_landmark,target_landmark,
                                         num_landmark,type,value)
                if M is not None:
                    dst = cv2.warpAffine(img, M, storeSize)
                    plotlandmarks(dst, target_landmark)
                    cv2.imwrite('viewtmp/' + imagename + '_'+ str(num_landmark) + '_' + str(value) + '_' + str(type)+'.jpg', dst)



image_name = '0178_01'
img = cv2.imread(image_name + '.jpg')
rows,cols,ch = img.shape

landmarks_dic = {
'0245_01':[186.5081,76.82716,236.3222,86.63969,230.8857,127.2552,177.623,156.2089,223.7141,160.0954],
'0182_01':[112.7538,107.1615,161.6744,106.113,146.3448,131.7143,111.4071,157.1061,165.0231,150.9682],
'0084_01':[53.44244,42.54384,69.42963,48.49438,63.80945,57.90367,48.65472,62.90659,64.14357,66.7372],
'0137_01':[139.2207,172.1392,206.4467,182.0987,146.2135,220.1134,139.4129,241.6071,201.94,251.8409],
'0104_01':[82.0164, 85.8547, 118.8134, 92.0044, 78.5118, 115.9370, 72.6978, 149.8426, 103.3686, 153.0575],
'0290_01':[61.82832,47.8886,85.01485,53.70487,78.41882,61.39191,62.05292,74.69647,80.38196,80.09325],
'0178_01':[115.4742,156.1078,173.0514,153.7889,143.9185,180.0462,124.0847,216.7807,173.0039,215.7335]

}
landmarks = landmarks_dic[image_name]
imgsize = (96, 112)

source_points = np.float32([[landmarks[0],landmarks[1]],
                            [landmarks[2],landmarks[3]],
                            [landmarks[4],landmarks[5]],
                            [landmarks[6],landmarks[7]],
                            [landmarks[8],landmarks[9]]])

target_points = np.float32([[30.2946, 51.6963],
                            [65.5318, 51.5014],
                            [48.0252, 71.7366],
                            [33.5493, 92.3655],
                            [62.7299, 92.2041]])

transformAndStore(img,image_name,imgsize,source_points,target_points)

