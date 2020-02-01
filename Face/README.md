# Description

Most tasks on face involve some preprocessing procedure,  I will put the codes written by myself about face preprocessing here.

### Face landmarks detection using [MTCNN](https://arxiv.org/pdf/1604.02878.pdf)

The code using [MTCNN](https://arxiv.org/pdf/1604.02878.pdf) to detect the face's bounding box and face landmarks. And the detected face landmarks can be using the following similarity transform task.

- /src/detector.py 

  The defination of face detector class

- detectAndTransform.py

  The example about face detect.

### Similarity transform

Different face images collected by people always vary in views and sizes. Before we input the image to the neural network, we want the size and specific landmarks' location of a face to be consistent. And the similarity transform is the algorithm to do such thing. Up to five landmarks (two eyes, nose and two mouth corners) will be used here to estimate similarity transform.

- testTransform.py 

  In this python script, I test the reasonable number and landmarks type to estimate a good similarity transform. And the test  result are put in viewtmp dir. I find that when use three landmarks to estimate transform, using two eyes and the nose is bad and will be abandoned in the face_process.py

- face_process.py

  Get the transformed and croped face. And there is an issue should be noted, that some of the opencv-python version don't have the estimateRigidTransform function. Here I use opencv-python==3.4.2.16

### 



