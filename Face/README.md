# Description

Most tasks on face involve some preprocessing procedure,  I will put the codes written by myself about face preprocessing here.

### Similarity transform

Different face images collected by people always vary in views and sizes. Before we input the image to the neural network, we want the size and specific landmarks' location of a face to be consistent. And the similarity transform is the algorithm to do such thing. Up to five landmarks (two eyes, nose and two mouth corners) will be used here to estimate similarity transform.

- testTransform.py 

  In this python script, I test the reasonable number and landmarks type to estimate a good similarity transform. And the test  result are put in viewtmp dir. I find that when use three landmarks to estimate transform, using two eyes and the nose is bad and will be abandoned in the face_process.py

- face_process.py

  Get the transformed and croped face.