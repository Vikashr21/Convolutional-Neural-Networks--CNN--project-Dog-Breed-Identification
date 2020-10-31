# Convolutional Neural Networks (CNN) project Dog Breed Identification

This is the repo of Dog breed classifier project in Udacity ML Nanodegree.
**The Road Ahead**
<br>notebook is broken into 6 steps:

<br>**Step 0:** Import Datasets:

<br>Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip the folder and place it in this project's home directory, at the location /dogImages.
<br>Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip). Unzip the folder and place it in the home directory, at location /lfw.


**Step 1:** Detect Humans
<br>In this section, we use OpenCV’s implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to
detect human faces in images.
OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades). We have
downloaded one of these detectors and stored it in the haarcascades directory.


**Step 2:** Detect Dogs
In this section, we use a  Pre-trained VGG-16 Model  to detect dogs in images.
The code cell below downloads the VGG-16 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks. ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).


**Step 3:** Create a CNN to Classify Dog Breeds (from Scratch)
**Step 4:** Create a CNN to Classify Dog Breeds (using Transfer Learning)
	
 build a CNN model from scratch to solve the problem. The model has 3 convolutional layers. All convolutional layers have kernel size of 3 and stride 1. The first conv layer (conv1) takes the 224*224 input image and the final conv layer (conv3) produces an output size of 128. ReLU activation function is used here. The pooling layer of (2,2) is used which will reduce the input size by 2. We have two fully connected layers that finally produces 133-dimensional output. A dropout of 0.25 is added to avoid over overfitting

**Step 5:** Write your Algorithm
**Step 6:** Test Your Algorithm


**Refinement**
<br>The CNN created from scratch have accuracy of 14%, though it meets the benchmarking, the model can
be significantly improved by using transfer learning. To create CNN with transfer learning, I have
selected the Resnet101 architecture which is pre-trained on ImageNet dataset, the architecture is 101
layers deep. The last convolutional output of Resnet101 is fed as input to our model. We only need to
add a fully connected layer to produce 133-dimensional output (one for each dog category). The model 
performed extremely well when compared to CNN from scratch. With 10 epochs, the model got 84%
accuracy.
![](https://github.com/Vikashr21/Convolutional-Neural-Networks--CNN--project-Dog-Breed-Identification/blob/main/output.PNG)

**Model Evaluation and Validation**
<br>**Human Face detector:** The human face detector function was created using OpenCV’s implementation
of Haar feature based cascade classifiers. 98% of human faces were detected in first 100 images of
human face dataset and 17% of human faces detected in first 100 images of dog dataset.
<br>**Dog Face detector:** The dog detector function was created using pre-trained VGG16 model. 100% of
dog faces were detected in first 100 images of dog dataset and 0% of dog faces detected in first 100
images of human dataset. 
<br>**CNN using transfer learning:** The CNN model created using transfer learning with ResNet101
architecture was trained for 10 epochs, and the final model produced an accuracy of 84% on test data.
The model correctly predicted breeds for 708 images out of 836 total images.
<br><br>**Accuracy on test data: 84% (708/836)**