# **Traffic Sign Recognition**

## By Sylvana Alpert

---


[//]: # (Image References)

[image1]: ./writeup_images/unbalancedClasses.png "Unbalanced Classes"
[image2]: ./writeup_images/visualExploration.png "Visual Exploration"
[image3]: ./writeup_images/originalImage.png "Original Image"
[image4]: ./writeup_images/augmentedImage.png "Augmentations Example"
[image5]: ./writeup_images/unprocessedImages.png "Unprocessed Images"
[image6]: ./writeup_images/processedImages.png "Preprocessed Images"
[image7]: ./writeup_images/newimages.png "Traffic Signs From the Web"
[image8]: ./writeup_images/top5Softmax.png "Top 5 Probabilities"


The goal of this project was to build a traffic sign classifier using convolutional neural networks and apply it to the German Traffic Sign Data Set. The classifier was built using TensorFlow and the code can be found [here](https://github.com/sylvanaalpert/CarND-TrafficSigns-P2/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

The data set contains 32x32 color images of 43 different types of signs. Here's a summary of the data set properties:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

As shown below, the classes in the training, validation and testing sets are not balanced. The images below contain histograms of the
class labels in each set:

![alt text][image1]

Some randomly picked example images are shown below:

![alt text][image2]

We can see that there is a large variation in the images average values, with some being very dark and some very bright. We will have to correct this during preprocessing by performing histogram equalization.


### Design and Test a Model Architecture

Given the class imbalance described earlier, I decided to perform data augmentations of the less represented classes. To do so, I used affine transformations of each image until the number of examples in the class reached the maximum number of examples for a class in the data set. Color transformations were not used because color information has significance in traffic signs. Similarly, flipping images up/down or left/right was avoided since the orientation of the images is also meaningful. With that in mind, when doing affine transformations, the rotation angle was limited to the range of [-20, 20] to avoid drastic changes in the directions of arrows in the images.
Here is an example of an image that was randomly transformed using affine transformations:

![alt text][image3]

![alt text][image4]

Following data augmentations, the image data was preprocessed with these steps:
1. High pass filter to sharpen the image (implemented by subtracting a blurred version of the image)
2. Histogram equalization to improve the contrast
3. Normalize values to range of [-0.5, 0.5] to achieve zero mean


Here are three examples of traffic signs before and after pre-processing.

![alt text][image5]

![alt text][image6]


I started to experiment with the LeNet model, as implemented during the LeNet lesson. That model did not perform sufficiently well and therefore, I decided to make the following modifications:

1. Add a convolutional layer to increase the networks depth and allow it to learn finer features.
2. Expanded the number of filters per convolutional layer.
3. Changed the padding type used in convolutional layers to SAME, to prevent the observed region in an image from becoming too small too quickly.
4. Added dropout layers after convolutional layers to prevent overfitting

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 size, 2x2 stride,  outputs 16x16x32 		|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 size, 2x2 stride,  outputs 8x8x64 		|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x128		|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 size, 2x2 stride,  outputs 4x4x128 		|
| Fully connected		| Outputs 512x1								|
| RELU					|												|
| Fully connected		| Outputs 512x1								|
| RELU					|												|
| Fully connected		| Outputs 43x1									|
| Softmax				| 	        									|



The model was trained over 15 epochs, with a batch size of 128. I experimented with the learning rate, and found a value of 0.001 to work best. As learnt during the lesson, the mean softmax cross entropy between logits and labels was used as a loss function. In addition, the Adam optimizer was chosen due to its exponentially decaying learning rate.


In order to get the validation accuracy to be at least 0.93, an iterative process was taken. I started by adapting the LeNet model to accept RGB images and output the right number of classes. Later on, after noticing the low accuracy measurements, the number of filters on each convolutional layer was progressively expanded until further expansion did not provide any benefit. In addition, an extra convolutional layer without any downsampling was added as the first layer to increase the network's depth and allow the network to better learn non-linearities. Fully connected layers were also expanded to contain more filters than the LeNet architecture, to allow for more linear combinations of the larger number of features computed in previous layers. At this point, the test set accuracy was already above the 0.93 requirement, however, I decided to add some dropout layers with a keep probability of 0.8 to avoid overfitting the data and improve performance on any new images fed through the network.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.959
* test set accuracy of 0.948

The high training set accuracy and lower test accuracy suggest that the model is overfitting the data in the training set and there are further improvements to do.

Convolution layers are very suitable for this kind of problem since there exist different levels of hierarchy in the traffic signs with different information in it (such as color, shape, orientation and specific sign content). To allow a network to learn about these features at different levels of complexity, a network needs to have a sufficiently deep architecture (numerous layers). The approach here was guided by trial and error and any changes in the architecture that produced a better outcome were kept while negative changes were reversed. Here, the addition of a single convolution layer was enough to achieve the target accuracy.  


### Test a Model on New Images

Five German traffic signs found on the web were used to further test the model.

Here are traffic signs selected from the web:

![alt text][image7]

The first three images were selected because of their similar shape and color. They differ on the small contents of the sign and could be used to briefly test the ability of the network to discern small details. The fourth image was expected to be easy to classify due to its distinct look from other signs in the data set. The fifth image was selected to test how well the network could identify the numbers on the sign and match them to the appropriate label.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Children crossing    	| Children crossing   							|
| Road work   			| Road work										|
| Bicycles crossing		| Bicycles crossing								|
| No entry	      		| No entry						 				|
| 60 km/h 				| No vehicles 	      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is comparable to the accuracy on the test set, given the size of this new testing set (5).

The top 5 softmax probabilities for each image were calculated and are plotted below.
The code for making predictions on my final model is located in the 3th cell of Step 3 of the Ipython notebook.

For images 1, 3 and 4, the model is correct and relatively certain of its predictions, which have a probability of over 0.8.  For image 2, the model makes the right prediction, although the certainty decreases. We can see that in that case, the contents in black get confused with the contents of "Road narrows to the right", a sign with the same triangular shape and colors. The prediction for the last sign is predicted with high certainty but it is wrong, as the sign gets confused with a "No vehicles" sign, which is a round sign, with red border but no image inside it.
From the observations from these 5 signs, we can conclude that the model is not properly learning the finer components of the sign images and would benefit from a deeper architecture.

![alt text][image8]
