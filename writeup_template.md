#**Traffic Sign Recognition** 

##By Sylvana Alpert

---


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

---

The goal of this project was to build a traffic sign classifier using convolutional neural networks and apply it to the German Traffic Sign Data Set. The classifier was built using TensorFlow and the code can be found [here](https://github.com/sylvanaalpert/CarND-TrafficSigns-P2/blob/master/Traffic_Sign_Classifier.ipynb).

###Data Set Summary & Exploration

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

![alt text][image1]

We can see that there is a large variation in the images average values, with some being very dark and some very bright. We will have to correct this during preprocessing by performing histogram equalization. 


###Design and Test a Model Architecture

Given the class imbalance described earlier, I decided to perform data augmentations of the less represented classes. To do so, I used affine trasformations of each image until the number of examples in the class reached the maximum number of examples for a class in the data set. Color transformations were not used because color information has significance in traffic signs. Similarly, flipping images up/down or left/right was avoided since the orientation of the images is also meaningful. With that in mind, when doing affine transformations, the rotation angle was limited to the range of [-20, 20] to avoid drastic changes in the directions of arrows in the images. 
Here is an example of an image that was randomly transformed using affine transformations: 
![alt text][image2]

Following data augmentations, the image data was preprocessed with these steps: 
#. High pass filter to sharpen the image (implemented by subtracting a blurred version of the image)
#. Histogram equalization to improve the contrast
#. Normalize values to range of [-0.5, 0.5] to achieve zero mean


Here are three examples of traffic signs before and after pre-processing.

![alt text][image2]


I started to experiment with the LeNet model, as implemented during the LeNet lesson. That model did not perform sufficiently well and therefore, I decided to make the following modifications: 

#. Add a convolutional layer to increase the networks depth and allow it to learn finer features. 
#. Expanded the number of filters per convolutional layer.
#. Changed the padding type used in convolutional layers to SAME, to prevent the observed region in an image from becoming too small too quickly.
#. Added dropout layers after convolutional layers to prevent overfitting

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 



