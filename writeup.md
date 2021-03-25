# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/random.PNG "Random"
[image2]: ./writeup_images/each_sign_count.png "sign count"
[image3]: ./writeup_images/train_sign_frequency.png "train freq"
[image4]: ./writeup_images/test_sign_frequency.png "test freq"
[image5]: ./writeup_images/valid_sign_frequency.png "valid freq"
[image6]: ./writeup_images/preprocessed_image.PNG "preprocess"
[image8]: ./writeup_images/web_images.PNG "New Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
You're reading it! and here is a link to my [project code](https://github.com/varshasathya/CarND_Traffic_Sign_Classifier/blob/main/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the pandas and numpy library to calculate summary statistics of the German traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

The below image is some random images from training dataset:

![alt text][image1]

The below is an analysis of No. of Samples in each unique classes/labels from training set.

![alt text][image2]

The below is an analysis of distribution of labels across train,test and validation dataset.

![alt text][image3]          ![alt text][image4]              ![alt text][image5]

### Design and Test a Model Architecture

Below is the preprocessing that I have done on the images.

1. I normalized the images because the neural networks tends to work well if the feature distribution have zero mean.
2. I converted the image to greyscale as it will reduce no. of features and thus increasing the execution time. Also, most of the images in the dataset is darker.

```python
def normalize(image):
    return cv2.normalize(image, np.zeros(image_shape[0:2]), 0, 255, cv2.NORM_MINMAX)
def rgb_to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def preprocess_image(images): 
    prep_shape = (32, 32, 1)
    prep = np.empty(shape=(len(images),) + prep_shape, dtype=int)
    for i in range(0, len(images)):
        normalized = normalize(images[i])
        gray_image = rgb_to_grey(normalized)
        prep[i] = np.reshape(gray_image, prep_shape)
    return prep
```
The comparison between the images before and after preprocessing

![alt text][image6]


#### 2. Final model architecture

The model architecture is based on the LeNet model architecture. I added dropout layers before each fully connected layer in order to prevent overfitting. My final model consisted of the following layers:

| Layer					|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input			 		| 32x32x1 Greyscale image						| 
| Convolution Layer1    | 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					| Activation Layer								|
| Max pooling			| 2x2 stride,  outputs 14x14x16 				|
| Convolution Layer2	| 1x1 strides, valid padding outputs 10x10x64	|
| RELU					| Activation Layer								|
| Max pooling			| 2x2 stride,  outputs 5x5x64					|
| Flatten				|												|
| Fully Connected1		| outputs 1600									|
| RELU					| Activation Layer								|
| Droput				| keep probability 0.7							|
| Fully Connected2		| outputs 240									|
| RELU					| Activation Layer								|
| Droput				| keep probability 0.7							|
| Fully Connected3		| outputs 43									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer and the following hyperparameters:

learning rate: 0.001
number of epochs: 40
batch size: 64
mu = 0.0 and sigma = 0.1
keep probalbility of the dropout layer: 0.7

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. 

1. Initially I used LeNet model architecture from the course.I changed only the no. of outputs to 43 as we have 43 labels in this problem.Since it works well for character recognition, this could also work well for traffic sign classification. 

2. With epochs = 30, Initially I got only 88.4% Accuracy while the training accuracy was 97.5%.

3. Later I increased the epochs to 50, and also added dropout layers with keep probability 0.5 after every fully connected layers to avoid this overfitting of data. Here I got 96% training accuracy while validation accuracy was 90.6%.

4. For above LeNet architecture with dropout with batch size 128 and epochs 50 and keep probability to 0.7, the results obtained are:
    -Train Accuracy = 98.6%
    -Validation Accuracy = 91.8%
    -Test Accuracy = 90.3%
    
5. To further tweek the model,I made the convolution layers deeper and increased the size of fully connected layer. After which I got the below results:
  -Validation Accuracy = 93.2
  -Test Accuracy = 92.3
  
6. Later I reduced batch size to 64 and epochs to 40, after which I got below results:
   -Train Accuracy = 99.3%
   -Validation Accuracy = 94.3%
   -Test Accuracy = 92.7%
The maximum validation accuracy obtained was 95.2%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web:

![alt text][image8] 

The "general caution" sign might be difficult to classify because the triangular shape is similiar to several other signs in the training set. Also, the "stop" sign might be confused with the "No entry" sign due to pretty much similar shape.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

Predicted:
18 : General caution
Actual:
18 : General caution

Predicted:
38 : Keep right
Actual:
38 : Keep right

Predicted:
13 : Yield
Actual:
13 : Yield

Predicted:
14 : Stop
Actual:
14 : Stop

Predicted:
34 : Turn left ahead
Actual:
34 : Turn left ahead

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 92.7% and validation set 95.2%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

Below is the top 5 softmax probabilities of the predictions on the new images:

-Image: Testing_images/general_caution.jpg
   -Probabilities:
    - 1.000000 : 18 - General caution
    - 0.000000 : 0 - Speed limit (20km/h)
    - 0.000000 : 1 - Speed limit (30km/h)
    - 0.000000 : 2 - Speed limit (50km/h)
    - 0.000000 : 3 - Speed limit (60km/h)

-Image: Testing_images/keep_right.jpg
  -Probabilities:
   - 1.000000 : 38 - Keep right
   - 0.000000 : 0 - Speed limit (20km/h)
   - 0.000000 : 1 - Speed limit (30km/h)
   - 0.000000 : 2 - Speed limit (50km/h)
   - 0.000000 : 3 - Speed limit (60km/h)

-Image: Testing_images/yield.jpg
  -Probabilities:
   - 1.000000 : 13 - Yield
   - 0.000000 : 0 - Speed limit (20km/h)
   - 0.000000 : 1 - Speed limit (30km/h)
   - 0.000000 : 2 - Speed limit (50km/h)
   - 0.000000 : 3 - Speed limit (60km/h)

-Image: Testing_images/stop.jpg
  -Probabilities:
   - 0.997981 : 14 - Stop
   - 0.002019 : 37 - Go straight or left
   - 0.000000 : 39 - Keep left
   - 0.000000 : 33 - Turn right ahead
   - 0.000000 : 19 - Dangerous curve to the left

-Image: Testing_images/turn_left_ahead.jpg
  -Probabilities:
   - 1.000000 : 34 - Turn left ahead
   - 0.000000 : 0 - Speed limit (20km/h)
   - 0.000000 : 1 - Speed limit (30km/h)
   - 0.000000 : 2 - Speed limit (50km/h)
   - 0.000000 : 3 - Speed limit (60km/h)


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
This was an optional step. Implementing and visualising these layers will be taken as my future work.
