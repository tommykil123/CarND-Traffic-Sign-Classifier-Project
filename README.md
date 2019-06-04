# Project: Build a Traffic Sign Recognition Program

## Overview
---
In this project I use a convolutional neural networks to classify traffic signs. I have trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After training and validating the model, I have verified the functionality of the CNN using pictures found on the web.


## The Project
---
The goals / steps of this project are the following:
Step 0: Load the data set
Step 1: Explore, summarize and visualize the data set
Step 2: Augument Training Data and Visualize
Step 3: Design, train and test a model architecture
Step 4: Use the model to make predictions on new images
Step 4B: Analyze the softmax probabilities of the new images


## Dataset

Base Dataset is the pickled dataset which is 32x32. It contains:
* Training (data/test.p)
* Validation (data/valid.p)
* Test set (data/test.p)

Augmented Dataset are also 32x32 and contained inside the data folder.  


### Step 0: Loading the Data
First step is to load the data. The database contains pictures of German Traffic signs
![GTSRB_43_classes](README_images/GTSRB_43_classes.png raw=true "GTSRB 43 Classes")

### Step 1: Visualize the Data Set
Because the data set we are working is huge, we want to visualize the data distibution of the data (how much data we have per label) for the train, dev, and test set.  
  
The figure on the bottom shows how skewed the dataset is with more data present for labels < 20 and fewer for labels > 20. This can overfit our prediction model to perform better for such labels.  

![train_valid_test_histo](README_images/train_valid_test_histo.png raw=true "Train Valid Test Histogram")  

### Step 2: PreProcess/Augumenting Data 
Because the data set we are working has noticible difference in amount between the different labels, we want to augment that dataset to produce more data for labels that do not have much on the training set. We are only augmenting the **Training SET** to create a model that is more robust to not only to certain labels, but all.  

Below are some of the preprocess methods to augument the training set:
* Flip the image along the vertical axis
* Salt and Pepper Noise
* Rotate the image +- n degree
![augment_example](README_images/augment_example.png raw=true "Augmentation Examples")   
  
After augmentating the data, plot another histogram to view the distribution of the training set
![train_histo.](README_images/train_histo.png raw=true "Original Training Set Histogram")  
![aug_train_histo](README_images/aug_train_histo.png raw=true "Augmented Training Set Histogram")  
  
In addition to adding noise and rotating the images, the following preprocess methods were also used
1. Grayscaling the images 
2. Normalizing the pixel values using the equation (pixel - 128)/ 128  

### Step 3: Design, train and test 
The CNN I used is similar to the LeNet-5 solution. The CNN takes in as input a 32x32x1 image and uses the softmax to determine which labels corresponds with the image. The model used containes the following layers:
1. Convolution Layer with 32 features.
* Kernel Size = 5x5
* Relu
* Dropout(0.7)
* MaxPooling (ksize = 2x2, stride = 2)
2. Convolution Layer with 64 features.
* Kernel Size = 5x5
* Relu
* Dropout(0.7)
* MaxPooling (ksize = 2x2, stride = 2)
3. Fully Connected Layer with 780 outputs
* Relu
* Dropout(0.7)
4. Fully Connected Layer with 360 outputs
* Relu
* Dropout(0.7)
5. Fully Connected Layer with 43 outputs
* Softmax

After creating this model, we ran the model on the validation data and achieved the following
* EPOCH 20 Validation Accuracy 0.924 with (BATCH_SIZE = 128, PreProcess: Grayscale, normalize)
* EPOCH 20 Validation Accuracy 0.936 with (BATCH_SIZE = 128, PreProcess: Grayscale, normalize, 10 deg rotate)
* EPOCH 20 Valdiation Accuracy 0.939 with (BATCH_SIZE = 128,PreProcess: Grayscale, normalize, 10 deg, 5 deg)
  
With the last augumented training set model, we achieved 92.5% with the test data.
  
### Step 4: Predictions on new images
The final final step was to find images from the web and see how well the model performed on these images. I chose 15 images as seen in the figure below.
![final_test_images](README_images/final_test_images.png raw=true "Final Test Images from the Web")  
  
The accuracy for this final test set was **60%**.
  
### Conclusion
From the final test set from the web, the softmax prediction shows that the model struggled with labels that had sparse amount of training data. That was expected as the training model did not see much of these images.
  
There are many improvements that could have been made. Instead of using the Le-Net5 architecture, other CNN architectures could have been made. Another improvement that could have been made was to use more augmented data for the training set.