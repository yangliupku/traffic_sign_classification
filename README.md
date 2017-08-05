#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./write_up_images/figure1_dataset_visualization.png "Visualization"
[image2]: ./write_up_images/figure2_class_histogram.png "histogram"
[image3]: ./write_up_images/figure3_preprocessing.png "preprocessing"
[image4]: ./write_up_images/figure4_augment.png "augmentation"
[image5]: ./write_up_images/figure5_training_progress.png "training"
[image6]: ./write_up_images/figure6_custom_image.png "custom"
[image7]: ./write_up_images/figure7_pred_prob.png "custom"



---


##Data Set Summary & Exploration

###1. Basic summary of the dataset

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

###2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. One example image of each type of traffic sign is shown. We can immediately notice the contrast and exposure of the images are vastly different. We can imagine global and local contrast normalization will be needed to improve classification accuracy.

![example images][image1]

Below shows the histogram of number of images in each class in the training, validation and testing set. We notice two facts. First of all, the number of images in each class is very unbalanced, as much as a factor of 10. This will likely to cause the classifier to be more accurate on the classes with more data and less accurate on others. Second, the histograms are similar between training, validation and testing sets. This suggests if we optimize the performance of our classifier on the validation set, it'll likely to be able to generalize well on the test set.  

![TTT][image2]

##Design and Test a Model Architecture

###1. Image preprocessing and augmentation

The steps for image preprocessing are similar to Pierre Sermanet and Yann LeCun mentioned in [their paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). The pipeline consists 3 steps

1. Turn the image into `YUV` space, and use the `Y` channel as gray scale. `U` and `V` channels are discarded.
2. Perform histogram based local contrast normalization (LCN). via the function `skimage.exposure.equalize_adapthist`
3. Apply global contrast normalization (GCN) to scale the pixel values to have mean of 0 and standard deviation of 1. 

Here's what the traffic sign images look like before and after preprocessing.

![alt text][image3]

The amount of images in the training set is insufficient for the classifier to gernalize well because the different angle, position, brightness and backgrounds of traffic signs. I performed image augmentation using `keras.preprocessing.image.ImageDataGenerator` with the following setting:

* random rotation with +/- 20deg range
* random shift in the x and y direction with +/- 6 pixels (20% of the image size)
* random zoom with a factor ranging from 0.8 to 1.2
* random shear within +/- 0.5 rad  


Here is an example of an original image and an augmented image (after preprocessing):

![alt text][image4]

The difference between the original data set and the augmented data set is the following ... 


###2. Final model architecture

My final model consisted of the following layers:

```
                                  Type               Size     Dropout (keep p)
· · · · · · · · · ·    input      32x32x1    
@ @ @ @ @ @ @ @ @ @    Layer 1    3x3 Conv, maxpool   64       0.8        
∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶    
  @ @ @ @ @ @ @ @      Layer 2    3x3 Conv, maxpool   128      0.8     
  ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶     
    @ @ @ @ @ @        Layer 3    3x3 Conv, maxpool   256      0.7  
    ∶∶∶∶∶∶∶∶∶∶∶
     @ @ @ @ @         Layer 4    3x3 Conv, maxpool   512      0.5  
     ∶∶∶∶∶∶∶∶∶
     \x/x\x\x/         Layer 5    FC                  128      0.5    
      · · · ·         
      \x/x\x/          Output     FC                  43       0.5      
       · · ·                     
 
```
Please see function `model_pass` in `traffic_sign_tf_model.py` for the tensorflow implementation of the neural network. 

###3. Model training

To train the model, I used an AdamOptimizer with the following parameters.

```
batch size = 512
learning rate = 0.0005
early_stopping_patience = 30
l2_lambda = 0.0001 
```
dropout was used in the convolution layers and full connected layers to prevent overfitting. The dropout ratio is shown in the NN architecture above. Early stopping is employed to stop the training process when the loss on validation set stopped improving in the last 30 epochs. The traniing progress is shown in the plot below. 

![alt text][image5]


###4. Final results and discussion

My final model results were:

* training set accuracy of **99.89%**
* validation set accuracy of **99.4%**
* test set accuracy of **98.25%**

At first, I tried the architecture proposed in Pierre Sermanet and Yann LeCun's paper: 2 conv layers followd by 2 fully connected layers, with the feature maps extracted from the first conv layer skip forward to the FC layer. However, I find the model trains slowly, and doesn't generalize well on the validation set. In the end, I was able to get about 97% accuracy on the validation set, with about 300 epochs of training. I then tried adding more conv layers and reduce FC layers. I found this allows the model to train much faster, and the overfitting problems seems to be alleviated. After fixing the basic structure, I tried to vary the depth of each conv layer, the size of the FC layer, as well as the dropout ratio to find the balance between the test and validation loss. 
 

##Test a Model on New Images

###1. Custom traffic sign images

Here are five German traffic signs that I found on the web:

![alt text][image6] 

The first image might be difficult to classify because of the building in the background. The rim on the left image might make it difficult. The background of the third and fourth image was not seen in the training set, while the color of background was yellow on the fifth image, which might make them hard to classify.

###2. Model performace on the custom images 


Here are the results of the prediction:

| Image			        |     Prediction  | Precision  | Recall | Sample Percentage|
|:---------------------:|:---------------:| :--------:|:--------|:-----:|
| Double Curve       	| Double curve     | 0.967 | 0.967| 0.7%| 
| Stop Sign     			| Speed Limit(120km)|0.997 | 0.994| 2.8%|
| Speed Limit (20km)	| Speed Limit (20km)|1.0   | 0.967| 0.47%|
| Yeild      		       | Yield            |0.975  | 0.997| 5.7%|
| Roundabout		       | Roundabout       |1.0    |0.967 | 0.7%|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The above table summarizes the model performance on the 5 classes where the custom images belong to. The precision and recall value is calculated on the test set. `Sample Percentage` means the percentage of training samples of the class in the total population of training samples. If the classes were balanced, each class would have 1/43 = 2.3% samples. As we can see, on the image where the model made mistake (Stop Sign), both precision and recall are decent on the test set, and stop sign is not a rare class in terms of population. Therefore the error is likely due to the extra rim on the image which doesn't appear on the traning images. 


####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located from cell 171 to cell 179 cell of the Ipython notebook. I did not use the top_k function in tensorflow, instead I get the logtis from tensorflow model and calculated softmax and selected top 5 classes using numpy.  See the plot below for the probability predicted by my model on each image. 

![alt text][image7]


