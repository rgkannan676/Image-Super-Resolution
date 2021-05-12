# Single-Image-Super-Resolution-using-DIV2K-dataset
Created a model that can increase the resolution of images by 4 times, using an SRResNet model provided by BasicSR framework and DIV2K data set. Multiple data augmentation techniques were used along with the MSE loss function. 

# Task
The task of this project is to  increase the result of a single image. A high resolution and 4 times scaled low resolution pairs are given. Need to build a model that can create a high-resolution image from the low-resolution image.  PSNR will be the matric used to compare the quality of the created images.
Image Loading and Processing
PIL library was used to load the image and convert the same to tensor value for processing.  PIL also provides a normalized tensor of the images for processing.
The functionalities provided by PIL was used for data augmentation also.
Model 
The model used this project is SRResNet provided by BasicSR framework. This is a modified version of the SR-Resnet. The model uses residual network without Batch Normalization.
from basicsr.models.archs.srresnet_arch import MSRResNet
model = MSRResNet()

The  default parameter values of the models were used. 
num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4

The total number of parameters in this model is 1,517,571

# Loss Function

The Loss function used for this project is the absolute differences  L1 Loss. The predicted image for each low-resolution image  was compared with the corresponding high-resolution  training image. The PyTorch implementation of L1  loss torch.nn.L1Loss() was used for this project. The main idea of choosing this loss function was to reduce the distance between predicted image and ground truth image.

The PyTorch provided the functionality to calculate the backward propagation gradient. Loss functions provide the function loss.backward() to calculate the gradient.

# Optimizer

The optimizer used in this project was Adam. The implementation provided by PyTorch optim library torch. optim.Adam() is used. Adam is an adaptive learning technique combined with momentum. Default parameters values of the optimizer was used while training. 

Optimizers in PyTorch provide a method to update the weights of the tensor using the function optimizer.step(). This will update the weight using the gradients calculated.

# GPU Setup

GPU is used for doing the tensor computation. In PyTorch framework,  we can send the set the optimizer and tensors to Nvidia GPU using <tensorName>.to(“cuda”) function. GPU hardware increases the computation capacity.

# Image Augmentation Techniques Used

Image augmentation techniques were used to improve the accuracy of the network. Following were the augmentations done.
•	Horizontal Flip – The image was flipped in the horizontal direction with the help of image transpose facility provided by the PIL library. The transpose functionality with argument PIL.Image.FLIP_LEFT_RIGHT was used for the horizontal flip.

•	Vertical Flip -  The image was flipped in the vertical direction with the help of image transpose facility provided by the PIL library. The transpose functionality with argument PIL.Image. FLIP_TOP_BOTTOM was used for the vertical  flip.

•	Increase and decrease brightness -  Both the Low resolution and high-resolution brightness were changes by a factor of 1.5 and 0.5 using PIL. The ImageEnhance functionality provided by PIL was used to change the brightness. 
            
      enhancer = ImageEnhance.Brightness(image)            
      #Increase brightness of image by 1.5 factor.
      image =enhancer.enhance(1.5)

•	Increase and decrease contrast -  Both the Low resolution and high-resolution contrast were changes by a factor of 1.5 and 0.5 using PIL. The ImageEnhance functionality provided by PIL was used to change the contrast. 


•	Increase and decrease sharpness -  Both the Low resolution and high-resolution brightness were changes by a factor of 2 and 0.05 using PIL. The ImageEnhance functionality provided by PIL was used to change the sharpness.

•	Rotate image by 90 degree - Both the Low resolution and high-resolution images were rotated 90 degrees using PIL. The rotate functionality provided by PIL was used.

image=image.rotate(90, expand=True)

•	Random Crop- Both the Low resolution and high-resolution images cropped in similar proportions using PIL. The coordinates of the patch that need to be cropped is initialized an passed to the crop functionality provided by PIL.

image=image.crop((left, top, right, bottom))

•	Crop from random image and paste – Crop a patch from random image and paste it in the image that is trained for both low and high resolution images using PIL. Crop and paste functionality provided by PIL is used.

      #Cropping a patch from random image.
      randromImageCrop=randromImage.crop((rLeft, rTop, rRight, rBottom))
      #Pasting the cropped patch to image that need to be trained.
      image.paste(randromImageCrop,(left, top, right, bottom))

•	Random crop and adding blur effect – The images were random cropped and Gaussian blur with radius of 1 was applied only to the low-resolution images. This was done to improve the PSNR for low resolution images with noise. Image filter functionality provided by PIL was used to apply the filter.

	#Applying a gaussian blur filter on the cropped portion.
	image=image.crop((left, top, right,bottom))
	.filter(ImageFilter.GaussianBlur(radius = 1))


# Saving the Checkpoints

PyTorch provide a functionality to save the checkpoints by saving the model parameters, optimizer parameters etc. The functions torch.save() and torch.load() can be used to save and load the checkpoints.

# Metric Used

PSNR(Peak Signal to noise ratio) was used as the metric for checking the image quality. The function provided by BasciSR framework was used for PSNR calculation.

from basicsr.metrics.psnr_ssim import calculate_psnr
psnr=calculate_psnr(predictedImage, groudTruth, crop_border=4, input_order='CHW')

# map_score_to_0_1
A function was written to map the predicted values in the range of 0 to 1 for converting it back to image. It takes in a tensor does the calculation for mapping.

# Training

The model was trained for a total of 62 epochs. Initially the learning rate was set to 0.002 but observed that the loss was increasing after few epochs. This indicated that the learning rate is high. Reduced the initial learning rate to 0.0015.
During the training, different augmented images gave different PSNR. For example, the images were brightness was increased gave very low PSNR values like in range of 14 – 18dB compared to other data augmentations. This caused the average training PSNR to be low.

# Validation
Validation was performed in the 62 saved models to find the model that best fits the validation data of 80 images. Below is the epoch vs PSNR graph of Validation set.

![image](https://user-images.githubusercontent.com/29349268/118012990-3a4b4b00-b384-11eb-950e-85933c819b04.png) 

From the above graph we can understand that the PSNR value of the 47th epoch is the highest PSNR value  and after that it starts to decrease.

![image](https://user-images.githubusercontent.com/29349268/118013085-4fc07500-b384-11eb-9eda-4210175bb393.png)

Chose the model parameters of the 47th epoch to create the super resolution images of the private test set.

# Sample Images before and after applying super resolution

![0048](https://user-images.githubusercontent.com/29349268/118014301-a11d3400-b385-11eb-98c2-34b042a1b051.png)
![0048](https://user-images.githubusercontent.com/29349268/118014333-ada18c80-b385-11eb-8d3c-9a8f4f323e72.png)

--------------------------------------------------------------------------------------------------------------
![0037](https://user-images.githubusercontent.com/29349268/118014751-1c7ee580-b386-11eb-9a1e-a6f3116f01ce.png)
![0037](https://user-images.githubusercontent.com/29349268/118014767-2274c680-b386-11eb-80d3-9a31404e7755.png)



