import torch
from torchvision import transforms
import torch.nn as nn
import PIL
from PIL import Image, ImageEnhance, ImageFilter
import torch.optim as optim
import numpy as np
import os
import torch.optim.lr_scheduler as lr_scheduler

from basicsr.models.archs.srresnet_arch import MSRResNet
from basicsr.metrics.psnr_ssim import calculate_psnr
from random import randrange


#Checking if cuda is present. Assigning cuda to device.
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device=torch.device('cuda')
    print("The device is" +str(device))
print("############################# Assigned Device ##############################")
torch.cuda.empty_cache()

#Function to map the predicted value in the range of 0 to 1 for converting it back to image.
def map_score_to_0_1(output):
    maxVal = torch.max(output)
    minVal = torch.min(output)
    output = (output - minVal)/(maxVal - minVal)
    return output
    
#Get all training image name.
trainImageList = os.listdir('./Mini-DIV2K/Train/LR_x4')

#Set total number of training samples i.e. 500
total_training_Samples=500

#Due to GPU memmory restriction Batch size used is 1.
default_bs=1


#initialize the model to ResNet.
model = MSRResNet()
model.train() 

#Sent model to GPU
model.to(device)

#Initilizing Loss function
criterion=nn.L1Loss()

#initializing optimizer.
optimizer = optim.Adam(model.parameters(),lr=0.0001)


for epoch in range(0,150):

    
    print('&&epoch is '+ str(epoch))
    total_Batches = int(total_training_Samples/default_bs)
    
    #Calulate running PSNR and loss of an epoch.
    running_loss=0
    runningPSNR=0
    for minibatch in range(0,total_Batches):        
        
        for imageNumber in range(minibatch*default_bs,(minibatch*default_bs)+default_bs):
        
            print('%%imageNumber is '+ str(imageNumber))
            #Average PSNR and loss of a batch
            avgLoss=0
            avgPSNR=0
            
            #Get image name.
            imageName = trainImageList[imageNumber]
            
            #get high and low resolution images.
            lrImage = Image.open('./Mini-DIV2K/Train/LR_x4/' +imageName).convert('RGB')
            hrImage = Image.open(r'./Mini-DIV2K/Train/HR/' +imageName).convert('RGB')
            
            #Get height and width of image.
            width, height = lrImage.size
            
            #Initialize image tensor
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)
                                
            
            #Transform Image to tensor
            lrImageTensor[0] = transforms.ToTensor()(lrImage)
            hrImageTensor[0] = transforms.ToTensor()(hrImage)
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
            

            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            
            print('##loss is' + str(loss.item()) + ' in 1 '+ str(imageName) +' epoch ' +str(epoch))
            
            #Calculating the PSNR
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 1 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            ###########################################
            ##Below are the Image augmentation tasks.##
            ###########################################
            
            ############################################################################################################################################### 
            
            #Image augmenataion - horizontal flip.
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)
                        
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            tempLrImage=tempLrImage.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)
            
            tempHrImage=tempHrImage.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage)
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
                        
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 2 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 2 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            ####################################################################################################################################

            #Image augmenataion - vertical flip.
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)
          
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            tempLrImage=tempLrImage.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            tempHrImage=tempHrImage.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 3 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 3 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()

            ###########################################################################################################################################            
            
            #Image augmenataion - Increase the brightness.
             
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)
                                  
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            enhancer = ImageEnhance.Brightness(tempLrImage)            
            tempLrImage=enhancer.enhance(1.5)
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            enhancer = ImageEnhance.Brightness(tempHrImage)
            tempHrImage=enhancer.enhance(1.5)
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 4 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 4 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            ############################################################################################################################           
            
            #Image augmenataion - Decrease the brightness.
             
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)
                       
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            enhancer = ImageEnhance.Brightness(tempLrImage)            
            tempLrImage=enhancer.enhance(0.5)
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            enhancer = ImageEnhance.Brightness(tempHrImage)
            tempHrImage=enhancer.enhance(0.5)
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 5 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 5 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
          
            ##############################################################################################################
            #Image augmenataion - Increase the contrast.
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)
            
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            enhancer = ImageEnhance.Contrast(tempLrImage)            
            tempLrImage=enhancer.enhance(1.5)
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            enhancer = ImageEnhance.Contrast(tempHrImage)
            tempHrImage=enhancer.enhance(1.5)
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 6 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 6 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            ####################################################################################################################            
            
            #Image augmenataion - Decrease the contrast.
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)            
            
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            enhancer = ImageEnhance.Contrast(tempLrImage)            
            tempLrImage=enhancer.enhance(0.5)
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            enhancer = ImageEnhance.Contrast(tempHrImage)
            tempHrImage=enhancer.enhance(0.5)
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 7 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 7 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            
            ###########################################################################################################################
            
            #Image augmenataion - Decrease the Sharpness.
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)
                        
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            enhancer = ImageEnhance.Sharpness(tempLrImage)            
            tempLrImage=enhancer.enhance(0.05)
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            enhancer = ImageEnhance.Sharpness(tempHrImage)
            tempHrImage=enhancer.enhance(0.05)
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 8 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 8 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            ####################################################################################################################
            
            #Image augmenataion - Increase the Sharpness.
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)            
            
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            enhancer = ImageEnhance.Sharpness(tempLrImage)            
            tempLrImage=enhancer.enhance(2)
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            enhancer = ImageEnhance.Sharpness(tempHrImage)
            tempHrImage=enhancer.enhance(2)
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 9 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 9 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            
            ####################################################################################################################################################            
            
            #Image augmenataion - Image Rotate 90
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,width,height)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3, width*4,height*4)
                        
            tempLrImage=lrImage
            tempHrImage=hrImage
            
                  
            tempLrImage=tempLrImage.rotate(90, expand=True)
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            tempHrImage=tempHrImage.rotate(90, expand=True)
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 10 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 10 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            
            ############################################################################################################################
                                          
            
            #Image augmenataion - Random Crop Image
            
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            left = randrange(0,int(width-width/6))
            top = randrange(0,int(height-height/6))
            right = int(left + int(width/6))
            bottom = int(top+ int(height/6))
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,bottom - top,right - left)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,(bottom - top)*4, (right - left)*4)    
            
                  
            tempLrImage=tempLrImage.crop((left, top, right, bottom))
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            tempHrImage=tempHrImage.crop((left*4, top*4, right*4, bottom*4))
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 11 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 11 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            
            
            ######################################################################################################################

            #Image augmenataion - Crop Image from random image and paste
           
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,height,width)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,height*4, width*4)                                   
            
            
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            randomImageName = trainImageList[randrange(total_training_Samples)]
            randomLrImage = Image.open('./Mini-DIV2K/Train/LR_x4/' +randomImageName).convert('RGB')
            randomHrImage = Image.open(r'./Mini-DIV2K/Train/HR/' +randomImageName).convert('RGB')
            randomImageWidth, randomImageHeight = randomLrImage.size 
            
            
            left = int(width/10)
            top = int(height/10)
            right = int(3 * width/10)
            bottom = int(3 * height/10)
            
            rImageLeft=0
            rImageTop=0
            
            if(randomImageHeight <= (bottom - top)):
                rImageBottom =randomImageHeight
                bottom = top + randomImageHeight
            else:
                rImageBottom = (bottom - top)
            
            
            if(randomImageWidth <= (right - left)):
                rImageRight =randomImageWidth
                right = left + randomImageWidth
            else:
                rImageRight = (right - left)
            
            
            croppedRandomLrImage=randomLrImage.crop((rImageLeft, rImageTop, rImageRight, rImageBottom))
            croppedRandomHrImage=randomHrImage.crop((rImageLeft, rImageTop, rImageRight *4, rImageBottom*4))
            

            tempLrImage.paste(croppedRandomLrImage,(left, top, right, bottom))
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            tempHrImage.paste(croppedRandomHrImage,(left*4, top*4, right*4, bottom*4))
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 

            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 12 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 12 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            
            
            ######################################################################################################################
                   

            #Image augmenataion - Random Crop Image
            
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            left = randrange(0,int(width-width/10))
            top = randrange(0,int(height-height/8))
            right = int(left + int(width/10))
            bottom = int(top+ int(height/8))
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,bottom - top,right - left)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,(bottom - top)*4, (right - left)*4)    
            
                  
            tempLrImage=tempLrImage.crop((left, top, right, bottom))
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            tempHrImage=tempHrImage.crop((left*4, top*4, right*4, bottom*4))
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 13 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 13 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            
            
            ######################################################################################################################                      
                                          
            
            #Image augmenataion - Random Crop Image and apply Gaussian Blur.
            
            tempLrImage=lrImage
            tempHrImage=hrImage
            
            left = randrange(0,int(width-width/5))
            top = randrange(0,int(height-height/6))
            right = int(left + int(width/5))
            bottom = int(top+ int(height/6))
            
            lrImageTensor = torch.cuda.FloatTensor(default_bs,3,bottom - top,right - left)
            hrImageTensor = torch.cuda.FloatTensor(default_bs,3,(bottom - top)*4, (right - left)*4)    
            
                  
            tempLrImage=tempLrImage.crop((left, top, right, bottom)).filter(ImageFilter.GaussianBlur(radius = 1))
            lrImageTensor[0] = transforms.ToTensor()(tempLrImage)

            tempHrImage=tempHrImage.crop((left*4, top*4, right*4, bottom*4))
            hrImageTensor[0] = transforms.ToTensor()(tempHrImage) 
            
            #Senting input to GPU
            lrImageTensor.to(device)        
            #Setting the gradients to 0 to remove all previous calculations.
            optimizer.zero_grad()
        
            #Predicting in input data
            scores=model(lrImageTensor)
        
            del lrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
        
            hrImageTensor.to(device)
        
            #Calculating the loss
            loss =  criterion(scores,hrImageTensor)
            
            #Calculating the gradients.
            loss.backward()
        
            #Updating the weights using the loss gradient.
            optimizer.step()
            
            outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
            
            print('##loss is' + str(loss.item()) + ' in 14 '+ str(imageName) +' epoch ' +str(epoch))
            psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
            print('**PSNR - ' +str(psnr)+ ' in 14 '+ str(imageName) +' epoch ' +str(epoch))
            
            avgLoss += loss.detach().item()
            avgPSNR +=psnr
        
            del hrImageTensor
            #Clearing GPU cache
            torch.cuda.empty_cache()
            
            
            
            ######################################################################################################################
            
            
            print('@@average loss of '+ str(imageName) +' in epoch ' +str(epoch) + ' is ' + str(avgLoss/14))
            print('$$average psnr of '+ str(imageName) +' in epoch ' +str(epoch) + ' is ' + str(avgPSNR/14))
            
            # adding total Loss and PSNR of the batch
            running_loss+=avgLoss
            runningPSNR+=avgPSNR
            
    
    #Calculating PSNR and Loss of 1 epoch
    epochLoss =  running_loss /(500*14)
    epochPSNR =  runningPSNR /(500*14)
    print('+++Loss of  epoch ' +str(epoch) + 'is ' + str(epochLoss))
    print('+++PSNR of  epoch ' +str(epoch) + 'is ' + str(epochPSNR))

    
    #Saving the model parameters of the currect epoch.    
    path_toSave = "./models/model_"+str(epoch)+"_epoch.pt"
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path_toSave)
