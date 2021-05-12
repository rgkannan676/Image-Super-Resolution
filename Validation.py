import torch
import torchvision.models as models
from torchvision import transforms
import torch.nn as nn
import PIL
from PIL import Image
from torchvision import transforms
import torch.optim as optim
import numpy as np
import os
import torch.optim.lr_scheduler as lr_scheduler

from basicsr.models.archs.srresnet_arch import MSRResNet
from basicsr.metrics.psnr_ssim import calculate_psnr

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
    

#Get validation image list.
trainImageList = os.listdir('./Mini-DIV2K/Val/LR_x4')
print(trainImageList)

#initiize model
model = MSRResNet()


bestPSNR = 0
bestEpoch = 0

for epoch in range(0,6):
    
    #get the model checkpoint of the corresponding epoch.
    checkpoint = torch.load("./models/model_"+str(epoch)+"_epoch.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() 
    model.to(device)
    
    averagePSNR=0
    
    for imageNumber in range(len(trainImageList)):

        imageName = trainImageList[imageNumber]
        
        #Load and convert images to tesnor.
        lrImage = Image.open('./Mini-DIV2K/Val/LR_x4/' +imageName).convert('RGB')
        hrImage = Image.open(r'./Mini-DIV2K/Val/HR/' +imageName).convert('RGB')

        width, height = lrImage.size
        lrImageTensor = torch.cuda.FloatTensor(1,3,height,width)
        hrImageTensor = torch.cuda.FloatTensor(1,3,height*4, width*4)
        
        lrImageTensor[0] = transforms.ToTensor()(lrImage)
        hrImageTensor[0] = transforms.ToTensor()(hrImage)
        
        #get the prediction.
        scores=model(lrImageTensor)
        
        del lrImageTensor
        torch.cuda.empty_cache()
        
        #Map prediction in 0 to 1 range.
        outputMappedScore = map_score_to_0_1(scores[0].detach().cpu())
        
        #Calculate the PSNR.
        psnr=calculate_psnr(np.array(outputMappedScore * 255), np.array(hrImageTensor[0].detach().cpu() * 255), crop_border=4, input_order='CHW')
        print('**PSNR of image name '+ str(imageName) + ' of model '+ str(epoch) + ' is ' +str(psnr))
        
        del hrImageTensor
        torch.cuda.empty_cache()
        
        averagePSNR +=psnr
    
    #calculate average PSNR of the epoch.
    avgPSNR =  averagePSNR/len(trainImageList)    
    print('>>AveragePSNR of model '+ str(epoch) + ' is ' + str(avgPSNR))
    
    #Find the epoch with largest PSNR
    if(avgPSNR > bestPSNR ):
        bestPSNR = avgPSNR
        bestEpoch = epoch


print('!!best PSNR of model is in epoch '+ str(bestEpoch) + ' is ' + str(bestPSNR))   