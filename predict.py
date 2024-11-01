import os
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.models import convnext_tiny
from torchvision.models.segmentation import FCN
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts, LinearLR, ExponentialLR, SequentialLR
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision.models._utils import IntermediateLayerGetter
import argparse
from segmentation.train_ronald import MyClassifier
from segmentation.stdc import STDCNetForSegmentation
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('model', type=str, help='A required model type argument. Available models: FCN, STDC')
    parser.add_argument('img', type=str, help='A required image file name argument')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.model == 'FCN':
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        backbone = IntermediateLayerGetter(convnext_tiny(), {'features': "out"})
        model = FCN(backbone, MyClassifier(768, 1)).to(device)
        model.load_state_dict(torch.load('model_fcn.pth', weights_only=True))
        model.eval()
        
        img = Image.open(args.img).convert('RGB')
        # width, height = img.size
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        
        output = model(img)['out']
        output = torch.sigmoid(output)
        output = torch.squeeze(output, 0)
        output = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))
                
        output[output < 0.5] = 0
        output[output >= 0.5] = 1
        
        plt.imshow(output, cmap='gray')
        plt.show()
        
    elif args.model == 'STDC':
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Load the trained model
        stdc_model = STDCNetForSegmentation(num_classes=1)
        stdc_model.load_state_dict(torch.load('model_stdc.pth'))
        stdc_model.eval()  
        
        img = Image.open(args.img).convert('RGB')
        # width, height = img.size
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        
        output = stdc_model(img)['out']
        output = torch.sigmoid(output)
        output = torch.squeeze(output, 0)
        output = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))
                
        output[output < 0.5] = 0
        output[output >= 0.5] = 1
        
        plt.imshow(output, cmap='gray')
        plt.show()
        
        
    else:
        print('No model available!')
        exit()