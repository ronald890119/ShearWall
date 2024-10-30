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

class MyClassifier(nn.Module):
    def __init__(self, in_channels=768, out_channels=1) -> None:
        super().__init__()
        inter_channels = in_channels // 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(inter_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(inter_channels, 1, kernel_size=(1, 1), stride=(1, 1)),
        )
        
    def forward(self, x):
        return self.conv(x)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('model', type=str, help='A required model type argument. Available models: FCN')
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
        model.load_state_dict(torch.load('model.pth', weights_only=True))
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
    else:
        print('No model available!')
        exit()