import os
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.models import convnext_tiny
from torchvision.models.segmentation import FCN
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import masks_to_boxes, box_iou

class CrackDataset(Dataset):
    def __init__(self, mode='TRAIN'):
        self.mode = mode
        
        if mode == 'TRAIN':
            self.imgs = sorted([os.getcwd()+'/segmentation/dataset/traincrop/img/' + i for i in os.listdir(os.getcwd()+'/segmentation/dataset/traincrop/img/')])
            self.masks = sorted([os.getcwd()+'/segmentation/dataset/traincrop/mask/' + i for i in os.listdir(os.getcwd()+'/segmentation/dataset/traincrop/mask/')])
        elif mode == 'TEST':
            self.imgs = sorted([os.getcwd()+'/segmentation/dataset/testcrop/img/' + i for i in os.listdir(os.getcwd()+'/segmentation/dataset/testcrop/img/')])
            self.masks = sorted([os.getcwd()+'/segmentation/dataset/testcrop/mask/' + i for i in os.listdir(os.getcwd()+'/segmentation/dataset/testcrop/mask/')])
        elif mode == 'VAL':
            self.imgs = sorted([os.getcwd()+'/segmentation/dataset/valcrop/img/' + i for i in os.listdir(os.getcwd()+'/segmentation/dataset/valcrop/img/')])[0:5]
            self.masks = sorted([os.getcwd()+'/segmentation/dataset/valcrop/mask/' + i for i in os.listdir(os.getcwd()+'/segmentation/dataset/valcrop/mask/')])[0:5]
        else:
            pass
        
        # transform image and mask together
        self.transform_together = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
        
        # transform image only
        self.train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomInvert(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=(0.15, 0.35)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert('L')
        
        if self.mode == 'TRAIN':
            img, mask = self.transform_together(img, mask)
            return self.train_transform(img), self.mask_transform(mask)
        else:
            return self.transform(img), self.mask_transform(mask)
    
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
    
def get_iou(pred, truth):
    pred = pred.squeeze()
    truth = truth.squeeze()
    pred_boxes = masks_to_boxes(pred)
    truth_boxes = masks_to_boxes(truth)
    
    ious = box_iou(pred_boxes, truth_boxes)
    return ious.diag().mean().item()
    
if __name__ == '__main__':
    print(os.path.dirname(os.path.realpath(__file__)))
    print(os.getcwd())
    lr = 0.001
    batch_size = 16
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = CrackDataset()
    test_data = CrackDataset(mode='TEST')
    val_data = CrackDataset(mode='VAL')
    
    # data loader
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True)
    
    # feature extraction
    backbone = IntermediateLayerGetter(convnext_tiny(), {'features': "out"})
    model = FCN(backbone, MyClassifier(768, 1)).to(device)
    optimiser = optim.AdamW(model.parameters(), lr=lr)
    scheduler1 = LinearLR(optimiser, start_factor=lr, total_iters=5)
    scheduler2 = ExponentialLR(optimiser, gamma=0.9)
    scheduler = SequentialLR(optimiser, schedulers=[scheduler1, scheduler2], milestones=[5])
    criterion = nn.BCEWithLogitsLoss()
    
    train_losses = []
    test_losses = []
    train_ious = []
    test_ious = []
    
    for epoch in tqdm(range(1, epochs+1)):
        print(f'Epoch: {epoch}')
        model.train()
        train_loss = 0
        total_img = 0
        
        for batch in tqdm(train_loader):
            imgs, masks = batch
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            pred = model(imgs)['out']
            loss = criterion(pred, masks)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            total_img += masks.size(0)
        
        scheduler.step()
            
        train_loss = float(train_loss / total_img)
        train_losses.append(train_loss)
            
        # validation
        model.eval()
        val_loss = 0
        total_img = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                imgs, masks = batch
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)['out']
                loss = criterion(outputs, masks)
                
                val_loss += loss
                total_img += masks.size(0)
        
        val_loss = float(val_loss / total_img)
        test_losses.append(val_loss)
        
        print('\n')
        print('='*8 + f'Epoch: {epoch}' + '='*8)
        print(f'Training loss: {train_loss:.5f}')
        print(f'Validation loss: {val_loss:.5f}')
        
    final_loss = 0
    total_img = 0
    with torch.no_grad():
        i = 0
        for batch in test_loader:
            imgs, masks = batch
            imgs = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            
            img = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))
            mask = np.transpose(masks[0].cpu().detach().numpy(), (1, 2, 0))
            output = torch.sigmoid(outputs['out'])
            output = np.transpose(output[0].cpu().detach().numpy(), (1, 2, 0))
            
            output[output <= 0.5] = 0
            output[output > 0.5] = 1
            
            plt.subplot(5, 3, 3*i+1)
            plt.imshow(img)
            
            plt.subplot(5, 3, 3*i+2)
            plt.imshow(mask, cmap='gray')
            
            plt.subplot(5, 3, 3*i+3)
            plt.imshow(output, cmap='gray')
            
            i = i+1
            if i >= 5:
                break
            
            final_loss += loss
            total_img += masks.size(0)
        plt.show()
    
    final_loss = final_loss / total_img
    
    x = range(1, epochs+1)
    plt.plot(x, train_losses, label='Training Loss')
    plt.plot(x, test_losses, label='Test Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.xticks(np.arange(min(x), max(x), 2))
    plt.legend(loc='best')
    plt.show()
    
    print('='*8 + f'Epoch: {epoch}' + '='*8)
    print(f'Final loss: {final_loss:.5f}')
    
    torch.save(model.state_dict(), f'model_fcn.pth')
    print(f'model_fcn.pth is saved!')