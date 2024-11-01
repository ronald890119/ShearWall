import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import torch.optim as optim
import time
import torchvision.transforms.functional as TF
import random


#Define Evaluation Functions
def iou(pred, target, threshold=0.5, smooth=1e-6):
    # Binarize the predictions
    pred = (pred > threshold).float()

    # Calculate intersection and union
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    # Check for empty masks
    if union == 0:  # Avoid division by zero
        return 1.0 if intersection == 0 else 0.0

    # Calculate IoU with smoothing factor
    iou_score = (intersection + smooth) / (union + smooth)

    return iou_score.item()  # Convert to a scalar value

def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    # Binarize the predictions
    pred = (pred > threshold).float()

    # Calculate intersection and sum of both masks
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    # Check for empty masks
    if pred_sum == 0 and target_sum == 0:  # Avoid division by zero
        return 1.0  # Perfect match for empty masks

    # Calculate Dice with smoothing factor
    dice_score = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)

    return dice_score.item()  # Convert to a scalar value

def save_model(model, model_name='best_model.pth', enable_message=True):
    """
    Save the trained model weights to a specified file.
    """
    torch.save(model.state_dict(), model_name)
    if enable_message:
        print(f"Model saved as {model_name}\n")


def load_model(model, model_name='dbrnet_model.pth'):
    """
    Load the model weights from a specified file.
    """
    model.load_state_dict(torch.load(model_name))
    print(f"Model loaded from {model_name}")


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # A 1x1 convolution to generate attention maps
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.global_pool(x)
        attention = self.conv(attention)
        attention = self.bn(attention)  # Apply batch normalization after conv
        attention = self.sigmoid(attention)
        # Broadcasting attention to match the input feature size
        attention = attention.expand_as(x)
        return x * attention


class FeatureFusionModule(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        # Projection to match encoder channels to decoder channels
        self.encoder_projection = nn.Conv2d(encoder_channels, decoder_channels, kernel_size=1, stride=1, padding=0)
        # Fusion convolution to reduce channels after concatenation
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(decoder_channels + decoder_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Global pooling and attention mechanism
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_sigmoid = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, encoder_feature, decoder_feature):
        # Ensure encoder feature size matches decoder feature size
        encoder_feature = F.interpolate(encoder_feature, size=decoder_feature.shape[2:], mode='bilinear', align_corners=False)
        # Match the encoder channels with decoder channels using a 1x1 convolution
        encoder_feature = self.encoder_projection(encoder_feature)
        # Concatenate encoder and decoder features along the channel dimension
        fusion = torch.cat([encoder_feature, decoder_feature], dim=1)
        # Apply convolution to fuse channels
        fusion = self.conv_bn_relu(fusion)
        # Apply global average pooling for attention
        pooled = self.global_pool(fusion)
        attention = self.conv_sigmoid(pooled)
        # Expand attention to match the spatial size
        attention = attention.expand_as(fusion)
        # Multiply attention weights with the fusion features
        fusion = fusion * attention
        # Residual connection: add encoder features
        fusion = fusion + encoder_feature

        return fusion


# Helper function to define convolution block (Conv -> BN -> ReLU)
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling
        scale = self.global_avg_pool(x)
        # Fully connected layers
        scale = F.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale


class STDCModule(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, stride):
        super(STDCModule, self).__init__()
        self.stride = stride
        # Define the blocks based on the stride
        if self.stride == 1:
            self.block_kernels = [1, 3, 5, 7]
        elif self.stride == 2:
            self.block_kernels = [1, 3, 7, 11]

        self.blocks = nn.ModuleList([ConvBNReLU(in_channels if i == 0 else out_channels,
                                                out_channels,
                                                kernel_size=self.block_kernels[i])
                                     for i in range(blocks)])

        # Use a 1x1 convolution to reduce the concatenated output back to `out_channels`
        self.reduction_conv = ConvBNReLU(in_channels + blocks * out_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.se_block = SEBlock(out_channels)

    def forward(self, x):
        outputs = [x]  # Collect initial input

        # Forward through all blocks and store the intermediate outputs
        for block in self.blocks:
            x = block(x)
            outputs.append(x)

        # Resize all outputs to match the size of the first output (input size)
        resized_outputs = [F.interpolate(output, size=outputs[0].shape[2:], mode='bilinear', align_corners=False)
                           for output in outputs]

        # Concatenate all resized outputs along the channel dimension
        x = torch.cat(resized_outputs, dim=1)

        # Apply the final reduction convolution to adjust the channel size
        x = self.reduction_conv(x)
        x = self.se_block(x)  # Apply SE block

        return x
    

class STDCNet(nn.Module):
    def __init__(self):
        super(STDCNet, self).__init__()

        # Stage 1 and 2: Basic Convolutions
        self.stage1 = ConvBNReLU(3, 64, stride=2)
        self.stage2 = ConvBNReLU(64, 128, stride=2)

        # Stage 3, 4, 5: Two STDC modules per stage (one with stride=2, one with stride=1)
        self.stage3_down = STDCModule(128, 256, blocks=3, stride=2)
        self.stage3_refine = STDCModule(256, 256, blocks=3, stride=1)

        self.stage4_down = STDCModule(256, 512, blocks=4, stride=2)
        self.arm4 = AttentionRefinementModule(512, 512)  # ARM for stage 4
        self.stage4_refine = STDCModule(512, 512, blocks=4, stride=1)

        self.stage5_down = STDCModule(512, 1024, blocks=2, stride=2)
        self.arm5 = AttentionRefinementModule(1024, 1024)  # ARM for stage 5
        self.stage5_refine = STDCModule(1024, 1024, blocks=2, stride=1)

        # Detail Aggregation Layer to reduce channels to 256
        self.detail_aggregation = ConvBNReLU(1024, 256, kernel_size=1, stride=1)

        # Feature Fusion Modules with channel alignment
        self.ffm = FeatureFusionModule(256, 256, 256)
        self.ffm2 = FeatureFusionModule(1024, 256, 256)

    def forward(self, x):
        # Stage 1 and 2: Basic feature extraction
        x = self.stage1(x)  # 224x224 -> 112x112
        x = self.stage2(x)  # 112x112 -> 56x56

        # Stage 3: Apply downsampling and refinement modules
        low_level_feat = self.stage3_down(x)  # Downsample: 56x56 -> 28x28
        low_level_feat = self.stage3_refine(low_level_feat)  # Refine: Keeps 28x28

        # Stage 4: Apply downsampling and refinement modules with ARM
        x = self.stage4_down(low_level_feat)  # Downsample: 28x28 -> 14x14
        x_arm4 = self.arm4(x)  # Apply attention refinement
        x = self.stage4_refine(x_arm4)  # Refine: Keeps 14x14

        # Stage 5: Apply downsampling and refinement modules with ARM
        x = self.stage5_down(x)  # Downsample: 14x14 -> 7x7
        x_arm5 = self.arm5(x)  # Apply attention refinement
        x = self.stage5_refine(x_arm5)  # Refine: Keeps 7x7

        # Pass through detail aggregation layer to reduce channels to 256
        x = self.detail_aggregation(x)

        # Fuse low-level and high-level features using FFM
        x = self.ffm(x, low_level_feat)      # Fuse stage 3 (low level)
        x = self.ffm2(x_arm5, x)             # Fuse stage 5 ARM

        return x

# STDCNet for segmentation task
class STDCNetForSegmentation(nn.Module):
    def __init__(self, num_classes=1, direction_classes=8):
        super(STDCNetForSegmentation, self).__init__()
        self.backbone = STDCNet()  # Use the STDCNet backbone
        # Final segmentation layer to reduce 256 channels to num_classes (e.g., 1 for binary segmentation)
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Forward through the backbone
        backbone_output = self.backbone(x)  # Shape: [batch_size, 256, H/32, W/32]
        # Final segmentation layer
        segmentation_output = self.final_conv(backbone_output)
        # Upsample the segmentation output to match input dimensions
        segmentation_output = F.interpolate(segmentation_output, size=x.shape[2:], mode='bilinear', align_corners=False)

        return segmentation_output


class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('image', 'mask').replace('.jpg', '.png'))

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self.apply_augmentation(image, mask)
        if self.transform:
            image, mask = self.apply_transforms(image, mask)

        return image, mask

    def apply_augmentation(self, image, mask):
        # RandomResizedCrop
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(1.0, 1.0))
        image = TF.resized_crop(image, i, j, h, w, size=(224, 224))
        mask = TF.resized_crop(mask, i, j, h, w, size=(224, 224))
        # Random flip (horizontal or vertical)
        flip_choice = random.choice(["horizontal", "vertical"])
        if flip_choice == "horizontal":
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        elif flip_choice == "vertical":
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        # RandomRotation
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        #Randomly invert the image colors with a probability of 0.5
        if random.random() > 0.5:
            image = TF.invert(image)
        # Randomly apply brightness, contrast, saturation, and hue changes to the image only
        color_jitter = transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3))
        image = color_jitter(image)

        return image, mask

    def apply_transforms(self, image, mask):
        # Resize and convert to tensor for val/test
        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224))
        # Convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0).float()

        return image, mask


# Define Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def __str__(self):
         return "Dice Loss"

    def forward(self, outputs, targets, smooth=1):
        outputs = torch.sigmoid(outputs)  # Convert logits to probability
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        intersection = (outputs * targets).sum()
        dice = (2. * intersection + smooth) / (outputs.sum() + targets.sum() + smooth)
        return 1 - dice


def dice_loss(pred, target, smooth=1e-6):
    # Flatten tensors for Dice loss calculation
    pred = pred.view(-1)
    target = target.view(-1)
    # Calculate Dice score
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    # Dice loss is 1 - Dice score
    return 1 - dice  


# Combined Binary Cross-Entropy and Dice Loss Function
class BCEWithDiceLoss(nn.Module):
    def __init__(self):
        super(BCEWithDiceLoss, self).__init__()
        self.bce_loss = nn.BCELoss() 

    def __str__(self):
         return "Binary Cross Entropy and Dice Loss"

    def forward(self, pred, target):
        # Calculate BCE loss using probabilities
        bce = self.bce_loss(pred, target)
        # Calculate Dice loss
        dice = dice_loss(pred, target)
        # Combined loss
        return bce + dice


def train_model(model, train_loader, val_loader, device, criterion, optimizer_type='Adam', learning_rate=0.001, num_epochs=20, use_scheduler=False, early_stop_patience=55, log_file="stdc_train_log.txt", augment=True):
    start_time = time.time()
    log_history = []
    log_history.append(f"### Training Begin For STDC model ###\n")
    log_history.append(f"Configuration loaded: Loss function={criterion.__class__.__name__}, optimizer={optimizer_type}, learning rate={learning_rate}, epochs={num_epochs}, batchSize={batch_size}\n")
    augment_data = 'augmented_data' if augment else 'raw_data'
    print(f"### Loaded {augment_data} ###")
    log_history.append(f"### Loaded {augment_data} ###\n")
    print(f"### Training Begin ###\nConfiguration loaded: Loss function={criterion.__class__.__name__}, optimizer={optimizer_type}, learning rate={learning_rate}, epochs={num_epochs}, batchSize={batch_size}")

    # Define the optimizer based on the selected optimizer type
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # Optionally use a ReduceLROnPlateau scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # Early stopping variables
    best_val_iou = 0
    epochs_no_improve = 0

    # Move the model to the appropriate device
    model = model.to(device)

    # Initialize variables to accumulate metrics over all epochs
    total_train_iou = 0.0
    total_train_dice = 0.0
    total_val_iou = 0.0
    total_val_dice = 0.0

    # Training loop
    for epoch in range(num_epochs):
        # ----------------------------- #
        # Training Phase
        # ----------------------------- #
        model.train()
        train_loss = 0.0
        train_iou_total = 0.0
        train_dice_total = 0.0
        train_batches = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            if isinstance(criterion, (nn.BCELoss, BCEWithDiceLoss)):
                outputs = torch.sigmoid(outputs)

            loss = criterion(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss and metrics
            train_loss += loss.item()
            current_train_iou = iou(outputs, masks)
            train_iou_total += current_train_iou
            train_dice_total += dice_coefficient(outputs, masks)
            train_batches += 1

        # Average the scores over the training set for this epoch
        avg_train_loss = train_loss / train_batches
        avg_train_iou = train_iou_total / train_batches
        avg_train_dice = train_dice_total / train_batches

        total_train_iou += avg_train_iou
        total_train_dice += avg_train_dice

        # ----------------------------- #
        # Validation Phase
        # ----------------------------- #
        model.eval()
        val_loss = 0.0
        val_iou_total = 0.0
        val_dice_total = 0.0
        val_batches = 0

        # Validation does not require gradient calculation
        with torch.no_grad():
            for val_images, val_masks in val_loader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)

                # Forward pass
                val_outputs = model(val_images)
                if isinstance(criterion, (nn.BCELoss, BCEWithDiceLoss)):
                    val_outputs = torch.sigmoid(val_outputs)

                # Calculate loss and metrics
                val_loss += criterion(val_outputs, val_masks).item()
                current_val_iou = iou(val_outputs, val_masks)
                val_iou_total += current_val_iou
                val_dice_total += dice_coefficient(val_outputs, val_masks)
                val_batches += 1

        # Average the scores over the validation set for this epoch
        avg_val_loss = val_loss / val_batches
        avg_val_iou = val_iou_total / val_batches
        avg_val_dice = val_dice_total / val_batches

        total_val_iou += avg_val_iou
        total_val_dice += avg_val_dice

        # Print the training and validation results for this epoch
        log_history.append(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}\n')
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}')

        # Step the scheduler if it's being used
        if use_scheduler:
            scheduler.step(avg_val_iou)
            current_lr = optimizer.param_groups[0]['lr']  # Get the current learning rate
            print(f"Scheduler stepping after epoch {epoch + 1}, validation IoU: {avg_val_iou:.4f}, current LR: {current_lr:.6f}")
            log_history.append(f"Scheduler stepping after epoch {epoch + 1}, validation IoU: {avg_val_iou:.4f}, current LR: {current_lr:.6f}\n")

        delta = 0.001  # Minimum change to reset early stopping counter

        # Early stopping
        if avg_val_iou > best_val_iou + delta:
            best_val_iou = avg_val_iou
            epochs_no_improve = 0  # Reset the counter if validation loss improves
            save_model(model, f"{augment_data}_best_STDC_model_{criterion.__class__.__name__}_{optimizer_type}_lr{learning_rate}_epochs{num_epochs}.pth", enable_message=False)
        else:
            epochs_no_improve += 1

        # Early stopping condition met
        if epochs_no_improve >= early_stop_patience:
            print(f"Scheduler stepping after epoch {epoch + 1}, validation IoU: {avg_val_iou:.4f}")
            log_history.append(f"Scheduler stepping after epoch {epoch + 1}, validation IoU: {avg_val_iou:.4f}\n")
            break

    actual_epochs = epoch + 1  # Use actual number of epochs, including early stopping

    # Calculate average IoU and Dice across the actual number of epochs
    avg_epoch_train_iou = total_train_iou / actual_epochs
    avg_epoch_train_dice = total_train_dice / actual_epochs
    avg_epoch_val_iou = total_val_iou / actual_epochs
    avg_epoch_val_dice = total_val_dice / actual_epochs

    # Print the overall average IoU and Dice Coefficient
    log_history.append("\n**Final Results Across All Epochs**\n")
    log_history.append(f"Overall Average Train IoU: {avg_epoch_train_iou:.4f}, Train Dice: {avg_epoch_train_dice:.4f}\n")
    log_history.append(f"Overall Average Validation IoU: {avg_epoch_val_iou:.4f}, Validation Dice: {avg_epoch_val_dice:.4f}\n")
    print("\n**Final Results Across All Epochs**")
    print(f"Overall Average Train IoU: {avg_epoch_train_iou:.4f}, Train Dice: {avg_epoch_train_dice:.4f}")
    print(f"Overall Average Validation IoU: {avg_epoch_val_iou:.4f}, Validation Dice: {avg_epoch_val_dice:.4f}")

    # Calculate total training time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"\nTraining Complete! Total Executed Time: {minutes} minutes {seconds:.2f} seconds\n")
    log_history.append(f"Finished timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n---------------------------\n\n")

    # Save the final model after all epochs or early stopping
    save_model(model, f"{augment_data}_best_STDC_model_{criterion.__class__.__name__}_{optimizer_type}_lr{learning_rate}_epochs{actual_epochs}.pth", enable_message=True)

    # Save the log history to the file
    with open(log_file, 'a') as f:
        f.writelines(log_history)

    return model


if __name__ == "__main__":
    # Define paths for train, validation, and test datasets
    train_image_dir = 'dataset/traincrop/img'
    train_mask_dir = 'dataset/traincrop/mask'
    val_image_dir = 'dataset/valcrop/img'
    val_mask_dir = 'dataset/valcrop/mask'
    test_image_dir = 'dataset/testcrop/img'
    test_mask_dir = 'dataset/testcrop/mask'
    # Create datasets for train, validation, and test
    raw_train_dataset = CrackDataset(train_image_dir, train_mask_dir, transform=True, augment=False)
    augmented_train_dataset = CrackDataset(train_image_dir, train_mask_dir, transform=True, augment=True)  # Augment=True for training
    val_dataset = CrackDataset(val_image_dir, val_mask_dir, transform=True, augment=False)       # Augment=False for validation
    test_dataset = CrackDataset(test_image_dir, test_mask_dir, transform=True, augment=False)    # Augment=False for testing
    # Set the target Batch size
    batch_size = 10
    # Create DataLoaders for each dataset
    raw_train_loader = DataLoader(raw_train_dataset, batch_size=batch_size, shuffle=True)
    augmented_train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Create the model instance
    model = STDCNetForSegmentation(num_classes=1)   # Binary segmentation

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Move the model to the appropriate device
    model = model.to(device)
    # Train the model
    train_model(model, raw_train_loader, val_loader, device, criterion=BCEWithDiceLoss(), optimizer_type='Adam', learning_rate=0.001, num_epochs=1, use_scheduler=True, augment=False)
   