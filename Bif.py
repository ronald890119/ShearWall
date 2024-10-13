import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import sys
import torch.nn.functional as F
import torch.nn as nn
import time  # 用于计算时间
from tqdm import tqdm  # 用于显示进度条
from torch.cuda.amp import GradScaler, autocast  # 用于混合精度训练

sys.path.append('D:\\syx\\Bif\\BiRefNet-main\\models')
sys.path.append('D:\\syx\\Bif\\BiRefNet-main')

from birefnet import BiRefNet  # 导入 BiRefNet 模型

# 定义图像和标签的转换，使用较小的图像尺寸
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整为 128x128 来加速训练
    transforms.ToTensor(),
])

label_transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整为 128x128 来加速训练
    transforms.ToTensor(),  # 标签也转换为张量
])

# 创建自定义Dataset类用于加载图像和标签
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_transform=None, label_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = os.listdir(img_dir)
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 加载图像和标签
        image = Image.open(img_path).convert("RGB")
        label = Image.open(mask_path).convert("L")  # 单通道标签

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        # 将标签中255的值替换为1，便于二分类
        label = np.array(label, dtype=np.int64)
        label[label == 255] = 1  # 将255替换为1
        label = torch.tensor(label, dtype=torch.long)  # 确保标签为 long 类型

        return image, label

# 定义IoU和Dice系数的计算函数
def calculate_iou(pred, target, num_classes):
    iou = []
    for cls in range(num_classes):
        pred_class = (pred == cls)
        target_class = (target == cls)
        intersection = (pred_class & target_class).sum().float().item()
        union = (pred_class | target_class).sum().float().item()
        if union > 0:
            iou.append(intersection / union)
    return np.mean(iou)

def calculate_dice(pred, target, num_classes):
    dice = []
    for cls in range(num_classes):
        pred_class = (pred == cls)
        target_class = (target == cls)
        intersection = (pred_class & target_class).sum().float().item()
        pred_sum = pred_class.sum().float().item()
        target_sum = target_class.sum().float().item()
        if pred_sum + target_sum == 0:
            dice_score = 0.0
        else:
            dice_score = (2 * intersection) / (pred_sum + target_sum)
        dice.append(dice_score)
    return np.mean(dice)

# 加载训练数据集和测试数据集
train_dataset = CustomImageDataset(
    img_dir='D:/syx/Bif/train_img/timg', 
    mask_dir='D:/syx/Bif/train_mask/tmask', 
    image_transform=image_transform, 
    label_transform=label_transform
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # 使用较小的批量大小

test_dataset = CustomImageDataset(
    img_dir='D:/syx/Bif/test_img', 
    mask_dir='D:/syx/Bif/test_mask/mask', 
    image_transform=image_transform, 
    label_transform=label_transform
)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化并加载BiRefNet模型
model = BiRefNet(bb_pretrained=True).to(device)

# 定义损失函数和优化器
num_classes = 2  
criterion = torch.nn.CrossEntropyLoss()  # 用于语义分割的损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 使用Adam优化器

# 使用梯度缩放器 (GradScaler) 和混合精度训练 (autocast)
scaler = GradScaler()

# 使用梯度累积来模拟较大的批量大小
accumulation_steps = 4  # 每4个小批量累加一次梯度

# 训练模型
num_epochs = 1  # 设定epoch数量
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    total_train_iou = 0.0
    total_train_dice = 0.0

    start_time = time.time()  # 记录开始时间
    optimizer.zero_grad()  # 在每个 epoch 开始时初始化梯度

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
        images = images.to(device)
        labels = labels.to(device).squeeze(1)  # 去掉多余的维度 (batch_size, 1, h, w) -> (batch_size, h, w)

        # 使用 autocast 进行混合精度训练
        with autocast():
            outputs = model(images)

            # 提取张量
            while isinstance(outputs, (list, tuple)):
                outputs = outputs[0]  # 提取出第一个张量
            if outputs.shape[1] == 1:
                outputs = outputs.repeat(1, num_classes, 1, 1)

            # 上采样，将输出尺寸调整为与目标标签相同
            outputs = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=True)

            # 计算损失
            loss = criterion(outputs, labels)

        # 梯度累积
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # 计算损失
        running_loss += loss.item()

        # 计算训练集上的 IoU 和 Dice
        predicted = outputs.argmax(dim=1)
        iou = calculate_iou(predicted, labels, num_classes=num_classes)
        dice = calculate_dice(predicted, labels, num_classes=num_classes)

        total_train_iou += iou
        total_train_dice += dice

    avg_train_loss = running_loss / len(train_loader)
    avg_train_iou = total_train_iou / len(train_loader)
    avg_train_dice = total_train_dice / len(train_loader)

    # 计算每个 epoch 的时间
    end_time = time.time()
    epoch_time = end_time - start_time

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}, Train Dice: {avg_train_dice:.4f}, Time: {epoch_time:.2f}s")
   
    # === 测试阶段 ===
    model.eval()  # 设置模型为评估模式
    total_test_iou = 0.0
    total_test_dice = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1)

            outputs = model(images)

            # 提取并上采样
            while isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            outputs = F.interpolate(outputs, size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=True)

            # 计算测试集上的 IoU 和 Dice
            predicted = outputs.argmax(dim=1)
            iou = calculate_iou(predicted, labels, num_classes=num_classes)
            dice = calculate_dice(predicted, labels, num_classes=num_classes)

            total_test_iou += iou
            total_test_dice += dice

        avg_test_iou = total_test_iou / len(test_loader)
        avg_test_dice = total_test_dice / len(test_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Test IoU: {avg_test_iou:.4f}, Test Dice: {avg_test_dice:.4f}")
import matplotlib.pyplot as plt

def visualize_predictions(images, labels, preds, epoch):
    images = images.cpu().numpy().transpose(0, 2, 3, 1)  # 将图像转换为 numpy 格式
    labels = labels.cpu().numpy()  # 标签
    preds = preds.cpu().numpy()  # 模型预测
    
    fig, axes = plt.subplots(len(images), 3, figsize=(10, len(images) * 3))
    
    for i in range(len(images)):
        if len(images) == 1:  # 如果只有一张图片
            ax_img = axes[0]
            ax_label = axes[1]
            ax_pred = axes[2]
        else:
            ax_img = axes[i, 0]
            ax_label = axes[i, 1]
            ax_pred = axes[i, 2]
        
        # 原始图像
        ax_img.imshow(images[i])
        ax_img.set_title(f"Epoch {epoch}: Image")
        ax_img.axis('off')
        
        # 真实标签
        ax_label.imshow(labels[i], cmap='gray')
        ax_label.set_title(f"Epoch {epoch}: Label")
        ax_label.axis('off')
        
        # 模型预测
        ax_pred.imshow(preds[i], cmap='gray')
        ax_pred.set_title(f"Epoch {epoch}: Prediction")
        ax_pred.axis('off')
    
    plt.tight_layout()
    plt.show()

# 在每个 epoch 完成时可视化预测结果
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).squeeze(1)

        outputs = model(images)

        # 提取并上采样
        while isinstance(outputs, (list, tuple)):
            outputs = outputs[0]
        outputs = F.interpolate(outputs, size=(labels.shape[1], labels.shape[2]), mode='bilinear', align_corners=True)

        # 获取模型的预测
        predicted = outputs.argmax(dim=1)

        # 可视化预测结果（随机选择一些样本进行展示）
        visualize_predictions(images[:4], labels[:4], predicted[:4], epoch)  # 只显示前4个样本
        
        break  # 只显示一次（去掉这个break，如果你想显示所有样本）
