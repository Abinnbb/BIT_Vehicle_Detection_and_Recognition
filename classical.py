from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
from torchvision import transforms, models
from PIL import Image
import glob
import matplotlib.pyplot as plt

class VehicleDataset(Dataset):
    def __init__(self, label_file, transform=None):
        self.samples = []
        with open(label_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 定义验证集 transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# 定义验证集的 Dataset 和 DataLoader
val_dataset = VehicleDataset("recognition_dataset/val/labels.txt", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


def predict_images_from_folder(model, folder_path, class_names, file_extension="*.jpg", device='cuda', save_dir='predicted_images'):
    """
    批量处理文件夹中的图像并返回预测类别，同时展示图像和预测标签，并保存图像。
    :param model: 训练好的模型
    :param folder_path: 文件夹路径
    :param class_names: 类别名称列表
    :param file_extension: 图像文件扩展名（默认 *.jpg）
    :param device: 使用的设备 ('cuda' 或 'cpu')
    :param save_dir: 保存预测图像的文件夹路径
    :return: 每张图像的预测类别
    """
    model.eval()
    predictions = []

    # 确保保存文件夹存在
    os.makedirs(save_dir, exist_ok=True)

    # 获取文件夹中的所有图像路径
    image_paths = glob.glob(os.path.join(folder_path, file_extension))

    # 预处理图像
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    plt.figure(figsize=(15, 10))  # 设置显示图像的大小

    # 遍历每张图像进行预测
    for i, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)  # 获取预测类别
            predicted_class = class_names[predicted.item()]
            predictions.append((img_path, predicted_class))

            # 可视化图像和预测标签
            plt.subplot(3, 5, i + 1)  # 3行5列的布局
            plt.imshow(image)
            plt.title(f"Predicted: {predicted_class}")
            plt.axis('off')

            # 保存带预测标签的图像
            if (i + 1) % 5 == 0:
                save_path = os.path.join(save_dir, f"predicted_{i+1}.jpg")
                plt.savefig(save_path)

        # 每隔 5 张图片显示一次
        if (i + 1) % 5 == 0:
            plt.tight_layout()
            plt.show()

    # 展示所有图像后显示一次
    save_path = os.path.join(save_dir, f"predicted_{i+1}.jpg")
    plt.savefig(save_path)
    plt.tight_layout()
    plt.show()
    print(f"识别完成: 共{i+1}张")

    return predictions

# 使用示例
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 6)  # Assuming 6 classes
model.load_state_dict(torch.load("modelbest/vehicle_classifier2.pt"))
model.to('cuda' if torch.cuda.is_available() else 'cpu')

class_names = ['Bus', 'Microbus', 'Minivan', 'Sedan', 'SUV', 'Truck']
folder_path = "yolov5/runs/detect/exp7/crops"  # 请更改为你的文件夹路径
save_dir = "predicted_images2"  # 保存预测图像的目录

# 测试文件夹中的所有图像并保存
predictions = predict_images_from_folder(model, folder_path, class_names, save_dir=save_dir)
