import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm

# 类别名称
class_names = ["Bus", "Microbus", "Minivan", "Sedan", "SUV", "Truck"]

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# 自定义数据集
class VehicleDataset(torch.utils.data.Dataset):
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
        return image, label, img_path  # 返回 img_path 以确保输出正确路径

# 验证函数
def validate_model(model, dataloader, device, output_file="validation_results.txt"):
    """
    对验证集进行推理，计算分类准确率并保存结果
    """
    model.eval()
    correct = 0
    total = 0

    with open(output_file, 'w') as f:
        f.write("Image, Predicted Class, True Class\n")

        with torch.no_grad():
            for images, labels, img_paths in tqdm(dataloader, desc="Validating", ncols=100):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                # 统计准确率
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # 保存结果 (逐行写入，避免路径索引错位)
                for i in range(len(images)):
                    img_path = img_paths[i]  # 直接使用 img_paths 中的路径，避免索引错位
                    pred_label = preds[i].item()
                    true_label = labels[i].item()
                    pred_class = class_names[pred_label]
                    true_class = class_names[true_label]
                    f.write(f"{img_path}, {pred_class}, {true_class}\n")

    accuracy = correct / total if total > 0 else 0
    print(f"✅ 验证集分类准确率: {accuracy:.4f}")
    print(f"📄 验证结果已保存到: {output_file}")

    return accuracy

# 主函数
def main():
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_path = "yolov5/modelbest/vehicle_classifier2.pt"
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"✅ 模型已加载: {model_path}")

    # 数据集
    val_dataset = VehicleDataset("recognition_dataset/val/labels.txt", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 开始验证
    validate_model(model, val_loader, device)

if __name__ == "__main__":
    main()
