import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# 可视化样本图像
def visualize_sample(dataset, class_names=None, samples=5, save_path="modelbest/sample_preview.jpg"):
    idxs = random.sample(range(len(dataset)), samples)
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(idxs):
        img, label = dataset[idx]
        img = img.permute(1, 2, 0) * 0.5 + 0.5
        img = img.clamp(0, 1)
        plt.subplot(1, samples, i+1)
        plt.imshow(img)
        title = f"class: {label}"
        if class_names:
            title = f"class: {class_names[label]}"
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 已保存图像样本预览至 {save_path}")
    plt.close()

# 可视化训练曲线
def plot_training_curve(history, save_path="modelbest/training_plot.png"):
    epochs = len(history['train_loss'])
    x = list(range(1, epochs + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, history['train_loss'], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, history['train_acc'], label="Train Acc")
    plt.plot(x, history['val_acc'], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 已保存训练曲线图至: {save_path}")
    plt.close()

# 自定义数据集
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

# 主训练函数（含早停止）
def train_model(train_loader, val_loader, num_classes, epochs=30, save_path="modelbest/vehicle_classifier2.pt", patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # save_path = "model_5/vehicle_classifier_best.pt"
    # if os.path.exists(save_path):
    #     print(f"📥 检测到已保存的模型，加载: {save_path}")
    #     model.load_state_dict(torch.load(save_path, map_location=device))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_epoch = 0

    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total_samples = 0

            print(f"\\n🟢 Epoch {epoch+1}/{epochs} - Training:")
            pbar = tqdm(train_loader, desc="Train", ncols=100)

            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += batch_size

                avg_loss = total_loss / total_samples
                acc = correct / total_samples
                pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

            train_acc = correct / len(train_loader.dataset)
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            print(f"📘 Epoch Summary - Train Loss: {avg_loss:.3f}, Accuracy: {train_acc:.3f}")

            # 验证阶段
            model.eval()
            val_correct = 0
            total_val = 0
            print("🔵 Validation:")
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Val", ncols=100):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    total_val += labels.size(0)

            val_acc = val_correct / total_val
            history['val_acc'].append(val_acc)
            print(f"✅ Validation Accuracy: {val_acc:.3f}")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), save_path)
                print(f"🏅 新最佳模型已保存: {save_path}")
            elif epoch - best_epoch >= patience:
                print(f"⏹ 提前停止：验证集准确率在 {patience} 个 epoch 内未提升。")
                break

    except KeyboardInterrupt:
        print("\\n⛔️ 中断检测（Ctrl+C），保存当前最佳模型...")
        torch.save(model.state_dict(), save_path)
        print(f"✅ 模型已保存至: {save_path}")

    print(f"\\n🎯 训练完成，最佳验证准确率: {best_val_acc:.4f}")
    plot_training_curve(history)
    return model

# 主函数
def main():
    num_classes = 6
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = VehicleDataset("recognition_dataset/train/labels.txt", transform=transform)
    val_dataset = VehicleDataset("recognition_dataset/val/labels.txt", transform=transform)

    # 可视化样本
    visualize_sample(train_dataset, class_names=["Bus", "Microbus", "Minivan", "Sedan", "SUV", "Truck"])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    train_model(train_loader, val_loader, num_classes)

if __name__ == "__main__":
    main()