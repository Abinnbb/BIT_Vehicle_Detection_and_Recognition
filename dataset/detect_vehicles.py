import os
import cv2

def crop_from_yolo_label(image_dir, label_dir, output_image_dir, output_label_file):
    os.makedirs(output_image_dir, exist_ok=True)
    label_output = open(output_label_file, "w")  # 分类任务用的全局标签文件

    vehicle_id = 0  # 裁剪图编号

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue

        # 读取图像和标签路径
        image_name = label_file.replace(".txt", ".jpg")
        img_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(img_path):
            print(f"图像不存在: {img_path}")
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                cls, x, y, bw, bh = map(float, line.strip().split())

                # 将 YOLO 格式转换为像素坐标
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)

                # 边界检查
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # 裁剪并保存图像
                cropped = img[y1:y2, x1:x2]
                crop_name = f"vehicle_{vehicle_id}.jpg"
                crop_path = os.path.join(output_image_dir, crop_name)
                cv2.imwrite(crop_path, cropped)

                # 写入识别用标签：图像路径 + 类别
                label_output.write(f"{crop_path} {int(cls)}\n")

                vehicle_id += 1

    label_output.close()
    print(f"完成裁剪，共生成图像 {vehicle_id} 张，标签写入 {output_label_file}")

# 示例调用
crop_from_yolo_label(
    image_dir="yolov5/images/val",            # 原图目录
    label_dir="yolov5/labels/val",            # 标签目录（已由 VehicleInfo.mat 转换）
    output_image_dir="recognition_dataset/val/images",   # 保存裁剪图像
    output_label_file="recognition_dataset/val/labels.txt"  # 分类训练用标签
)
