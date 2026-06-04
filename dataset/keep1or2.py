import os
import scipy.io
import shutil

CATEGORY_MAP = {
    "Bus": 0,
    "Microbus": 1,
    "Minivan": 2,
    "Sedan": 3,
    "SUV": 4,
    "Truck": 5
}

def convert_keep_1_or_2_and_save_skipped(mat_path, label_output_dir, image_input_dir, image_output_dir, skipped_dir):
    os.makedirs(label_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(skipped_dir, exist_ok=True)

    data = scipy.io.loadmat(mat_path)
    info = data['VehicleInfo']

    keep_count = 0
    skip_count = 0
    total = len(info)

    for i in range(total):
        item = info[i][0]
        name = item['name'][0]
        width = item['width'][0][0]
        height = item['height'][0][0]
        vehicles = item['vehicles'][0]
        n = len(vehicles)

        image_path = os.path.join(image_input_dir, name)
        skip_reason = ""

        if not os.path.exists(image_path):
            skip_reason = "file_not_found"
        elif n > 2:
            skip_reason = "too_many_vehicles"

        if skip_reason:
            skip_count += 1
            skipped_path = os.path.join(skipped_dir, name)
            if os.path.exists(image_path):  # 避免不存在时复制报错
                shutil.copy(image_path, skipped_path)
            continue

        label_lines = []
        for v in vehicles:
            category_str = v['category'][0]
            if category_str not in CATEGORY_MAP:
                continue
            cls = CATEGORY_MAP[category_str]

            left = v['left'][0][0]
            top = v['top'][0][0]
            right = v['right'][0][0]
            bottom = v['bottom'][0][0]

            x_center = ((left + right) / 2) / width
            y_center = ((top + bottom) / 2) / height
            bbox_width = (right - left) / width
            bbox_height = (bottom - top) / height

            label_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        if label_lines:
            with open(os.path.join(label_output_dir, name.replace(".jpg", ".txt")), "w") as f:
                f.write("\n".join(label_lines))
            shutil.copy(image_path, os.path.join(image_output_dir, name))
            keep_count += 1

    print(f"✅ 保留图像（1~2车）: {keep_count} 张")
    print(f"❌ 被跳过图像: {skip_count} 张，已保存到: {skipped_dir}")

# 使用示例
convert_keep_1_or_2_and_save_skipped(
    mat_path="BIT-Vehicle/VehicleInfo.mat",
    label_output_dir="../yolov5/labels/val",
    image_input_dir="BIT-Vehicle/images_0/val",
    image_output_dir="../yolov5/images/val",
    skipped_dir="yolov5/images/skipped"
)
