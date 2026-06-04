import os
import scipy.io

CATEGORY_MAP = {
    "Bus": 0,
    "Microbus": 1,
    "Minivan": 2,
    "Sedan": 3,
    "SUV": 4,
    "Truck": 5
}


def convert_to_yolo_labels(mat_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data = scipy.io.loadmat(mat_path)
    info = data['VehicleInfo']

    for i in range(len(info)):
        item = info[i][0]
        name = item['name'][0]
        width = item['width'][0][0]
        height = item['height'][0][0]
        vehicles = item['vehicles'][0]

        label_lines = []
        for v in vehicles:
            left = v['left'][0][0]
            top = v['top'][0][0]
            right = v['right'][0][0]
            bottom = v['bottom'][0][0]
            category_str = v['category'][0]
            if category_str not in CATEGORY_MAP:
                continue
            cls = CATEGORY_MAP[category_str]

            x_center = ((left + right) / 2) / width
            y_center = ((top + bottom) / 2) / height
            bbox_width = (right - left) / width
            bbox_height = (bottom - top) / height

            label_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        if label_lines:  # 有车时才保存
            with open(os.path.join(output_dir, name.replace(".jpg", ".txt")), "w") as f:
                f.write("\n".join(label_lines))


# 使用示例
convert_to_yolo_labels("BIT-Vehicle/VehicleInfo.mat", "../yolov5/labels/train")
