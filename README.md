# BIT Vehicle Detection using YOLOv5 and classify 

## 环境准备
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
conda activate yolo10
pip install -r requirements.txt

```

## 数据准备
1. 将图像放入 `images/train` 和 `images/val`
2. 运行 `convert_labels.py` 生成 YOLO 标签



## 训练检测模型
```bash
python train.py --img 640 --batch 16 --epochs 5 \
--data config/bit_vehicle.yaml --weights yolov5s.pt \
--name bit_vehicle_detect --save-period 5
#测试检测与裁剪效果
python detect.py --weights runs/train/bit_vehicle_detect1/weights/best.pt --source test1/  --save-txt --save-crop --save-conf
python detect.py --weights runs/train/bit_vehicle_detect1/weights/best.pt --source test1/vehicle_0000009.jpg --save-txt --save-crop --save-conf

```

## 训练识别模型
```bash
1. 运行`detect_vehicles.py` 裁剪生成识别数据集
python train_classical.py
python val_classical.py
python classical.py
```
## 使用自动化系统
```bash
运行 yolov5/decl2.py
```






0 大客车
1 小客车
2 小货车
3 小轿车
4 SUV
5 大货车