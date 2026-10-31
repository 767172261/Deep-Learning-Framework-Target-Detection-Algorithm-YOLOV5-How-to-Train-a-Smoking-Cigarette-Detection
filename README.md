深度学习框架目标检测算法YOLOV5如何训练吸烟香烟检测数据集通过训练的权重推理识别 检测吸烟者中的香烟
@[以下文字及代码仅供参考学习使用]

**数据集描述：**

 ***6000多张，已标注txt格式和xml格式，可直接应用yolo目标检测模型的训练***
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b08d65634b374256a19898a0776f1ecc.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ae90175c0cf04e109892247b97fc155d.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7b514f62d82d4dd1a999868d3b37fd6b.jpeg)

：**6000+张标注好的“smoke”类香烟检测数据集，已划分训练/验证集，目标明确（检测吸烟行为中的香烟）**。：

> ✅ **YOLOv5 依然非常稳定、高效、部署方便，尤其适合工业级目标检测任务**，完全适用于你的“香烟检测”项目。
 **环境搭建 → 数据准备 → YOLOv5 训练 → 推理识别 → 模型导出** 的完整流程，基于官方 [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) 仓库。

---

## 🚀 一、环境搭建（CUDA + Conda + PyTorch）

### 1. 检查 CUDA 驱动

```bash
nvidia-smi
```

- 确保 GPU 正常工作
- 推荐 CUDA 11.8 或 12.1

---

### 2. 安装 Anaconda

下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

---

### 3. 创建虚拟环境并安装依赖

```bash
# 创建环境
conda create -n smoke_detection python=3.9

# 激活环境
conda activate smoke_detection

# 克隆 YOLOv5 官方仓库
git clone https://github.com/ultralytics/yolov5.git
cd yolov5

# 安装依赖
pip install -r requirements.txt
```

✅ 验证 PyTorch 和 GPU：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 应输出 True
```

---

## 📂 二、数据集结构（你已划分好）

假设你的数据集结构如下：

```
smoke_dataset/
├── images/
│   ├── train/        # 4800 张（80%）
│   └── val/          # 1200 张（20%）
├── labels/
│   ├── train/        # 对应 .txt
│   └── val/
├── data.yaml         # 需创建
```

---

## 📄 三、创建 `data.yaml` 文件

```yaml
# data.yaml
train: ../smoke_dataset/images/train
val: ../smoke_dataset/images/val

nc: 1
names: ['smoke']
```

> ⚠️ 路径根据你的实际路径调整，如果是相对路径，请确保在 `yolov5/` 目录下能正确访问。

---

## 🚀 四、训练命令（使用官方预训练权重）

YOLOv5 提供了多个预训练模型（`yolov5s.pt`, `yolov5m.pt`, `yolov5l.pt`, `yolov5x.pt`），推荐使用 `yolov5s` 或 `yolov5m`。

### 下载预训练权重

```bash
# 在 yolov5/ 目录下执行
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt
# 或
wget https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt
```

---

### 开始训练（`train.py`）

```bash
python train.py \
--img 640 \
--batch 16 \
--epochs 300 \
--data data.yaml \
--weights yolov5s.pt \
--cfg models/yolov5s.yaml \
--name smoke_yolov5s_640 \
--project runs/smoke \
--device 0 \
--workers 4 \
--cache disk \
--augment \
--mosaic 1.0 \
--copy-paste 0.3 \
--close-mosaic 25
```

#### 参数说明：

| 参数 | 说明 |
|------|------|
| `--img 640` | 输入图像大小 |
| `--batch 16` | 批次大小（根据显存调整） |
| `--epochs 300` | 训练轮数 |
| `--data data.yaml` | 数据配置文件 |
| `--weights yolov5s.pt` | 预训练权重 |
| `--cfg` | 模型结构（s/m/l/x） |
| `--cache disk` | 缓存图像到磁盘，加速训练 |
| `--mosaic`, `--copy-paste` | 数据增强，提升小目标检测能力 |
| `--close-mosaic 25` | 最后25轮关闭 Mosaic，稳定收敛 |

✅ 训练完成后，最佳权重保存在：  
`runs/smoke/smoke_yolov5s_640/weights/best.pt`

---

## 🔍 五、推理代码（`detect.py` 或自定义脚本）

### 方法一：使用命令行检测图像/视频

```bash
python detect.py \
--weights runs/smoke/smoke_yolov5s_640/weights/best.pt \
--source test.jpg \
--imgsz 640 \
--conf-thres 0.4 \
--iou-thres 0.45 \
--device 0 \
--project runs/detect/smoke \
--name result \
--save-txt \
--save-conf
```

支持：
- `--source test.jpg`：单张图像
- `--source video.mp4`：视频文件
- `--source 0`：调用摄像头

---

### 方法二：自定义推理脚本（`infer.py`）

```python
# infer.py
import torch
import cv2

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/smoke/smoke_yolov5s_640/weights/best.pt')

# 图像检测
results = model('test.jpg')
results.save()  # 保存结果到 runs/detect/exp/

# 显示结果
results.print()
print(results.pandas().xyxy[0])  # 输出检测框 [xmin, ymin, xmax, ymax, confidence, class]

# 视频或摄像头检测
results = model('0')  # 0 表示摄像头
results.show()
```

---

## 📊 六、评估模型性能（`val.py`）

```bash
python val.py \
--weights runs/smoke/smoke_yolov5s_640/weights/best.pt \
--data data.yaml \
--img 640 \
--batch 16 \
--device 0 \
--project runs/val/smoke \
--name result
```

输出：
- mAP@50: 0.9+（理想情况下）
- Precision, Recall, F1-score
- 各类别的 AP（你只有一类，所以是 `smoke` 的 AP）

---

## 🏁 七、主流程脚本（`main.py`）

```python
# main.py
import os

def main():
    print("🚬 香烟检测系统启动...")

    print("🏋️ 正在训练模型...")
    os.system("""
        python train.py \
        --img 640 \
        --batch 16 \
        --epochs 300 \
        --data data.yaml \
        --weights yolov5s.pt \
        --cfg models/yolov5s.yaml \
        --name smoke_yolov5s_640 \
        --project runs/smoke \
        --device 0 \
        --cache disk \
        --augment \
        --mosaic 1.0 \
        --copy-paste 0.3
    """)

    print("📊 正在评估模型...")
    os.system("""
        python val.py \
        --weights runs/smoke/smoke_yolov5s_640/weights/best.pt \
        --data data.yaml \
        --img 640 \
        --device 0
    """)

    print("🔍 正在推理测试...")
    os.system("""
        python detect.py \
        --weights runs/smoke/smoke_yolov5s_640/weights/best.pt \
        --source test.jpg \
        --imgsz 640 \
        --conf-thres 0.4 \
        --device 0
    """)

    print("✅ 全流程完成！")

if __name__ == "__main__":
    main()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/65b5c6e4929349b7b1776c3c094e526f.png)

---

## 📦 八、模型导出（ONNX / TensorRT / CoreML）

```bash
# 导出为 ONNX（用于部署到边缘设备）
python export.py \
--weights runs/smoke/smoke_yolov5s_640/weights/best.pt \
--include onnx \
--imgsz 640 \
--device 0
```

输出：`best.onnx`，可用于：
- OpenVINO（Intel）
- TensorRT（NVIDIA Jetson）
- ONNX Runtime（跨平台）

---


