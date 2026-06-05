# SLR_System — 实时手语识别系统

基于 YOLOv8 + MediaPipe + CNN-LSTM 的实时中文手语识别系统。通过摄像头捕捉手部动作，提取三维骨骼关键点序列，经混合神经网络分类后在图形界面中实时输出对应的中文词条。

---

## 系统架构

```
摄像头画面
    │
    ▼
YOLOv8n（人体检测，裁剪出人体区域）
    │
    ▼
MediaPipe Hands（提取双手 21 个三维关键点）
    │
    ▼
126 维特征向量（左手 63 维 + 右手 63 维）× 40 帧
    │
    ▼
Conv1D → MaxPooling → LSTM → Dense（动作分类）
    │
    ▼
PyQt5 图形界面（实时显示识别结果并拼接为句子）
```

---

## 目录结构

```
SLR_System/
├── yolo_data_collector.py   # 数据采集脚本
├── train_model_cnn_lstm.py  # 模型训练脚本
├── main_gui.py              # 实时识别图形界面
├── action_cnn_3d.h5         # 训练好的模型文件（训练后生成）
├── yolov8n.pt               # YOLOv8n 预训练权重（首次运行自动下载）
└── dataset/                 # 关键点数据集
    ├── good/
    │   ├── 0/
    │   │   ├── 0.npy
    │   │   └── ...          # 共 40 帧
    │   └── ...              # 共 40 组
    ├── idle/
    └── ...                  # 其余词条同上结构
```

---

## 环境要求

- Python 3.9
- CUDA 环境（可选，有 GPU 时推理更流畅，CPU 也可运行）

### 安装依赖

```bash
pip install opencv-python mediapipe numpy scikit-learn
pip install tensorflow==2.10.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install ultralytics
pip install PyQt5
```

> **注意**：`tensorflow` 与 `torch` 的版本需与本机 CUDA 版本对应，请根据实际环境调整。
> TensorFlow 2.10 是最后一个原生支持 Windows GPU 的版本，Linux 用户可使用更新版本。

---

## 使用流程

### 第一步：采集数据

打开 `yolo_data_collector.py`，修改顶部的词条配置：

```python
TARGET_ACTION = 'hello'   # 当前录制的词条代号，与 dataset 子目录名一致
NO_SEQUENCES = 40         # 每个词条录制的样本组数
SEQUENCE_LENGTH = 40      # 每组样本的帧数，须与训练脚本保持一致
```

运行脚本：

```bash
python yolo_data_collector.py
```

**操作说明：**

| 按键 | 功能 |
|------|------|
| `s` | 开始录制当前组 |
| `r` | 撤销上一组，重新录制 |
| `q` | 退出采集 |

每个词条需重复以上步骤，修改 `TARGET_ACTION` 后重新运行。数据保存在 `dataset/<词条代号>/` 下。

> **idle 类说明**：`idle` 是一个特殊词条，用于表示"无有效手语动作"的状态。采集时让双手自然垂放或不出现在画面中即可，建议与其他词条采集相同数量的样本。

---

### 第二步：训练模型

```bash
python train_model_cnn_lstm.py
```

脚本会自动扫描 `dataset/` 目录下的所有词条，进行数据加载、增强和训练。训练完成后在当前目录生成 `action_cnn_3d.h5`。

**训练策略说明：**
- 数据增强：对训练集施加两种强度的高斯噪声，训练样本扩充为原来的 3 倍
- EarlyStopping：验证集准确率连续 25 轮不提升时提前停止，自动恢复最优权重
- ReduceLROnPlateau：验证集损失连续 10 轮不下降时学习率减半，下限 1e-5

---

### 第三步：运行识别界面

```bash
python main_gui.py
```

点击「开启摄像头」按钮，在镜头前做手语动作，识别到的词条会实时显示并自动拼接为句子。点击「清空句子」可重新开始。

---

## 新增词条

1. 在 `yolo_data_collector.py` 中设置新的 `TARGET_ACTION`，完成采集
2. 在 `main_gui.py` 的 `SIGN_TRANSLATOR` 字典中添加对应的中文映射：
   ```python
   SIGN_TRANSLATOR = {
       ...
       'hello': '你好',  # 新增词条
   }
   ```
3. 重新运行 `train_model_cnn_lstm.py` 训练新模型

---

## 模型结构

| 层 | 参数 | 作用 |
|----|------|------|
| Conv1D | filters=64, kernel_size=3 | 提取相邻帧间的局部运动特征 |
| MaxPooling1D | pool_size=2 | 降采样，保留显著特征 |
| Conv1D | filters=128, kernel_size=3 | 提取更抽象的运动模式 |
| MaxPooling1D | pool_size=2 | 再次降采样 |
| LSTM | units=128, return_sequences=True | 建模长程时序依赖 |
| Dropout | rate=0.3 | 防止过拟合 |
| LSTM | units=64, return_sequences=False | 将时序压缩为固定长度向量 |
| Dropout | rate=0.3 | 防止过拟合 |
| Dense | units=64, activation=relu | 整合特征 |
| Dense | units=类别数, activation=softmax | 输出各类别概率 |

输入形状：`(batch_size, 40, 126)`，即 40 帧 × 126 维关键点。

---

## 常见问题

**Q：运行 `main_gui.py` 提示找不到 `action_cnn_3d.h5`**
A：请先运行 `train_model_cnn_lstm.py` 完成训练，生成模型文件后再启动界面。

**Q：摄像头画面正常但始终识别不到动作**
A：确认手部在画面中清晰可见，且 YOLO 已检测到人体（画面中有绿色边界框）。识别需积累满 40 帧有效手部数据后才会触发一次推理。

**Q：新词条加入后，旧词条识别结果错乱**
A：`os.listdir()` 返回的目录顺序与训练时不一致会导致标签错位。建议在 `train_model_cnn_lstm.py` 和 `main_gui.py` 中均将扫描目录的代码改为 `sorted(os.listdir(...))`，保证标签顺序固定。

**Q：CPU 运行时帧率较低**
A：YOLO 已设置为每 5 帧推理一次以降低计算压力，但 CPU 下整体仍会较慢。建议使用带独立显卡的设备运行。

---

## 技术栈

| 组件 | 用途 |
|------|------|
| [YOLOv8n](https://github.com/ultralytics/ultralytics) | 人体检测与区域裁剪 |
| [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands) | 手部三维关键点提取 |
| [TensorFlow / Keras](https://www.tensorflow.org/) | CNN-LSTM 模型训练与推理 |
| [PyTorch](https://pytorch.org/) | YOLO 运行后端 |
| [PyQt5](https://pypi.org/project/PyQt5/) | 图形界面 |
| [OpenCV](https://opencv.org/) | 摄像头读取与图像处理 |
