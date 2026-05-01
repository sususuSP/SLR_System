import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

print("====================================")
print("🧠 正在启动 LSTM 神经网络训练核心...")
print("====================================")

# ================= 1. 载入打包好的数据 =================
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# 自动扫描 dataset 获取动作名称
DATA_PATH = os.path.join('dataset')
actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
num_classes = len(actions)

print(f"✅ 数据载入成功！准备让 AI 学习这 {num_classes} 个动作: {actions}")

# ================= 2. 搭建 LSTM 大脑架构 =================
model = Sequential()

# 第一层 LSTM：接收 60 帧，每帧 84 个骨架坐标，提取时间序列特征
# return_sequences=True 表示把每一帧的特征都往后传
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(60, 84)))

# 第二层 LSTM：进一步提炼更深层的运动规律
model.add(LSTM(128, return_sequences=True, activation='relu'))

# 第三层 LSTM：浓缩核心特征
model.add(LSTM(64, return_sequences=False, activation='relu'))

# 添加两个全连接层（可以理解为最后的分类器）
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# 最后一层：输出层。有几个词就有几个神经元，softmax 负责算出每个词的概率
model.add(Dense(num_classes, activation='softmax'))

# ================= 3. 编译与日志设置 =================
# Adam 优化器是目前最经典的配置
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 设置 TensorBoard 日志保存路径 (以后写毕业论文时，可以用这个画出好看的训练曲线)
log_dir = os.path.join('Logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb_callback = TensorBoard(log_dir=log_dir)

print("\n🚀 开始炼丹！(让 AI 反复看数据 100 遍...)")

# ================= 4. 开始训练 =================
# epochs=100 表示训练 100 轮
model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])

print("\n✅ 训练完成！正在导出模型...")

# ================= 5. 保存模型文件 =================
model.save('action.h5')
print("🎉 恭喜！你的第一个 AI 手语模型 (action.h5) 已经成功诞生！")
print("====================================")