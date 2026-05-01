import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ================= 1. 深度遍历与数据加载 =================
DATA_PATH = os.path.join('dataset')

if not os.path.exists(DATA_PATH):
    print(f"❌ 错误：找不到 {DATA_PATH} 文件夹！")
    exit()

actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
print(f"📂 发现 {len(actions)} 个动作分类：{actions}")
print("-" * 30)

sequence_length = 60
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    seq_folders = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]

    valid_seq_count = 0
    for seq_folder in seq_folders:
        seq_path = os.path.join(action_path, seq_folder)
        window = []
        frame_files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]

        try:
            frame_files.sort(key=lambda x: int(x.split('.')[0]))
        except ValueError:
            pass

        if len(frame_files) == sequence_length:
            for frame_file in frame_files:
                res = np.load(os.path.join(seq_path, frame_file))


                # 保护神经网络不被巨大的数字崩坏
                res[res == -5.0] = 0.0

                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
            valid_seq_count += 1

    print(f"  -> 动作 '{action}' : 成功拼接 {valid_seq_count} 组完整数据")

print("-" * 30)

if len(sequences) == 0:
    print("\n🚨 致命错误：没有提取到完整的 60 帧数据！")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"✅ 数据清洗并打包完毕！输入矩阵 X 形状: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# ================= 2. 搭建进阶版 LSTM 模型 =================
print("🧠 正在搭建进阶版 LSTM 神经网络...")
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 84)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # 这个机制会自动解决单双手混淆！
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ================= 3. 调用显卡开始训练 =================
print("🚀 开启 RTX 3050 显卡加速训练！")
early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=25, restore_best_weights=True, verbose=1)
model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), callbacks=[early_stop])

# ================= 4. 保存最佳模型 =================
model.save('action.h5')
print("🎉 训练大功告成！全新健康的 action.h5 大脑已保存！")