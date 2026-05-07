import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("====================================")
print("🧠 启动 3D 轻量级 CNN + LSTM 混合神经网络...")
print("====================================")

DATA_PATH = os.path.join('dataset')

if not os.path.exists(DATA_PATH):
    print(f"❌ 找不到 {DATA_PATH} 文件夹！")
    exit()

actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])
print(f"📂 发现 {len(actions)} 个动作分类：{actions}")

sequence_length = 40
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    seq_folders = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))]
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
                res[np.isnan(res)] = 0.0
                res[res == -5.0] = 0.0
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

if len(sequences) == 0:
    print("\n🚨 没有提取到有效数据！")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


def add_noise(data, noise_factor=0.015):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=data.shape)
    return np.where(data == 0.0, 0.0, data + noise)


X_train_noisy_1 = add_noise(X_train.copy(), noise_factor=0.01)
X_train_noisy_2 = add_noise(X_train.copy(), noise_factor=0.02)

X_train_augmented = np.concatenate((X_train, X_train_noisy_1, X_train_noisy_2))
y_train_augmented = np.concatenate((y_train, y_train, y_train))

print(f"📈 增强完毕！训练数据暴增到了 {X_train_augmented.shape[0]} 组！")

model = Sequential()

# 🌟 核心升级：输入形状变为 126 维！
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 126)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stop = EarlyStopping(monitor='val_categorical_accuracy', patience=25, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)

print("🚀 开启 3D 模型训练...")
model.fit(X_train_augmented, y_train_augmented,
          epochs=200,
          validation_data=(X_test, y_test),
          callbacks=[early_stop, reduce_lr])

model.save('action_cnn_3d.h5')  # 🌟 保存为 3D 专属模型名字
print("🎉 恭喜！高精度 3D AI 大脑 (action_cnn_3d.h5) 已保存！")