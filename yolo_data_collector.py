import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
from ultralytics import YOLO

print("🔄 正在加载终极数据采集系统 (3D 全息感知版)...")

TARGET_ACTION = '9'  # 👈 在这里修改你要录制的词汇拼音/代号
NO_SEQUENCES = 40  # 录制 40 次
SEQUENCE_LENGTH = 40  # 每次 40 帧

# 🌟 将保存根目录修改回默认的 dataset 文件夹
DATA_PATH = os.path.join('dataset')

# 自动创建根目录（如果不存在的话），防止程序因为找不到文件夹而报错
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

action_path = os.path.join(DATA_PATH, TARGET_ACTION)
if not os.path.exists(action_path):
    os.makedirs(action_path)
for sequence in range(NO_SEQUENCES):
    try:
        os.makedirs(os.path.join(action_path, str(sequence)))
    except:
        pass

model = YOLO('yolov8n.pt')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_normalized_keypoints(results):
    # 🌟 升级：每只手现在有 21个点 * 3维(x,y,z) = 63个特征
    lh = np.zeros(63)
    rh = np.zeros(63)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label

            # 🌟 核心升级：同时提取 x, y, z 三个维度的信息！
            keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()

            if hand_label == 'Left':
                lh = keypoints
            else:
                rh = keypoints

    # 两只手拼起来就是 126 个特征
    return np.concatenate([lh, rh])


cap = cv2.VideoCapture(0)
print(f"\n🚀 开始录制动作：【{TARGET_ACTION}】")
print(f"📁 数据将安全保存在目录：{DATA_PATH}/{TARGET_ACTION}")

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    sequence = 0
    while sequence < NO_SEQUENCES:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        results_yolo = model(frame, classes=[0], conf=0.5, verbose=False)

        cv2.putText(frame, f"Word: {TARGET_ACTION} | Progress: {sequence}/{NO_SEQUENCES}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 's' to Record | 'r' to Redo | 'q' to Quit", (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Data Collector Pro', frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r') and sequence > 0:
            sequence -= 1
            seq_path = os.path.join(action_path, str(sequence))
            if os.path.exists(seq_path):
                shutil.rmtree(seq_path)
                os.makedirs(seq_path)
            continue

        if key == ord('s'):
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                results_yolo = model(frame, classes=[0], conf=0.5, verbose=False)

                cropped_person = None
                # 🌟 如果找不到手，就填充 126 个 0
                keypoints = np.zeros(126)

                if len(results_yolo[0].boxes) > 0:
                    largest_area = 0
                    best_box = None
                    for box in results_yolo[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            best_box = box

                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    h, w = frame.shape[:2]
                    y1, y2 = max(0, y1), min(h, y2)
                    x1, x2 = max(0, x1), min(w, x2)

                    if y2 > y1 and x2 > x1:
                        cropped_person = frame[y1:y2, x1:x2]
                        image_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
                        results_mp = hands.process(image_rgb)

                        if results_mp.multi_hand_landmarks:
                            for hand_landmarks in results_mp.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(cropped_person, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            keypoints = extract_normalized_keypoints(results_mp)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                cv2.putText(frame, f"RECORDING... {frame_num + 1}/{SEQUENCE_LENGTH}", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('Data Collector Pro', frame)

                if cropped_person is not None:
                    cv2.imshow('MediaPipe Core Vision', cropped_person)

                npy_path = os.path.join(action_path, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                cv2.waitKey(1)

            sequence += 1
            try:
                cv2.destroyWindow('MediaPipe Core Vision')
            except:
                pass

cap.release()
cv2.destroyAllWindows()
