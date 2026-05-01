import sys
import cv2
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, \
    QTextBrowser
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from tensorflow.keras.models import load_model
from ultralytics import YOLO

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

# ================= 1. 🌟 你的专属字典 =================
SIGN_TRANSLATOR = {
    'beautiful': '漂亮', 'come': '来', 'drink': '喝', 'eat': '吃', 'good': '好',
    'have': '有', 'help': '帮助', 'here': '这里', 'home': '家', 'is': '是',
    'like': '喜欢', 'look': '看', 'me': '我', 'morning': '早上', 'no': '不',
    'sorry': '抱歉', 'think': '想', 'very': '很', 'what': '什么', 'you': '你',
    'idle': ''
}


# ================= 2. AI 视觉处理后台线程 =================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_text_signal = pyqtSignal(str)
    action_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def extract_normalized_keypoints(self, hand_landmarks):
        base_x = hand_landmarks.landmark[0].x
        base_y = hand_landmarks.landmark[0].y
        return np.array([[res.x - base_x, res.y - base_y] for res in hand_landmarks.landmark]).flatten()

    def run(self):
        self.update_text_signal.emit("正在加载纯净版交流模型...")

        yolo_device = 0 if torch.cuda.is_available() else 'cpu'
        model_yolo = YOLO('yolov8n.pt')
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        model_lstm = load_model('action.h5')

        DATA_PATH = os.path.join('dataset')
        actions = np.array([name for name in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, name))])

        sequence = []
        threshold = 0.80

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.update_text_signal.emit("✅ 系统已就绪，请在镜头前做动作！")

        frame_counter = 0
        cached_box = None

        with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while self._run_flag and cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                frame_counter += 1
                frame = cv2.flip(frame, 1)

                hand_detected = False

                if frame_counter % 5 == 0 or cached_box is None:
                    results_yolo = model_yolo(frame, classes=[0], conf=0.5, device=yolo_device, verbose=False)
                    if len(results_yolo[0].boxes) > 0:
                        largest_area = 0
                        best_box = None
                        for box in results_yolo[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            area = (x2 - x1) * (y2 - y1)
                            if area > largest_area:
                                largest_area = area
                                best_box = box
                        if best_box is not None:
                            cached_box = tuple(map(int, best_box.xyxy[0]))

                # 🌟 恢复理性：找不到手就老老实实用 0.0，不给模型增加负担
                lh = np.zeros(42)
                rh = np.zeros(42)

                if cached_box is not None:
                    x1, y1, x2, y2 = cached_box
                    h, w = frame.shape[:2]
                    y1, y2 = max(0, y1), min(h, y2)
                    x1, x2 = max(0, x1), min(w, x2)

                    if y2 > y1 and x2 > x1:
                        cropped = frame[y1:y2, x1:x2]
                        results_mp = hands.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

                        if results_mp.multi_hand_landmarks:
                            hand_detected = True
                            for idx, hand_lms in enumerate(results_mp.multi_hand_landmarks):
                                mp_drawing.draw_landmarks(cropped, hand_lms, mp_hands.HAND_CONNECTIONS)
                                hand_label = results_mp.multi_handedness[idx].classification[0].label
                                kp = self.extract_normalized_keypoints(hand_lms)

                                if hand_label == 'Left':
                                    lh = kp
                                else:
                                    rh = kp

                        box_color = (0, 255, 0) if hand_detected else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                keypoints = np.concatenate([lh, rh])

                if hand_detected:
                    sequence.append(keypoints)
                    sequence = sequence[-60:]

                    if len(sequence) < 60:
                        self.update_text_signal.emit(f"🔄 正在收集动作... ({len(sequence)}/60)")
                else:
                    if len(sequence) > 0:
                        self.update_text_signal.emit("⏸️ 等待动作...")
                    sequence = []

                if len(sequence) == 60:
                    input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                    res = model_lstm(input_data, training=False).numpy()[0]
                    best_action_idx = np.argmax(res)
                    confidence = res[best_action_idx] * 100

                    if res[best_action_idx] > threshold:
                        current_action = actions[best_action_idx]
                        if current_action == 'idle':
                            self.update_text_signal.emit(f"⏸️ 检测到无动作 (Idle: {confidence:.1f}%)")
                            sequence = sequence[20:]
                        else:
                            zh_word = SIGN_TRANSLATOR.get(current_action, current_action)
                            self.update_text_signal.emit(f"✅ 成功识别: 【{zh_word}】 (把握: {confidence:.1f}%)")
                            self.action_signal.emit(current_action)
                            sequence = []
                    else:
                        self.update_text_signal.emit(
                            f"❌ 动作不清晰 (猜想: {actions[best_action_idx]} {confidence:.1f}%)")
                        sequence = sequence[10:]

                if frame_counter % 2 == 0:
                    self.change_pixmap_signal.emit(frame)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# ================= 3. 软件主界面 =================
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手语识别交互系统 - [最终守护版]")
        self.resize(900, 750)

        self.sentence_list = []
        self.last_word = ""

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1c1c1c; border-radius: 10px;")
        self.image_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.image_label)

        self.result_label = QLabel("点击下方按钮开启摄像头", self)
        self.result_label.setFont(QFont("微软雅黑", 16, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("color: #007BFF; margin-top: 10px;")
        main_layout.addWidget(self.result_label)

        self.sentence_box = QTextBrowser(self)
        self.sentence_box.setFont(QFont("微软雅黑", 24, QFont.Bold))
        self.sentence_box.setStyleSheet(
            "background-color: #F8F9FA; border: 2px solid #CED4DA; border-radius: 8px; padding: 10px; color: #343A40;")
        self.sentence_box.setMinimumHeight(100)
        self.sentence_box.setMaximumHeight(120)
        main_layout.addWidget(self.sentence_box)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("开启摄像头 / 开始识别")
        self.btn_start.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_video)
        btn_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("关闭摄像头")
        self.btn_stop.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet("background-color: #F44336; color: white; border-radius: 5px;")
        self.btn_stop.clicked.connect(self.stop_video)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        self.btn_clear = QPushButton("清空整句")
        self.btn_clear.setFont(QFont("微软雅黑", 14, QFont.Bold))
        self.btn_clear.setMinimumHeight(50)
        self.btn_clear.setStyleSheet("background-color: #FFC107; color: #333; border-radius: 5px;")
        self.btn_clear.clicked.connect(self.clear_sentence)
        btn_layout.addWidget(self.btn_clear)

        main_layout.addLayout(btn_layout)
        self.thread = None

    def start_video(self):
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("background-color: #A5D6A7; color: white; border-radius: 5px;")
        self.btn_stop.setEnabled(True)
        self.btn_stop.setStyleSheet("background-color: #F44336; color: white; border-radius: 5px;")

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_text_signal.connect(self.update_text)
        self.thread.action_signal.connect(self.update_sentence)
        self.thread.start()

    def stop_video(self):
        self.btn_start.setEnabled(True)
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #EF9A9A; color: white; border-radius: 5px;")
        if self.thread:
            self.thread.stop()
            self.image_label.clear()
            self.image_label.setText("摄像头已关闭")
            self.result_label.setText("系统已休眠")

    def update_image(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def update_text(self, text):
        if "✅" in text:
            self.result_label.setStyleSheet("color: #28A745; margin-top: 10px;")
        elif "❌" in text:
            self.result_label.setStyleSheet("color: #DC3545; margin-top: 10px;")
        elif "⏸️" in text:
            self.result_label.setStyleSheet("color: #6C757D; margin-top: 10px;")
        else:
            self.result_label.setStyleSheet("color: #007BFF; margin-top: 10px;")

        self.result_label.setText(text)

    def update_sentence(self, action_en):
        zh_word = SIGN_TRANSLATOR.get(action_en, "")
        if not zh_word: return

        if zh_word != self.last_word:
            self.sentence_list.append(zh_word)
            self.last_word = zh_word
            self.sentence_box.setText(" ".join(self.sentence_list))
            self.sentence_box.moveCursor(self.sentence_box.textCursor().End)

    def clear_sentence(self):
        self.sentence_list.clear()
        self.last_word = ""
        self.sentence_box.setText("")

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())