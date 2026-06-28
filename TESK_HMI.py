import sys
import cv2
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, \
    QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont


class HMIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HMI SYSTEM - YOLOv8 BOX DETECTION")
        self.setGeometry(100, 100, 1050, 650)
        self.setStyleSheet("background-color: #2C3E50; color: white;")  # Giao diện tối kiểu công nghiệp

        # --- Khởi tạo YOLOv8 và Biến cấu hình ---
        self.model = YOLO("best.pt")
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Biến đếm tích lũy
        self.total_counts = {}
        self.prev_detected_classes = set()
        self.is_emergency = False  # Trạng thái dừng khẩn cấp

        # --- KHỞI TẠO GIAO DIỆN ---
        self.initUI()

    def initUI(self):
        # Widget trung tâm
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ================= VÙNG TRÁI: CAMERA HIỂN THỊ =================
        left_layout = QVBoxLayout()
        self.lbl_cam = QLabel("SYSTEM READY - PRESS START")
        self.lbl_cam.setFixedSize(640, 480)
        self.lbl_cam.setAlignment(Qt.AlignCenter)
        self.lbl_cam.setStyleSheet(
            "background-color: #1A252F; border: 3px solid #34495E; font-size: 16px; font-weight: bold;")
        left_layout.addWidget(self.lbl_cam)

        # Hiển thị thời gian thực phía dưới cam
        self.lbl_time = QLabel("Thời gian: --/--/---- --:--:--")
        self.lbl_time.setFont(QFont("Arial", 11))
        left_layout.addWidget(self.lbl_time)
        main_layout.addLayout(left_layout)

        # ================= VÙNG PHẢI: ĐIỀU KHIỂN & BÁO CÁO =================
        right_layout = QVBoxLayout()

        # Bảng hiển thị kết quả (HMI Display)
        self.lbl_title = QLabel("HMI CONTROL PANEL")
        self.lbl_title.setFont(QFont("Arial", 16, QFont.Bold))
        self.lbl_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_title)

        self.lbl_result = QLabel("Trạng thái: Đang dừng\n\nChưa có dữ liệu nhận dạng.")
        self.lbl_result.setFont(QFont("Arial", 12))
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setStyleSheet(
            "background-color: #11171D; border: 2px solid #1ABC9C; padding: 15px; border-radius: 5px;")
        self.lbl_result.setFixedSize(350, 250)
        right_layout.addWidget(self.lbl_result)

        # Cụm nút nhấn điều khiển (Buttons)
        btn_layout = QGridLayout()

        self.btn_start = QPushButton("BẬT (START)")
        self.btn_start.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_start.setStyleSheet("background-color: #2ECC71; color: white; padding: 12px; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_system)

        self.btn_stop = QPushButton("TẮT (STOP)")
        self.btn_stop.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_stop.setStyleSheet("background-color: #E67E22; color: white; padding: 12px; border-radius: 5px;")
        self.btn_stop.clicked.connect(self.stop_system)

        self.btn_emergency = QPushButton("DỪNG KHẨN CẤP\n(EMERGENCY)")
        self.btn_emergency.setFont(QFont("Arial", 12, QFont.Bold))
        self.btn_emergency.setStyleSheet(
            "background-color: #C0392B; color: white; padding: 15px; border-radius: 5px; border: 2px solid #FFFFFF;")
        self.btn_emergency.clicked.connect(self.emergency_system)

        btn_layout.addWidget(self.btn_start, 0, 0)
        btn_layout.addWidget(self.btn_stop, 0, 1)
        btn_layout.addWidget(self.btn_emergency, 1, 0, 1, 2)  # Nút khẩn cấp kéo dài 2 cột

        right_layout.addLayout(btn_layout)
        main_layout.addLayout(right_layout)

    # --- XỬ LÝ LOGIC HỆ THỐNG ---
    def start_system(self):
        if self.is_emergency:
            return  # Nếu đang kẹt nút khẩn cấp thì không cho bật
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)  # Cập nhật khoảng 30fps
            self.lbl_title.setText("SYSTEM STATUS: RUNNING")
            self.lbl_title.setStyleSheet("color: #2ECC71; font-weight: bold; font-size: 16px;")

    def stop_system(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.lbl_cam.clear()
        self.lbl_cam.setText("SYSTEM STOPPED")
        self.lbl_title.setText("SYSTEM STATUS: STOPPED")
        self.lbl_title.setStyleSheet("color: #E67E22; font-weight: bold; font-size: 16px;")

    def emergency_system(self):
        self.is_emergency = not self.is_emergency
        if self.is_emergency:
            # Kích hoạt trạng thái dừng khẩn cấp lập tức
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.lbl_cam.setStyleSheet(
                "background-color: #7F8C8D; border: 3px solid #C0392B; color: #C0392B; font-size: 24px;")
            self.lbl_cam.setText("⚠️ EMERGENCY STOP ACTIVATED ⚠️")
            self.lbl_title.setText("SYSTEM STATUS: EMERGENCY!!")
            self.lbl_title.setStyleSheet("color: #C0392B; font-weight: bold; font-size: 16px;")
            self.btn_emergency.setText("RESET EMERGENCY")
            self.btn_emergency.setStyleSheet("background-color: #7F8C8D; color: white; padding: 15px;")
        else:
            # Nhấn lần nữa để Reset trạng thái khẩn cấp
            self.lbl_cam.setStyleSheet(
                "background-color: #1A252F; border: 3px solid #34495E; color: white; font-size: 16px;")
            self.lbl_cam.setText("EMERGENCY RESET - SYSTEM READY")
            self.lbl_title.setText("HMI CONTROL PANEL")
            self.lbl_title.setStyleSheet("color: white; font-size: 16px;")
            self.btn_emergency.setText("DỪNG KHẨN CẤP\n(EMERGENCY)")
            self.btn_emergency.setStyleSheet("background-color: #C0392B; color: white; padding: 15px;")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 1. Lật camera chống ngược hình
        frame = cv2.flip(frame, 1)
        now = datetime.now()
        self.lbl_time.setText(f"Thời gian: {now.strftime('%d/%m/%Y %H:%M:%S')}")

        # 2. Dự đoán vật thể YOLOv8
        results = self.model(frame, conf=0.5, verbose=False)
        detected_boxes = results[0].boxes

        current_classes = []
        current_frame_counts = {}

        for box in detected_boxes:
            class_id = int(box.cls[0])
            label = self.model.names[class_id]
            current_classes.append(label)
            current_frame_counts[label] = current_frame_counts.get(label, 0) + 1

        # 3. Logic xử lý đếm tích lũy riêng biệt từng loại
        has_change = False
        for label, count in current_frame_counts.items():
            if label not in self.prev_detected_classes:
                self.total_counts[label] = self.total_counts.get(label, 0) + count
                has_change = True

        self.prev_detected_classes = set(current_classes)

        # 4. Hiển thị thông tin lên bảng HMI Display bên phải
        if len(current_frame_counts) > 0:
            result_text = f"⏱️ Thời gian ghi nhận:\n{now.strftime('%H:%M:%S - %d/%m/%Y')}\n\n"
            result_text += "=== VẬT THỂ HIỆN TẠI ===\n"
            for label, count in current_frame_counts.items():
                result_text += f"👉 {label}: {count} cái\n"
            result_text += "\n=== TỔNG TÍCH LŨY SYSTEM ===\n"
            for label, total in self.total_counts.items():
                result_text += f"📊 {label}: xuất hiện {total} lần\n"
            self.lbl_result.setText(result_text)
        elif has_change or not current_classes:
            # Nếu không có vật thể nào trong khung hình
            self.lbl_result.setText(
                f"⏱️ Trạng thái thực tế:\n{now.strftime('%H:%M:%S')}\n\n🔍 Không phát hiện vật thể nào trong vùng quét.")

        # 5. Chuyển đổi ảnh OpenCV (BGR) thành dạng QPixmap để hiển thị lên Giao diện PyQt5
        annotated_frame = results[0].plot()
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(convert_to_Qt_format)

        # Đẩy ảnh lên khung QLabel bên trái
        self.lbl_cam.setPixmap(pixmap.scaled(self.lbl_cam.width(), self.lbl_cam.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        # Giải phóng cam khi tắt ứng dụng tránh lỗi bộ nhớ
        self.stop_system()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = HMIApp()
    ex.show()
    sys.exit(app.exec_())