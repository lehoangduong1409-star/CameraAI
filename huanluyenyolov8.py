import cv2
from ultralytics import YOLO
from datetime import datetime

# 1. Khởi tạo model
model_path = "best.pt"
model = YOLO(model_path)

# 2. Cấu hình camera
source = 0
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("LỖI: Không thể mở được Webcam.")
    exit()

# --- BIẾN ĐẾM VÀ QUẢN LÝ TRẠNG THÁI RIÊNG BIỆT ---
total_counts = {}
prev_detected_classes = set()
# ------------------------------------------------

print("Mô hình đang chạy... Hãy đưa đồ vật trước camera. Nhấn 'q' để THOÁT.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("LỖI: Không thể nhận dữ liệu từ webcam.")
        break

    # ========================================================
    # 👉 CÂU LỆNH LẬT CAMERA: Lật ngang frame (tránh bị ngược)
    frame = cv2.flip(frame, 1)
    # ========================================================

    # 3. Dự đoán vật thể với YOLOv8 trên frame đã lật
    results = model(frame, conf=0.5, verbose=False)
    detected_boxes = results[0].boxes

    # Lấy danh sách tất cả các nhãn trong frame này
    current_classes = []
    current_frame_counts = {}

    for box in detected_boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        current_classes.append(label)
        current_frame_counts[label] = current_frame_counts.get(label, 0) + 1

    # Kiểm tra xem có bất kỳ đồ vật mới nào xuất hiện hay không
    has_new_appearance = False
    for label in current_frame_counts.keys():
        if label not in prev_detected_classes:
            total_counts[label] = total_counts.get(label, 0) + current_frame_counts[label]
            has_new_appearance = True

    prev_detected_classes = set(current_classes)

    # 4. IN KẾT QUẢ ĐỒNG THỜI KÈM NGÀY THÁNG NĂM
    if has_new_appearance and len(current_frame_counts) > 0:
        now = datetime.now()
        current_time = now.strftime("%d/%m/%Y %H:%M:%S")

        print(f"\n⏱️  [THỜI GIAN GHI NHẬN: {current_time}]")
        print("================ THỐNG KÊ CHI TIẾT ================")

        for label, count in current_frame_counts.items():
            print(f" 👉 {label} hiện tại: {count} | Tổng số lần xuất hiện: {total_counts.get(label, 0)}")

        print("--------------------------------------------------")

    # 5. Vẽ khung nhận dạng lên hình ảnh
    annotated_frame = results[0].plot()

    # Hiển thị thời gian thực lên góc màn hình camera
    now = datetime.now()
    cv2.putText(annotated_frame, now.strftime("%d/%m/%Y %H:%M:%S"), (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 6. Hiển thị cửa sổ camera
    cv2.imshow("Nhan dang hop - YOLOv8", annotated_frame)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nĐã đóng chương trình.")