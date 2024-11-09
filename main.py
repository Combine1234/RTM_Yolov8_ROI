################################################################################
#### Made by Thanawat Sukamporn ; President of Return to monkey - Tech Team ####
################################################################################

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from ultralytics import YOLO
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from datetime import datetime
import os
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, model_path, class_names):
        super().__init__()
        self.model = YOLO(model_path) 
        self.class_names = class_names
        self.roi_coords_list = [] 
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.detections_log = "data.txt"  # Log file for detections
        self.previous_detections = []

        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("YOLOv8 Detection with Multiple ROIs")
        self.setGeometry(100, 100, 1270, 940)
        self.setWindowIcon(QtGui.QIcon("icon/RTM_logo.png"))

        self.label_video = QtWidgets.QLabel(self)
        self.label_video.setGeometry(QtCore.QRect(10, 10, 1251, 691))
        self.label_video.setStyleSheet("background-color: black;")

        self.btn_add_roi = QtWidgets.QPushButton("Add ROI", self)
        self.btn_add_roi.setGeometry(QtCore.QRect(10, 710, 641, 51))
        self.btn_add_roi.clicked.connect(self.toggle_drawing_mode)

        self.btn_remove_roi = QtWidgets.QPushButton("Remove Last ROI", self)
        self.btn_remove_roi.setGeometry(QtCore.QRect(660, 710, 601, 51))
        self.btn_remove_roi.clicked.connect(self.remove_last_roi)


        self.txt_class_names = QtWidgets.QPlainTextEdit(self)
        self.txt_class_names.setGeometry(QtCore.QRect(10, 770, 1251, 51))
        self.txt_class_names.setPlaceholderText("Enter class names, one per line")

        self.btn_delete_last_class = QtWidgets.QPushButton("Delete Last Class", self)
        self.btn_delete_last_class.setGeometry(QtCore.QRect(10, 880, 1251, 51))
        self.btn_delete_last_class.clicked.connect(self.delete_last_class)

        self.btn_update_classes = QtWidgets.QPushButton("Update Classes", self)
        self.btn_update_classes.setGeometry(QtCore.QRect(10, 830, 1251, 51))
        self.btn_update_classes.clicked.connect(self.update_class_names)

        self.cap = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_class_names(self):
        with open("classnames.txt", "a") as f:
            f.write(self.txt_class_names.toPlainText() + "\n")
        
        with open("classnames.txt", "r") as f:
            self.class_names = [line.strip() for line in f]

    def delete_last_class(self):
        with open("classnames.txt", "r") as f:
            lines = f.readlines()
        if lines:
            lines.pop()
            with open("classnames.txt", "w") as f:
                f.writelines(lines)

            self.class_names = [line.strip() for line in lines]

    def toggle_drawing_mode(self):
        self.is_drawing = not self.is_drawing
        if self.is_drawing:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def remove_last_roi(self):
        if self.roi_coords_list:
            self.roi_coords_list.pop()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame = cv2.resize(frame, (1251, 691))
        
        for (roi_x, roi_y, roi_w, roi_h) in self.roi_coords_list:
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        if self.is_drawing and self.start_point and self.end_point:
            cv2.rectangle(frame, self.start_point, self.end_point, (0, 255, 255), 2)

        results = self.model.predict(frame, imgsz=640, conf=0.25)
        object_counts = [0] * len(self.roi_coords_list)
        roi_classes = [[] for _ in self.roi_coords_list]

        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]
            class_id = int(result.cls)
            label = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"

            for i, (roi_x, roi_y, roi_w, roi_h) in enumerate(self.roi_coords_list):
                if (x1 >= roi_x and y1 >= roi_y and x2 <= roi_x + roi_w and y2 <= roi_y + roi_h):
                    object_counts[i] += 1
                    roi_classes[i].append(class_id)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for i, count in enumerate(object_counts):
            cv2.putText(frame, f"ROI {i + 1}: {count} objects", (10, 30 + i * 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if any(count >= 1 for count in object_counts):
            self.log_detections(object_counts, roi_classes)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_image))
    def log_detections(self, object_counts, roi_classes):
        current_detections = [(count, classes) for count, classes in zip(object_counts, roi_classes)]
        if current_detections != self.previous_detections:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.detections_log, "a") as f:
                for i, (count, classes) in enumerate(current_detections):
                    class_list = ", ".join(map(str, classes)) if classes else "-"
                    f.write(f"ROI{i + 1} : {count} at {timestamp} classes: {class_list}\n")
            self.previous_detections = current_detections

    def mousePressEvent(self, event):
        if self.is_drawing and event.button() == Qt.LeftButton:
            self.start_point = (event.x(), event.y())

    def mouseMoveEvent(self, event):
        if self.is_drawing and self.start_point:
            self.end_point = (event.x() + 10, event.y() + 10)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.is_drawing and event.button() == Qt.LeftButton and self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            roi_x, roi_y = min(x1, x2), min(y1, y2)
            roi_w, roi_h = abs(x2 - x1), abs(y2 - y1)
            self.roi_coords_list.append((roi_x, roi_y, roi_w, roi_h))
            self.start_point = None
            self.end_point = None
            self.is_drawing = False
            self.setCursor(Qt.ArrowCursor)

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    try:
        with open("ptpath.txt", "r") as f:
            model_path = f.read().strip()
    except FileNotFoundError:
        print("Error: ptpath.txt not found.")
        sys.exit(1)

    class_names = []
    app = QtWidgets.QApplication(sys.argv)
    main_window = Ui_MainWindow(model_path, class_names)
    main_window.show()
    sys.exit(app.exec_())