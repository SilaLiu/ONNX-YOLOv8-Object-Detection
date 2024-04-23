import cv2
import rclpy
from rclpy.node import Node
from pixmoving_hmi_msgs.msg import TargetedDetection  # 导入自定义消息类型
from yolov8 import YOLOv8
from yolov8.utils import class_names
from datetime import datetime  # 导入 datetime 模块

# 初始化网络摄像头
cap = cv2.VideoCapture(0)

class ObjectDetectionPublisher(Node):
    def __init__(self):
        super().__init__('object_detection_publisher')
        self.publisher_ = self.create_publisher(TargetedDetection, 'object_detection', 10)
        
        # 初始化YOLOv8目标检测器
        model_path = "models/yolov8m.onnx"
        self.yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

        self.timer = self.create_timer(0.1, self.detect_and_publish)

    def detect_and_publish(self):
        # 从视频中读取帧
        ret, frame = cap.read()

        if ret:
            # 更新对象定位器
            boxes, scores, class_ids = self.yolov8_detector(frame)
            
            # 初始化目标对象数量和类别列表
            target_objects = []

            # 检测画面中的目标对象数量和类别
            for class_id in class_ids:
                if class_id == 0:
                    target_objects.append(class_id)
                    detected_label = class_names[class_id]                      # 获取与 class_id 对应的标签
            # 发布目标对象数量和类别
            msg = TargetedDetection()
            msg.timestamp = self.get_current_time()                             # 添加系统当前时间
            msg.object_id = int(target_objects[0]) if target_objects else -1    # 发布检测到的目标对象类别列表ID
            msg.object_name = detected_label if target_objects else "None"
            msg.object_count = len(target_objects)                              # 发布检测到的目标对象数量
            self.publisher_.publish(msg)

            # 将检测结果绘制在帧上
            combined_img = self.yolov8_detector.draw_detections(frame)
            cv2.imshow("Detected Objects", combined_img)

            # 按下键盘上的“q”键停止
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.get_logger().info('Exiting...')
                rclpy.shutdown()

    def get_current_time(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def main():
    rclpy.init()
    object_detection_publisher = ObjectDetectionPublisher()
    rclpy.spin(object_detection_publisher)

if __name__ == '__main__':
    main()
