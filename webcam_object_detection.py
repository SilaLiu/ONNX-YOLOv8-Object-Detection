# import cv2

# from yolov8 import YOLOv8

# # 初始化网络摄像头
# cap = cv2.VideoCapture(0)

# # 初始化YOLOv8目标检测器
# model_path = "models/yolov8m.onnx"
# yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

# while cap.isOpened():

#     # 从视频中读取帧
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # 更新对象定位器
#     boxes, scores, class_ids = yolov8_detector(frame)

#     # 初始化人数计数器
#     person_count = 0
    
#     # 检查是否检测到“person”类
#     # if any(class_id == 0 for class_id in class_ids):
#     #     print("您好！ 欢迎光临～")
#     # else:
#     #     print("没有检测到人物")
    
#     # 检测画面中的人物数量
#     for class_id in class_ids:
#         if class_id == 0:
#             person_count += 1
    
#     # 输出当前画面中的人数
#     print(f"当前画面中的人数: {person_count}")

#     # 将检测结果绘制在帧上
#     combined_img = yolov8_detector.draw_detections(frame)
#     cv2.imshow("Detected Objects", combined_img)

#     # 按下键盘上的“q”键停止
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# # 释放网络摄像头并关闭所有窗口
# cap.release()
# cv2.destroyAllWindows()


import cv2
import rclpy
from rclpy.node import Node
from pixmoving_hmi_msgs.msg import TargetedDetection  # 导入自定义消息类型
from yolov8 import YOLOv8
from yolov8.utils import label,class_names

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
                    detected_label = class_names[class_id]  # 获取与 class_id 对应的标签
            # 发布目标对象数量和类别
            msg = TargetedDetection()
            msg.object_id = int (target_objects[0])      # 发布检测到的目标对象类别列表ID
            msg.object_name = detected_label
            msg.object_count = len(target_objects)       # 发布检测到的目标对象数量
            self.publisher_.publish(msg)

            # 将检测结果绘制在帧上
            combined_img = self.yolov8_detector.draw_detections(frame)
            cv2.imshow("Detected Objects", combined_img)

            # 按下键盘上的“q”键停止
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                self.get_logger().info('Exiting...')
                rclpy.shutdown()

def main():
    rclpy.init()
    object_detection_publisher = ObjectDetectionPublisher()
    rclpy.spin(object_detection_publisher)

if __name__ == '__main__':
    main()
