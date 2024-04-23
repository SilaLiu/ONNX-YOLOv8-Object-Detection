from ultralytics import YOLO
import onnx
print("ONNX 版本:", onnx.__version__)


model = YOLO("yolov8m.pt") 
model.export(format="onnx", imgsz=[480,640])

