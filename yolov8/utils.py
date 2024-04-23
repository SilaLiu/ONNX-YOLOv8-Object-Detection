import numpy as np
import cv2

class_names = [
    'person',            # 人
    'bicycle',           # 自行车
    'car',               # 汽车
    'motorcycle',        # 摩托车
    'airplane',          # 飞机
    'bus',               # 公共汽车
    'train',             # 火车
    'truck',             # 卡车
    'boat',              # 船
    'traffic light',     # 交通灯
    'fire hydrant',      # 消防栓
    'stop sign',         # 停止标志
    'parking meter',     # 停车计时器
    'bench',             # 长凳
    'bird',              # 鸟
    'cat',               # 猫
    'dog',               # 狗
    'horse',             # 马
    'sheep',             # 绵羊
    'cow',               # 牛
    'elephant',          # 大象
    'bear',              # 熊
    'zebra',             # 斑马
    'giraffe',           # 长颈鹿
    'backpack',          # 背包
    'umbrella',          # 雨伞
    'handbag',           # 手提包
    'tie',               # 领带
    'suitcase',          # 手提箱
    'frisbee',           # 飞盘
    'skis',              # 滑雪板
    'snowboard',         # 雪板
    'sports ball',       # 运动球
    'kite',              # 风筝
    'baseball bat',      # 棒球棒
    'baseball glove',    # 棒球手套
    'skateboard',        # 滑板
    'surfboard',         # 冲浪板
    'tennis racket',     # 网球拍
    'bottle',            # 瓶子
    'wine glass',        # 酒杯
    'cup',               # 杯子
    'fork',              # 叉子
    'knife',             # 刀
    'spoon',             # 勺子
    'bowl',              # 碗
    'banana',            # 香蕉
    'apple',             # 苹果
    'sandwich',          # 三明治
    'orange',            # 橙子
    'broccoli',          # 西兰花
    'carrot',            # 胡萝卜
    'hot dog',           # 热狗
    'pizza',             # 比萨
    'donut',             # 甜甜圈
    'cake',              # 蛋糕
    'chair',             # 椅子
    'couch',             # 沙发
    'potted plant',      # 盆栽植物
    'bed',               # 床
    'dining table',      # 餐桌
    'toilet',            # 马桶
    'tv',                # 电视
    'laptop',            # 笔记本电脑
    'mouse',             # 鼠标
    'remote',            # 遥控器
    'keyboard',          # 键盘
    'cell phone',        # 手机
    'microwave',         # 微波炉
    'oven',              # 烤箱
    'toaster',           # 烤面包机
    'sink',              # 水槽
    'refrigerator',      # 冰箱
    'book',              # 书
    'clock',             # 时钟
    'vase',              # 花瓶
    'scissors',          # 剪刀
    'teddy bear',        # 泰迪
    'hair drier',        # 吹风机
    'toothbrush'         # 牙刷
]

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

label = ""

def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box( image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
