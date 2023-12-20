from tflite_runtime.interpreter import Interpreter, load_delegate
import platform
import torch
from PIL import Image, ImageDraw
import numpy as np 
import re
import glob 
import os 
import time 
import matplotlib.pyplot as plt

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(boxes, conf=0.5, iou_threshold=0.5):
    # Sort boxes by object confidence scores in descending order
    
    boxes.sort(key=lambda x: x[5], reverse=True)
    
    selected_boxes = []
    
    while len(boxes) > 0:
        # Select box with highest object confidence
        best_box = boxes[0]
        selected_boxes.append(best_box)
        del boxes[0]
        
        # Calculate IoU with remaining boxes
        iou_scores = [calculate_iou(best_box, box) for box in boxes]
        
        # Remove boxes with high IoU
        boxes = [box for i, box in enumerate(boxes) if iou_scores[i] < iou_threshold]
    
    return selected_boxes

def calculate_iou(box1, box2):
    # Convert box format to [xmin, ymin, xmax, ymax]
    box1 = convert_to_corners(box1)
    box2 = convert_to_corners(box2)
    
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # Calculate areas of the bounding boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Calculate IoU
    iou = intersection_area / float(area_box1 + area_box2 - intersection_area)
    
    return iou


def iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # Calculate areas of the bounding boxes
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    # Calculate IoU
    iou = intersection_area / float(area_box1 + area_box2 - intersection_area)
    
    return iou

def convert_to_corners(box):
    center_x, center_y, width, height = box[0], box[1], box[2], box[3]
    xmin = center_x - (width / 2)
    ymin = center_y - (height / 2)
    xmax = center_x + (width / 2)
    ymax = center_y + (height / 2)
    
    return [xmin, ymin, xmax, ymax, box[4]]


def load_interpreter(weight, delegate):
    print(f'Loading {weight} for TensorFlow Lite Edge TPU inference...')
    interpreter = Interpreter(model_path=weight, experimental_delegates=[load_delegate(delegate)])
    interpreter.allocate_tensors()
    return interpreter


def get_interpreter_output(interpreter, img: Image, conf_threshold = 0.5, iou_threshold = 0.5):
    
    w, h= img.size
    input_details = interpreter.get_input_details()  # inputs
    input = input_details[0]
    scale, zero_point = input['quantization']
    shape = input["shape"]
    b, width, height, channel = shape
    img = img.resize((width, height))
    img = np.array(img).astype(float)
    img /= 255 
    img = img[None]
    img = (img / scale + zero_point).astype(np.uint8)
    interpreter.set_tensor(input['index'], img)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output = output_details[0]
    y = interpreter.get_tensor(output['index'])
    scale, zero_point = output['quantization']
    y = (y.astype(np.float32) - zero_point) * scale

    # nms 
    y[..., 5:] *= y[..., 4:5]
    conf = np.amax(y[..., 5:], axis=2, keepdims=True)
    bboxs = y[..., :4]
    pred = np.concatenate([bboxs, conf], axis=2)
    mask = pred[..., 4] >= conf_threshold
    bboxs = pred[mask]
    bboxs = np.array(sorted(bboxs, key=lambda x: x[4], reverse=True))
    selected_boxes = []
    while len(bboxs) > 0:
        # Select box with highest object confidence
        best_box = bboxs[0]
        corner_bbox = convert_to_corners(best_box)
        selected_boxes.append(corner_bbox)
        bboxs = np.delete(bboxs, 0, axis=0)
        # Calculate IoU with remaining boxes
        iou_scores = [calculate_iou(best_box, box) for box in bboxs]
        # Remove boxes with high IoU
        bboxs = [box for i, box in enumerate(bboxs) if iou_scores[i] < iou_threshold]

    selected_boxes = np.array(selected_boxes)
    bboxes = []
    if len(selected_boxes) > 0:
        bboxes = selected_boxes[:,:4] * np.array([w, h, w, h])
    results = []
    for bbox in bboxes:
        bbox = [int(x) for x in bbox]
        results.append(bbox)
    
    # selected_boxes[:,:4] = selected_boxes[:,:4] * np.array([w, h, w, h])
    return results


def xywh2xyxy(bbox):
    "bbox is a list contains x,y,w,h"
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    xmin = x - w/2
    ymin = y - h/2
    xmax = x + w/2
    ymax = y + h/2
    return [xmin, ymin, xmax, ymax]


def read_yolo_anno(anno_path):
    """yolo anno [label, x, y, w, h] in normalized form 

    Args:
        anno_path (path): path to the yolo anno file

    Returns:
        dict: return a dict of yolo anno 
    """
    cls_ids = []
    bboxes = []
    with open(anno_path, "r") as f:
        for line in f.readlines():
            data = line.strip().split()
            data = [float(x) for x in data]
            cls_ids.append(int(data[0])) 
            x, y, w, h = data[1], data[2], data[3], data[4]
            bbox = xywh2xyxy([x, y, w, h])
            bboxes.append(bbox)
            
    return cls_ids, bboxes


def scale_bbox_by_img(bboxes, width, height):
    s_bboxes = []
    for bbox in bboxes:
        s_bbox = [int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)]
        s_bboxes.append(s_bbox)
    return s_bboxes



def inf_one_img(img_path,  conf_threshold=0.5, iou_threshold=0.5):
    img = Image.open(img_path)
    label_path = re.sub("/images", "/labels", img_path)
    label_path = re.sub(".jpg", ".txt", label_path)
    cls_ids, bboxes = read_yolo_anno(label_path)
    label_bboxes = scale_bbox_by_img(bboxes, 324, 324)
    
    pred_bboxes = get_interpreter_output(interpreter, img, conf_threshold, iou_threshold)
    
    return pred_bboxes, label_bboxes


class Metrics:
    def __init__(self, iou_threshold):
        self.total = 0
        self.total_pred = 0
        self.correct = 0
        self.noise = 0
        self.mis_product = 0
        self.not_tight = 0
        self.iou_threshold = iou_threshold
    
    def update(self, pred_bboxes, label_bboxes):
        num_labels = len(label_bboxes)
        num_preds = len(pred_bboxes)
        self.total += num_labels
        self.total_pred += num_preds

        if num_labels == 0 and num_preds == 0:
            return
        
        if num_labels == 0 and num_preds > 0:
            self.noise += num_preds
        elif num_labels > 0 and num_preds == 0:
            self.mis_product += num_labels
        else:
            iou_score = iou(label_bboxes[0], pred_bboxes[0])
            if iou_score > iou_threshold_for_inf:
                self.correct += 1
            elif iou_score > 0:
                self.not_tight += 1
            else:
                self.mis_product += 1
                self.noise += 1

    def output(self):
        print(f"total_label: {self.total}")
        print(f"total_pred: {self.total_pred}")
        print(f"{iou_threshold_for_inf} precise: {self.correct/self.total_pred*100:.2f}%")
        print(f"noise out from total predict {self.noise/self.total_pred*100:.2f}%")
        print(f"bbox_no_tight out of all pred: {self.not_tight/self.total_pred*100:.2f}%")
        print(f"missing {self.mis_product/self.total*100:.2f}%")

delegate = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'}[platform.system()]

iou_threshold_for_inf = 0.5

w = "/home/walter/nas_cv/walter_stuff/saved_models/blackout_nano/19_oct4/weights/best-int8_edgetpu.tflite"
interpreter = load_interpreter(w, delegate)
imgs = glob.glob(f"/home/walter/nas_cv/walter_stuff/yolov5_dataset/images/blackout/test/*.jpg")
result_save_dir = f"/home/walter/inf_result/blackout_{iou_threshold_for_inf}"

num_imgs = len(imgs)
total_label = 0
total_pred = 0
correct_pred = 0
noise = 0
mis_product = 0
bbox_no_tight = 0
total_small = 0
find_small = 0
total_medium = 0
find_medium = 0
total_large = 0
find_large = 0

overall = Metrics(0.5)
small_m = Metrics(0.5)
medium_m = Metrics(0.5)
large_m = Metrics(0.5)


data = []
for img in imgs:
    img_basename = os.path.basename(img)
    is_noise = False
    is_mis_product = False
    is_bbox_no_tight = False
    is_correct = False
    is_small = False
    is_medium = False
    is_large = False
    
    pred_bboxes, label_bboxes = inf_one_img(img, conf_threshold=0.5)
    
    img = Image.open(img)
    for pred_bbox in pred_bboxes:
        draw = ImageDraw.Draw(img)
        draw.rectangle(pred_bbox, outline="green", width=3)

    for label_bbox in label_bboxes:
        width = label_bbox[2] - label_bbox[0]
        height = label_bbox[3] - label_bbox[1]
        data.append(width * height)
        if width * height <= 2048:
            is_small = True
        elif width * height >= 16384:
            is_large = True
        else:
            is_medium = True
        draw = ImageDraw.Draw(img)
        draw.rectangle(label_bbox)

#     overall.update(pred_bboxes, label_bboxes)
#     if is_large:
#         large_m.update(pred_bboxes, label_bboxes)
#     elif is_medium:
#         medium_m.update(pred_bboxes, label_bboxes)
#     elif is_small:
#         small_m.update(pred_bboxes, label_bboxes)

# print("overall result --------------------------------------------------------")
# overall.output()
# print("small_m result --------------------------------------------------------")
# small_m.output()
# print("medium_m result --------------------------------------------------------")
# medium_m.output()
# print("large_m result --------------------------------------------------------")
# large_m.output()
    num_labels = len(label_bboxes)
    total_label += num_labels
    num_preds = len(pred_bboxes)
    total_pred += num_preds

    if num_labels == 0 and num_preds == 0:
        continue
    elif num_labels == 0 and num_preds > 0:
        noise += num_preds
        is_noise = True
    elif num_labels > 0 and num_preds == 0:
        mis_product += num_labels
        is_mis_product = True
        
    else:
        iou_score = iou(label_bboxes[0], pred_bboxes[0])
        if iou_score > iou_threshold_for_inf:
            correct_pred += 1
            is_correct = True
        elif iou_score > 0:
            bbox_no_tight += 1
            is_bbox_no_tight = True
        else:
            mis_product += 1
            noise += 1
            is_mis_product = True
            is_noise = True
    
    if is_noise:
        savedir = os.path.join(result_save_dir, "noise")
        os.makedirs(savedir, exist_ok=True)
        dst = os.path.join(savedir, img_basename)
        img.save(dst)
    
    if is_mis_product:
        savedir = os.path.join(result_save_dir, "mis")
        os.makedirs(savedir, exist_ok=True)
        dst = os.path.join(savedir, img_basename)
        img.save(dst)

    if is_bbox_no_tight:
        savedir = os.path.join(result_save_dir, "not_tight")
        os.makedirs(savedir, exist_ok=True)
        dst = os.path.join(savedir, img_basename)
        img.save(dst)

    if is_correct:
        savedir = os.path.join(result_save_dir, "correct")
        os.makedirs(savedir, exist_ok=True)
        dst = os.path.join(savedir, img_basename)
        img.save(dst)
    
    if is_small:
        savedir = os.path.join(result_save_dir, "small")
        os.makedirs(savedir, exist_ok=True)
        dst = os.path.join(savedir, img_basename)
        img.save(dst)

    if is_medium:
        savedir = os.path.join(result_save_dir, "medium")
        os.makedirs(savedir, exist_ok=True)
        dst = os.path.join(savedir, img_basename)
        img.save(dst)

    if is_large:
        savedir = os.path.join(result_save_dir, "large")
        os.makedirs(savedir, exist_ok=True)
        dst = os.path.join(savedir, img_basename)
        img.save(dst)


print(f"total num of imgs: {num_imgs}")
print(f"total_label: {total_label}")
print(f"total_pred: {total_pred}")
print(f"correct_pred: {correct_pred}")
print(f"noise: {noise}")
print(f"mis_product: {mis_product}")
print(f"bbox_no_tight: {bbox_no_tight}")
print(f"{iou_threshold_for_inf} precise: {correct_pred/total_pred*100:.2f}%")
print(f"noise out from total predict {noise/total_pred*100:.2f}%")
print(f"missing {mis_product/total_label*100:.2f}%")
print(f"bbox_no_tight out of all pred: {bbox_no_tight/total_pred*100:.2f}%")