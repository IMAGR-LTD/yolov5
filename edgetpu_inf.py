from tflite_runtime.interpreter import Interpreter, load_delegate
import platform
import contextlib
import zipfile
import ast
import torch
import cv2 
from PIL import Image, ImageDraw
import numpy as np 
from utils.general import non_max_suppression
import glob 


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

def convert_to_corners(box):
    center_x, center_y, width, height = box[0], box[1], box[2], box[3]
    xmin = center_x - (width / 2)
    ymin = center_y - (height / 2)
    xmax = center_x + (width / 2)
    ymax = center_y + (height / 2)
    
    return [xmin, ymin, xmax, ymax, box[4]]

w = "/home/walter/nas_cv/walter_stuff/saved_models/blackout_edgetpu/blackout_edgetpu_nano_all_data2/weights/best-int8_edgetpu.tflite"
print(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
delegate = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'}[platform.system()]


interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()  # inputs
output_details = interpreter.get_output_details()  # outputs


with contextlib.suppress(zipfile.BadZipFile):
    with zipfile.ZipFile(w, "r") as model:
        meta_file = model.namelist()[0]
        meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
        stride, names = int(meta['stride']), meta['names']




input = input_details[0]
scale, zero_point = input['quantization']

img_dir = "/home/walter/git/pipeline/models/data_imagr/images/testset"
imgs = glob.glob(f"{img_dir}/*.jpg")
img = "/home/walter/images/cam0/ED_170823_IN_3_1692240508340_7289447_0_00_4465.jpg"
img = imgs[2]

im0 = Image.open(img)
im = im0.resize((320,320))
im = np.array(im).astype(float)
im /= 255 
im = im[None]

b, w, h, ch= im.shape
im = (im / scale + zero_point).astype(np.uint8)

interpreter.set_tensor(input['index'], im)
interpreter.invoke()

output = output_details[0]

y = interpreter.get_tensor(output['index'])
scale, zero_point = output['quantization']
# (1, 6300, 6) [center_x, center_y, width, height, obj_conf, cls_1_conf, cls_2_conf, ...]
y = (y.astype(np.float32) - zero_point) * scale

y[..., 5:] *= y[..., 4:5]
conf = np.amax(y[..., 5:], axis=2, keepdims=True)
bboxs = y[..., :4]
# bboxs = xywh2xyxy(bboxs)
pred = np.concatenate([bboxs, conf], axis=2)

mask = pred[..., 4] >= 0.5

bboxs = pred[mask]

bboxs = np.array(sorted(bboxs, key=lambda x: x[4], reverse=True))


selected_boxes = []
iou_threshold = 0.5

    
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
selected_boxes[:,:4] = selected_boxes[:,:4] * np.array([w, h, w, h])


for result in selected_boxes:
    
    bbox = result[:4]
    # print(bbox)
    # print(type(bbox))
    bbox = list(bbox)

    draw = ImageDraw.Draw(im0)
    draw.rectangle(bbox)
    im0.show()