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

w = "/home/walter/git/yolov5/blackout_edgetpou/blackout/weights/best-int8_edgetpu.tflite"
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

img = "/home/walter/images/cam0/ED_170823_OUT_4_1692241145664_7926606_0_00_1092.jpg"


im0 = Image.open(img)
im = im0.resize((320,320))
im = np.array(im).astype(float)
im /= 255 
im = im[None]

b, w, h, ch= im.shape
im = (im / scale + zero_point).astype(np.uint8)

interpreter.set_tensor(input['index'], im)
interpreter.invoke()

y = []

for output in output_details:
    x = interpreter.get_tensor(output['index'])
    scale, zero_point = output['quantization']
    x = (x.astype(np.float32) - zero_point) * scale
    y.append(x)

y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]

y[0][..., :4] *= [w, h, w, h]

pred = np.array(y[0])
print(pred)
pred = torch.from_numpy(pred)

results = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)
print(pred.shape)
print(results)

for result in results:
    result_np = result.numpy()
    print(result_np[0])
    bbox = result_np[0][0:4]
    print(bbox)

    draw = ImageDraw.Draw(im0)
    draw.rectangle(bbox)
    im0.show()