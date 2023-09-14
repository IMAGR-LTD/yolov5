from tflite_runtime.interpreter import Interpreter, load_delegate
import platform
import contextlib
import zipfile
import ast


w = "/home/walter/git/yolov5/blackout_edgetpou/blackout/weights/best-int8_edgetpu.tflite"
print(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
delegate = {
    'Linux': 'libedgetpu.so.1',
    'Darwin': 'libedgetpu.1.dylib',
    'Windows': 'edgetpu.dll'}[platform.system()]

print(delegate)
interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()  # inputs
output_details = interpreter.get_output_details()  # outputs


with contextlib.suppress(zipfile.BadZipFile):
    with zipfile.ZipFile(w, "r") as model:
        meta_file = model.namelist()[0]
        meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
        stride, names = int(meta['stride']), meta['names']