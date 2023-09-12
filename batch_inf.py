import torch
from pathlib import Path
from PIL import Image
import glob 
import os 




# Model
device = torch.device('cuda:1')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./blackout/blackout/weights/blackout.pt').to(device)

image_root = "/home/walter/big_daddy/onboard_jpg"
save_root = "/home/walter/crops"
barcodes = os.listdir(image_root)

batch_size = 128

def image_loader(directory, batch_size):
    image_files = glob.glob(f"{directory}/*/*.jpg")
    num_images = len(image_files)
    start_idx = 0

    while start_idx < num_images:
        batch_images = []
        end_idx = min(start_idx + batch_size, num_images)
        
        for i in range(start_idx, end_idx):
            image_path = os.path.join(directory, image_files[i])
            img = Image.open(image_path)
            batch_images.append(img)
        
        yield batch_images
        start_idx = end_idx



for barcode in barcodes:
    image_dir = os.path.join(image_root, barcode)
    save_dir = os.path.join(save_root, barcode)
    os.makedirs(save_dir, exist_ok=True)
    image_generator = image_loader(image_dir, batch_size)
    for batch in image_generator:
        results = model(batch, size=640) 
        results.crop(save_dir=save_dir, exist_ok=True, save=False)
