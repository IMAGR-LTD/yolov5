import torch
from pathlib import Path
from PIL import Image
import glob 
import os 
from models.common import DetectMultiBackend, AutoShape
import multiprocessing as mp
import argparse
from utils.general import non_max_suppression

def image_loader(directory, batch_size):
    image_files = glob.glob(f"{directory}/*/*.jpg")
    num_images = len(image_files)
    print(image_files[0])
    print(num_images)
    start_idx = 0

    while start_idx < num_images:
        batch_images = []
        end_idx = min(start_idx + batch_size, num_images)
        
        for i in range(start_idx, end_idx):
            # image_path = os.path.join(directory, image_files[i])
            img = Image.open(image_files[i])
            batch_images.append(img)
        
        yield batch_images
        start_idx = end_idx


def process(device, weights, image_root, save_root, render_root, barcodes, batch_size):
    model = DetectMultiBackend(weights, device=device)
    model = AutoShape(model)

    for barcode in barcodes:
        img_dir = os.path.join(image_root, barcode)
        save_dir = Path(os.path.join(save_root, barcode))
        os.makedirs(save_dir, exist_ok=True)
        image_generator = image_loader(img_dir, batch_size)
        print(f"processing barcode: {barcode}")
        # save_whole_dir = Path(os.path.join(render_root, barcode))
        
        # os.makedirs(save_whole_dir, exist_ok=True)
        for batch in image_generator:
            results = model(batch, size=320) 
            # results.save(save_dir=save_whole_dir, exist_ok=True)
            results.crop(save_dir=save_dir, exist_ok=True, save=False)


def split_task(barcodes, num_task):
    split_barcodes = []
    num_barcodes = len(barcodes)
    split_index = int(num_barcodes  / num_task) + 1
    for i in range(num_task):
        start_index = i * split_index
        end_index = i * split_index + split_index
        end_index = min(end_index, num_barcodes)
        split_barcodes.append(barcodes[start_index: end_index])

    return split_barcodes


# Model
def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--image_root', required=True, help='path to the images dir')
    parser.add_argument('-s', '--save_root', required=True, help='path to the save dir')
    parser.add_argument('-r', '--render_root', required=True, help='path to the save the render image')
    parser.add_argument('-w', '--weights', required=True, help='path to model weight .pt')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='img batch size ')
    parser.add_argument('-g', '--gpus',  required=True, nargs='+', type=str, help="list of available gpus cuda:0, cuda:1, ...")
    args = parser.parse_args()

    image_root = args.image_root
    save_root = args.save_root
    render_root = args.render_root
    weights = args.weights
    batch_size = args.batch_size
    gpus = args.gpus
    
    devices = [torch.device(x) for x in gpus]
    num_devices = len(devices)
    print(f"number of gpus devices: {num_devices}")

    pool = mp.Pool(num_devices)
    barcodes = os.listdir(image_root)
    barcodes_to_precess = []
    for barcode in barcodes:
        done_imgs = glob.glob(f"{save_root}/{barcode}/*/*.jpg")
        if len(done_imgs) < 1000:
            barcodes_to_precess.append(barcode)
    # done_barcodes = os.listdir(save_root)
    # barcodes_to_precess = [x for x in barcodes if x not in done_barcodes]
    print(f"processing total number of :{len(barcodes_to_precess)} barcodes")
    split_barcodes = split_task(barcodes_to_precess, num_devices)
    

    for i in range(num_devices):
        pool.apply_async(process, args=(devices[i], weights, image_root, save_root, render_root, split_barcodes[i], batch_size))

    pool.close()
    pool.join()


if __name__=="__main__":
    main()




