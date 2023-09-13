# ENV

```bash
conda create --name yolov5 python=3.8
pip install -r requirements.txt  
```



# Batch inference to crop onboarding data

```bash
{
python3 batch_inf.py \
    --image_root='/home/walter/big_daddy/onboard_jpg' \
    --save_root='/home/wa' \
    --weights='/home/walter/git/yolov5/new_office_onboard/new_office_onboard2/weights/best.pt' \
    --batch_size=128 \
    --gpus "cuda:0" "cuda:1" "cuda:2" "cuda:3"
}
```

