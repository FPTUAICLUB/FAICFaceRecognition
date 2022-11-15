# AI-checkin

## Step 1
- Tải [pretrained](https://drive.google.com/file/d/1LySevGtWg0srT400iG4DrUmJYgS3CaN9/view?usp=sharing) models bao gồm 1 face recognition model checkpoint từ [InsightFace](https://github.com/deepinsight/insightface) và 2 model checkpoints từ OpenCV
## Step 2
```
conda create -n ckin python=3.8
conda activate ckin
pip install opencv-python onnxruntime-gpu scikit-learn unidecode tqdm
```

## Step 3
```
git clone https://github.com/FPTUAICLUB/AI-Checkin.git
```
- Unzip folder tải từ **Step 1** và đặt ở trong folder AI-Checkin vừa clone


## Step 4
```
export PYTHONPATH=AI-Checkin
cd AI-Checkin
python main.py [-e ${PAD}]  
```