# About AI-checkin
Project thuộc về quyền sở hữu của FPTU AI Club - FAIC Đại học FPT Hà Nội, được phát triển nhằm hỗ trợ công việc Checkin cửa vào cho sự kiện Retro9 \
AI-checkin sử dụng pretrained InsightFace cùng với MTCNN là công nghệ chính
# AI-checkin

## Step 1
- Tải [DroidCam](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam&hl=vi&gl=US) trên điện thoại Android
- Tải [pretrained](https://drive.google.com/file/d/1LySevGtWg0srT400iG4DrUmJYgS3CaN9/view?usp=sharing) models bao gồm 1 face recognition model checkpoint từ [InsightFace](https://github.com/deepinsight/insightface) và 2 model checkpoints từ OpenCV
## Step 2
```
conda create -n ckin python=3.8
conda activate ckin
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
pip install -r requirements.txt
```
Mở app DroidCam và chạy ```python main.py -u [URL] -s [Tên directory lưu embedding]``` với URL chính là IP Cam Access trên DroidCam. 

Ví dụ:
```
python main.py -u http://192.168.1.2:4747/video -s new_embedding
```

## Save new face
```
python save_face.py -n [NAME] -s [EMBEDDING_DIRECTORY]
```
