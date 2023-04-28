# FaceNet for Face Recognition
Face Recognition with FaceNet &amp;&amp; Tensorflow &amp;&amp; OpenCV

## 1. DataSet
Dataset/FaceData:
- raw: Folder contains raw images (thư mục chứa ảnh raw)
- processed: Folder contains processed images (thư mục chứa ảnh đã được xử lý)

Note:
- folder images must be named by person's name (tên thư mục chứa ảnh phải là tên của người đó)
- processed images have size 160x160 (ảnh đã được xử lý có kích thước 160x160)

### 1.1 Create dataset with raw images
Step 1: Create folder contains raw images (tạo thư mục chứa ảnh raw). 

    Ex: ./raw/tan_phat/*.jpg

Step 2: Run on terminal (chạy trên terminal)

    python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160 --margin 32  --random_order --gpu_memory_fraction 0.25

### 1.2 Create dataset with camera (realtime)
Step 1: Run on terminal (chạy trên terminal)

    python NewUser.py

Step 2: Enter your name 
    
    Enter your name: tan_phat

Step 3: Press 'q' to quit

## 2. Train model with FaceNet

### 2.1 Download model FaceNet
You can download model FaceNet 2018 from [here](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view?usp=sharing), extract and put it in folder `Models`

### 2.2 Train model

    python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000

## 3. Test model 
### 3.1 Test with video
    python src/face_rec.py --path video/haoquangrucro.mp4

### 3.2 Test with camera (realtime)
    python src/face_rec_cam.py 
