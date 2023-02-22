# CustomActionRecognition-TensorFlow-CNN-LSTM
Custom Action Recognition using TensorFlow (CNN + LSTM), this repository will help you to create your own Custom Action Recognition Model.

## 🚀 New Updates
- (**03/11/2022**): **MLFlow** to Model training. So that we can Manage the ML lifecycle
  - Experimentation
  - Reproducibility
  - Deployment
  - Central Model Registry
 
- (**17/11/2022**): Added **VideoGenerator**
  - Why we need **VideoGenerator** 🤔?
    - Previously we are unsing **Data Extraction** technic, take all data and stored in array.
    - But if we have large data, we will get **RAM outoff memory Error**.
    - Data Extraction take **60-80% of total time**.
  - Advantages of **VideoGenerator** 🥳
    - Videogenerator solve all of these problems.
    - Its same like Image-generator
    - Option to add **Data Augmentation**

## 🥱 My Experience
I create my own Model to predict my custom Action Classes.<br>
Here is my **Sample Output**:<br>
Its predicting **Horse Race** and **Rope Climbing** Classes. (you can see the prediction on **Top Left Corner** (in color ***Green***))

https://user-images.githubusercontent.com/88816150/189287071-f4ec071b-1f9f-492c-934c-f587a50dbdc4.mp4


## But If You need create Model, focusing Action Area (ROI) with help of YOLOv7 Object Detection Model
```diff
+ Go to branch "detect" 
git checkout detect

! Follow the Instruction
```
**Output Example**:

https://user-images.githubusercontent.com/88816150/189055326-a02ade8f-0129-4e1a-886c-05cd180bec4f.mp4

# 😎 Custom Action Recognition Model
### Clone this Repository
```
git clone https://github.com/naseemap47/CustomActionRecognition-TensorFlow-CNN-LSTM.git
cd CustomActionRecognition-TensorFlow-CNN-LSTM
```
### Install dependency
```
pip3 install -r requirement.txt
```
```
xargs sudo apt-get install <packages.txt
```
### Take Custom Action Recognition Data
Example: **UCF50 Dataset** (Demo)<br>
**Downlaod the UCF50 Dataset:**
```
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF50.rar

#Extract the Dataset
unrar x UCF50.rar
```
**Inside Data Directory - Folder with Class Name - Inside each class folder - Video data for that Action Class**
## 🤖 Train Custom Model
:memo: Note: <br>
Model Dir: **LRCN** and **convLSTM** for sample Demo to understand how will be the output, You can remove that Dir.
If you NOT remove the Dir, its will never affect your Model or Training,
It will replace with your Model <br>

`--dataset`: path to dataset dir <br>
`--seq_len`: The number of frames of a video that will be fed to the model as one sequence <br>
`--size`: The height and width to which each video frame will be resized in our dataset <br>
`--model`: Choose Model Type
  - **convLSTM**: `convLSTM`
  - **LRCN**: `LRCN` <br>

`--epochs`: Number of epochs for model Training <br>
`--batch_size`: Size of Batch on Training the Model

### 1.convLSTM
```
python3 train.py --dataset data/ --seq_len 20 \
                 --size 64 --model convLSTM \
                 --epochs 50 --batch_size 4
```
### 2.LRCN
```
python3 train.py --dataset data/ --seq_len 20 \
                 --size 64 --model LRCN \
                 --epochs 70 --batch_size 4
```
**The Output model, history plot and Model str plot will be Saved in corresponding its Model Dir**
### :warning: Training on Colab [Error]
**DNN library is not found**
```
# Install latest version
!apt install --allow-change-held-packages libcudnn8=8.4.1.50-1+cuda11.6
```

## 📊 MLFlow UI
**MLflow** is an open source platform for managing the end-to-end machine learning lifecycle <br>
terminal is in the same directory that contains mlruns, and
type the following:
```
mlflow ui

# OR
mlflow ui -p 1234
```
The command mlflow ui hosts the MLFlow UI locally on the default
port of **5000**.<br>
However, the options `-p 1234` tell it that you want to host it specifically on the port **1234**.<br>

open a browser and type in http://localhost:1234 or
http://127.0.0.1:1234

## 📺 Inference
`--model`: path to trained custom model <br>
`--conf`: Model Prediction Confidence <br>
`--save`: Save output video ("output.mp4") <br>
`--source`: path to test video
- Web-cam: `--source 0`
```
python3 inference.py --model LRCN_model.h5 \
                     --conf 0.75 --source data/test/video.mp4

# to save output video
python3 inference.py --model LRCN_model.h5 --conf 0.75 \
                     --source data/test/video.mp4 \
                     --save
```
```
# web-cam
python3 inference.py --model LRCN_model.h5 \
                     --conf 0.75 --source 0

# to save output video
python3 inference.py --model LRCN_model.h5 \
                     --conf 0.75 --source 0 \
                     --save
```
**To Exit Window - Press Q-key**
