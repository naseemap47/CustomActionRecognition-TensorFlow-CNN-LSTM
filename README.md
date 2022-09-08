# CustomActivityRecognition-TensorFlow-CNN-LSTM
Custom Activity Recognition using TensorFlow (CNN + LSTM) <br>
With help of Object detetction model (YOLOv7), We focusing on **Action Area (ROI)** NOT entire Frame.<br>
It can help to **increase Action Model Accuracy** and to **identify different types of Action in a single frame**.<br>

**Example**:

https://user-images.githubusercontent.com/88816150/189055326-a02ade8f-0129-4e1a-886c-05cd180bec4f.mp4

## Custom Activity Recognition model with Object Detection (YOLOv7)
### Clone this Repository
```
git clone https://github.com/naseemap47/CustomActivityRecognition-TensorFlow-CNN-LSTM.git
cd CustomActivityRecognition-TensorFlow-CNN-LSTM
git checkout detect
```
### Install dependency
```
pip3 install -r requirement.txt
```
```
xargs sudo apt-get install <packages.txt
```
### Take Custom Activity Recognition Data
Example: **UCF50 Dataset** <br>
**Downlaod the UCF50 Dataset:**
```
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF50.rar

#Extract the Dataset
unrar x UCF50.rar
```
**Inside Data Directory - Folder with Class Name - Inside each class folder - Video data for that Action Class**
## Train Custom Model
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
`--batch_size`: Size of Batch on Training the Model <br>
`--yolov7_model`: Path to YOLOv7 detection Model <br>
`--yolov7_conf`: YOLOv7 detection model confidenece (0<conf<1)

### 1. convLSTM
```
python3 train.py --dataset data/ --seq_len 20 \
                 --size 64 --model convLSTM \
                 --epochs 50 --batch_size 4 \
                 --yolov7_model best.pt --yolov7_conf 0.6
```
### 2. LRCN
```
python3 train.py --dataset data/ --seq_len 20 \
                 --size 64 --model LRCN \
                 --epochs 70 --batch_size 4 \
                 --yolov7_model best.pt --yolov7_conf 0.6
```
**The Output model, history plot and Model str plot will be Saved in corresponding its Model Dir**
### :warning: Training on Colab [Error]
**DNN library is not found**
```
# Install latest version
!apt install --allow-change-held-packages libcudnn8=8.4.1.50-1+cuda11.6
```
## Inference
`--dataset`: path to dataset dir <br>
`--seq_len`: The number of frames of a video that will be fed to the model as one sequence <br>
`--size`: The height and width to which each video frame will be resized in our dataset <br>
`--model`: path to trained custom model <br>
`--act_conf`: Action Model Prediction Confidence (0<conf<1) <br>
`--source`: path to test video
- Web-cam: `--source 0` <br>

`--save`: To save output video in "output.avi" <br>
`--detect_model`: Path to YOLOv7 Model <br>
`--yolov7_conf`: YOLOv7 detection Model conf (0<conf<1)
```
python3 inference.py --dataset data/ --seq_len 20 \
                     --size 64 --model LRCN_model.h5 \
                     --act_conf 0.75 --source data/test/video.mp4 \
                     --detect_model best.pt --yolov7_conf 0.6

# if need to save output
python3 inference.py --dataset data/ --seq_len 20 \
                     --size 64 --model LRCN_model.h5 \
                     --act_conf 0.75 --source data/test/video.mp4 \
                     --detect_model best.pt --yolov7_conf 0.6 \
                     --save
```
```
# web-cam
python3 inference.py --dataset data/ --seq_len 20 \
                     --size 64 --model LRCN_model.h5 \
                     --act_conf 0.75 --source 0 \
                     --detect_model best.pt --yolov7_conf 0.6

# if need to save output
python3 inference.py --dataset data/ --seq_len 20 \
                     --size 64 --model LRCN_model.h5 \
                     --act_conf 0.75 --source 0 \
                     --detect_model best.pt --yolov7_conf 0.6 \
                     --save
```
**To Exit Window - Press Q-key**
