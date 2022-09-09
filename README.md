# CustomActionRecognition-TensorFlow-CNN-LSTM
Custom Action Recognition using TensorFlow (CNN + LSTM), I create my own Model to predict my custom Action Classes.<br>
Here is my **Sample Output**:<br>
Its predicting **Horse Race** and **Rope Climbing** Classes. (you can see the prediction on **Top Left Corner** (in color ***Green***))

https://user-images.githubusercontent.com/88816150/189287071-f4ec071b-1f9f-492c-934c-f587a50dbdc4.mp4


## But If You need create Model or Project, only focusing Action Area (ROI)
```diff
+ Go to branch "detect" 
git checkout detect

! Follow the Instruction
```
**Output Example**:

https://user-images.githubusercontent.com/88816150/189055326-a02ade8f-0129-4e1a-886c-05cd180bec4f.mp4

# Custom Action Recognition Model
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
## Inference
`--dataset`: path to dataset dir <br>
`--seq_len`: The number of frames of a video that will be fed to the model as one sequence <br>
`--size`: The height and width to which each video frame will be resized in our dataset <br>
`--model`: path to trained custom model <br>
`--conf`: Model Prediction Confidence <br>
`--save`: Save output video ("output.mp4")
`--source`: path to test video
- Web-cam: `--source 0`
```
python3 inference.py --dataset data/ --seq_len 20 \
                     --size 64 --model LRCN_model.h5 \
                     --conf 0.75 --source data/test/video.mp4

# to save output video
python3 inference.py --dataset data/ --seq_len 20 \
                     --size 64 --model LRCN_model.h5 \
                     --conf 0.75 --source data/test/video.mp4 \
                     --save
```
```
# web-cam
python3 inference.py --dataset data/ --seq_len 20 \
                     --size 64 --model LRCN_model.h5 \
                     --conf 0.75 --source 0

# to save output video
python3 inference.py --dataset data/ --seq_len 20 \
                     --size 64 --model LRCN_model.h5 \
                     --conf 0.75 --source 0 \
                     --save
```
**To Exit Window - Press Q-key**
