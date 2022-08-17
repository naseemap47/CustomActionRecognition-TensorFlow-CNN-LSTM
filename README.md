# CustomActivityRecognition-TensorFlow-CNN-LSTM
Custom Activity Recognition using TensorFlow (CNN + LSTM)
## Custom Activity Recognition model
### Clone this Repository
```
git clone https://github.com/naseemap47/CustomActivityRecognition-TensorFlow-CNN-LSTM.git
cd CustomActivityRecognition-TensorFlow-CNN-LSTM
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
python3 train.py --dataset data/ --seq_len 20 --size 64 --model convLSTM --epochs 50 --batch_size 4
```
### 2.LRCN
```
python3 train.py --dataset data/ --seq_len 20 --size 64 --model LRCN --epochs 70 --batch_size 4
```
**The Output model, history plot and Model str plot will be Saved in corresponding its Model Dir**
## Inference
`--dataset`: path to dataset dir <br>
`--seq_len`: The number of frames of a video that will be fed to the model as one sequence <br>
`--size`: The height and width to which each video frame will be resized in our dataset <br>
`--model`: path to trained custom model <br>
`--source`: path to test video
- Web-cam: `--source 0`
```
python3 inference.py --dataset data/ --seq_len 20 --size 64 --model LRCN_model.h5 --source data/test/video.mp4

# web-cam
python3 inference.py --dataset data/ --seq_len 20 --size 64 --model LRCN_model.h5 --source 0
```
**To Exit Window - Press Q-key**
