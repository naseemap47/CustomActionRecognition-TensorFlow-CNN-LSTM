import cv2
from collections import deque
import numpy as np
from keras.models import load_model
import argparse
import os
import tensorflow as tf


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to csv Data")
# Specify the number of frames of a video that will be fed to the model as one sequence.
ap.add_argument("-l", "--seq_len", type=int, default=20,
                help="length of Sequence")
ap.add_argument("-s", "--size", type=int, default=64,
                help="Specify the height and width to which each video frame will be resized in our dataset.")
ap.add_argument("-m", "--model", type=str,  required=True,
                help="path to model.h5")
ap.add_argument("-v", "--source", type=str, required=True,
                help="path to video or web-cam")
ap.add_argument("-c", "--conf", type=float, required=True,
                help="Prediction confidence (0<conf<1)")
ap.add_argument("--save", action='store_true',
                help="Save video")

args = vars(ap.parse_args())
DATASET_DIR = args["dataset"]
SEQUENCE_LENGTH = args["seq_len"]
IMAGE_SIZE = args["size"]
path_to_model = args["model"]
video_path = args["source"]
thresh = args['conf']
save = args['save']

CLASSES_LIST = sorted(os.listdir(DATASET_DIR))

# Load LRCN_model
saved_model = load_model(path_to_model)

# Web-cam
if video_path.isnumeric():
    video_path = int(video_path)
video_reader = cv2.VideoCapture(video_path)

# Get the width and height of the video.
original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video_reader.get(cv2.CAP_PROP_FPS)

# Write Video
if save:
    out_vid = cv2.VideoWriter('output.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         fps, (original_video_width, original_video_height))

# Declare a queue to store video frames.
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

while video_reader.isOpened():
    success, frame = video_reader.read()

    if not success:
        break
    
    # BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the Frame to fixed Dimensions.
    resized_frame = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    # Normalize - between 0 and 1.
    normalized_frame = tf.keras.utils.img_to_array(resized_frame) / 255.

    # Appending the pre-processed frame into the frames list.
    frames_queue.append(normalized_frame)

    # Check if the number of frames in the queue are equal to the fixed sequence length.
    if len(frames_queue) == SEQUENCE_LENGTH:

        # Pass the normalized frames to the model and get the predicted probabilities.
        predicted_labels_probabilities = saved_model.predict(
            np.expand_dims(frames_queue, axis=0))[0]

        # Get the index of class with highest probability.
        predicted_label = np.argmax(predicted_labels_probabilities)
        
        if max(predicted_labels_probabilities) > thresh:

            # Get the class name using the retrieved index.
            predicted_class_name = CLASSES_LIST[predicted_label]

            # Write predicted class name on top of the frame.
            cv2.putText(frame, predicted_class_name, (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            cv2.putText(frame, 'Action NOT Detetced', (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Write Video
    if save:
        out_vid.write(frame)

    cv2.imshow('Out', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_reader.release()
if save:
    out_vid.release()
cv2.destroyAllWindows()
