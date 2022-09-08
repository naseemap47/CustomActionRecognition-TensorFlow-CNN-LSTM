import cv2
from collections import deque
import numpy as np
from keras.models import load_model
import argparse
import os
from utils.hubconf import custom
from utils.plots import plot_one_box


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
ap.add_argument("-d", "--detect_model", type=str,  required=True,
                help="path to YOLOv7 model")
ap.add_argument("-dc", "--yolov7_conf", type=float, default=0.6,
                help="YOLOv7 detection model confidenece (0<conf<1)")

args = vars(ap.parse_args())
DATASET_DIR = args["dataset"]
SEQUENCE_LENGTH = args["seq_len"]
IMAGE_SIZE = args["size"]
path_to_model = args["model"]
video_path = args["source"]
thresh = args['conf']
save = args['save']
yolov7_model_path = args["detect_model"]
yolov7_conf = args["yolov7_conf"]

CLASSES_LIST = sorted(os.listdir(DATASET_DIR))

# Load LRCN_model
saved_model = load_model(path_to_model)

# YOLOv7 Model
yolov7_model = custom(path_or_model=yolov7_model_path)

# Web-cam
if video_path.isnumeric():
    video_path = int(video_path)
video_reader = cv2.VideoCapture(video_path)

# Get the width and height of the video.
original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Write Video
if save:
    out_vid = cv2.VideoWriter('output.avi',
                              cv2.VideoWriter_fourcc(*'MJPG'),
                              10, (original_video_width, original_video_height))

# Declare a queue to store video frames.
frames_queue = deque(maxlen=SEQUENCE_LENGTH)

while video_reader.isOpened():
    success, frame = video_reader.read()

    if not success:
        break

    bbox_list = []
    # Action - ROI
    results = yolov7_model(frame)
    # Bounding Box
    box = results.pandas().xyxy[0]

    for i in box.index:
        xmin, ymin, xmax, ymax, conf = int(box['xmin'][i]), int(
            box['ymin'][i]), int(box['xmax'][i]), int(box['ymax'][i])
        bbox_list.append([xmin, ymin, xmax, ymax, conf])

    if len(bbox_list) > 0:
        for bbox in bbox_list:
            if bbox[4] > yolov7_conf:
                frame_roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # Resize the Frame to fixed Dimensions.
                resized_frame = cv2.resize(frame_roi, (IMAGE_SIZE, IMAGE_SIZE))

                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
                normalized_frame = resized_frame / 255

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

                        plot_one_box(
                            bbox, frame, label=predicted_class_name,
                            color=[0, 255, 0], line_thickness=2
                        )

                    else:
                        plot_one_box(
                            bbox, frame, label='Action NOT Detetced',
                            color=[0, 0, 255], line_thickness=2
                        )

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
