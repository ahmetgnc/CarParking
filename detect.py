import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import tensorflow.keras.preprocessing.image as imgpros
import time

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6

# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []
    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3, 8, 6]:
            car_boxes.append(box)
    return np.array(car_boxes)


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Location of parking spaces
file = open("parkinglots.txt", "r")
parkinglot_boxes = file.read()

parkarr = parkinglot_boxes.split('\n')
parkarr.pop(0)
parkarr.remove('')

pinned_car_boxes = []
for box in parkarr:
    box = box.split(' ')
    box[0] = int(box[0])
    box[1] = int(box[1])
    box[2] = int(box[2])
    box[3] = int(box[3])
    pinned_car_boxes.append([box[1], box[0], box[3], box[2]])
pinned_car_boxes = np.array(pinned_car_boxes)

###VIDEO####

# Video file or camera to process - set this to 0 to use your webcam instead of a video file
VIDEO_SOURCE = os.path.join(ROOT_DIR,"test_video/parking.mp4")

# # Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

# FPS calculate and timing to frames
seconds = 3
fps = video_capture.get(cv2.CAP_PROP_FPS)  # Gets the frames per second
multiplier = int(fps * seconds)


while video_capture.isOpened():
    success, frame = video_capture.read()
    if success:
        frameId = int(round(video_capture.get(1)))
        if frameId % multiplier == 0:
            #rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image = imgpros.img_to_array(frame)
            # Run the image through the Mask R-CNN model to get results.
            t1 = time.time()
            results = model.detect([rgb_image], verbose=0)
            t2 = time.time()
            print('Time model running = ' + str(t2-t1))
            # Mask R-CNN assumes we are running detection on multiple images.
            # We only passed in one image to detect, so only grab the first result.
            r = results[0]
            parked_car_boxes = get_car_boxes(r['rois'], r['class_ids'])
            print(parked_car_boxes)
            print(pinned_car_boxes)

            overlaps = mrcnn.utils.compute_overlaps(pinned_car_boxes, parked_car_boxes)
            # overlaps = overlaps.transpose()

            free_space = False
            # for parking_area, overlap_areas in zip(parked_car_boxes, overlaps):
            for parking_area, overlap_areas in zip(pinned_car_boxes, overlaps):
                max_IoU_overlap = np.max(overlap_areas)
                # x1, y1 ,x2, y2 = parking_area
                y1, x1, y2, x2 = parking_area
                if max_IoU_overlap < 0.25:
                    # Parking space not occupied! Draw a green box around it
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # Flag that we have seen at least one open space
                    free_space = True
                else:
                    # Parking space is still occupied - draw a red box around it
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{max_IoU_overlap:0.2}", (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255))
            # for box in parked_car_boxes:
            #     print("Car: ", box)
            #
            #     y1, x1, y2, x2 = box
            #
            #     # Draw the box
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.imwrite('gif/'+ str(frameId)+'.jpg', frame)
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else: break
video_capture.release()
cv2.destroyAllWindows()