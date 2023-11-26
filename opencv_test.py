##### OpenCV test #####

import cv2
import numpy as np
import os
import requests
from urllib.request import urlopen

weights_path = "yolov4.weights"
cfg_path = "yolov4.cfg"
names_path = "coco.names"

weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights"
cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

def download_file(url, file_path):
    if not os.path.isfile(file_path):
        print(f"Downloading {file_path}...")
        r = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(r.content)

download_file(weights_url, weights_path)
download_file(cfg_url, cfg_path)
download_file(names_url, names_path)

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
classes = []
colors = []
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

layer_names = net.getLayerNames()

out_layers_indices = net.getUnconnectedOutLayers()

output_layers = [layer_names[i - 1] for i in out_layers_indices]

# Capture video from webcam
cap = cv2.VideoCapture(0)
mode = "normal" 

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    original_frame = frame.copy()  

    # Process frame based on the current mode
    if mode == "blur":
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
    elif mode == "canny":
        frame = cv2.Canny(frame, 80, 200)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  
    elif mode == "hsv":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    elif mode == "threshold":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(gray, 87, 255, cv2.THRESH_BINARY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif mode == "sobel":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = cv2.magnitude(sobelx, sobely)
        frame = cv2.convertScaleAbs(sobel)  
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif mode == "object_detection":
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                color = colors[class_ids[i]]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    # Display current mode on the top right of the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Mode: {mode}"
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    textX = frame.shape[1] - textsize[0] - 10  
    textY = textsize[1] + 10  
    cv2.putText(frame, text, (textX, textY), font, 1, (155, 155, 155), 2)

    cv2.imshow("Image", frame)

    # Key press events
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break
    elif key == ord('b'):
        mode = "blur" if mode != "blur" else "normal"
    elif key == ord('c'):
        mode = "canny" if mode != "canny" else "normal"
    elif key == ord('h'):
        mode = "hsv" if mode != "hsv" else "normal"
    elif key == ord('t'):
        mode = "threshold" if mode != "threshold" else "normal"
    elif key == ord('s'):
        mode = "sobel" if mode != "sobel" else "normal"
    elif key == ord('o'):
        mode = "object_detection" if mode != "object_detection" else "normal"

cap.release()
cv2.destroyAllWindows()
