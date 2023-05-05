# Live Object Recognition Script Python
## Description
This script is used to recognize objects in real time using the webcam. It uses the YOLOv3 algorithm to recognize the objects. The script is written in Python and uses the OpenCV library to capture the webcam feed and to draw the bounding boxes around the recognized objects. The script is also able to recognize multiple objects at the same time. The script
is able to recognize 80 different objects
## Requirements
- Python 3.6 or higher
- OpenCV 4.0 or higher
- Numpy
- YOLOv3 weights
- COCO names file
- Webcam

## Installation
1. Install Python 3.6 or higher
2. Install OpenCV 4.0 or higher
3. Install Numpy
4. Download the YOLOv3 weights from [here](https://pjreddie.com/media/files/yolov3.weights)
5. Download the COCO names file from [here]( https://github.com/pjreddie/darknet/blob/master/data/coco.names )
6. Place the YOLOv3 weights and the COCO names file in the same directory as the script
7. Run the script using the command `python3 object_recognition.py`
8. Press `q` to quit the script 
9. Enjoy!
