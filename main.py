import cv2
import numpy as np

# Load YOLOv3 network
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Set up classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up colors for each class
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Set up video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Get dimensions of the frame
    (H, W) = frame.shape[:2]

    # Preprocess input image
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input blob for the neural network
    net.setInput(blob)

    # Get the outputs of the output layers
    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists for detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each output layer
    for output in layerOutputs:
        # Loop over each detection in the output layer
        for detection in output:
            # Get the class ID and confidence of the current detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak detections
            if confidence > 0.5:
                # Scale the bounding box coordinates to the input image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Calculate the top-left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Add the bounding box, confidence, and class ID to their respective lists
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the original frame
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()