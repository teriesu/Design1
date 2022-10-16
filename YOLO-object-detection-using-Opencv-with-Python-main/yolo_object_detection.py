import cv2 as cv
import numpy as np

# load yolo
net = cv.dnn.readNet("yolov3.weights",
                     "yolov3.cfg")
clasees = []
with open("coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# print(classes)
layer_name = net.getLayerNames()
output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
print(classes)
cap = cv.VideoCapture(1)

while True:

    ret, frame, = cap.read()
    if ret == False:
        break
    # Detect Objects
    height, width, _ = frame.shape
    frame_resized = cv.resize(frame, (416, 416))

    blob = cv.dnn.blobFromImage(
        frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layer)
    # print(outs)

    # Showing Information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detection
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # cv.circle(img, (center_x, center_y), 10, (0, 255, 0), 2 )
                # Reactangle Cordinate
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # print(len(boxes))
    # number_object_detection = len(boxes)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)

    font = cv.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # print(label)
            color = colors[i]
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if label == 'handbag' or label == 'bottle' or label == 'wine glass' or label == 'book' or label == 'vase' or label == 'toothbrush':
                cv.putText(frame, label, (x, y + 30), font, 3, color, 3)
    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
