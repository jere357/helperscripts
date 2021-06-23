import cv2
import json

f = open('1_DJI_0687_detections.json',)
detections = json.load(f)
detections_in_frames = {}
#print(detections['classified'].keys())
for detection in detections['classified']:
    if detection['frame_number'] not in detections_in_frames.keys():
        detections_in_frames[detection['frame_number']] = []
    detections_in_frames[detection['frame_number']].append(detection)

#print("detections in frames {}".format(detections_in_frames))
cap = cv2.VideoCapture("1_DJI_0687_lowres.MP4")
frame_number = 0

while True:
    # Read new frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    frame_number_as_key = str(frame_number)
    if frame_number_as_key in detections_in_frames.keys():
        detections_in_this_frame = detections_in_frames[frame_number_as_key]
        for detection in detections_in_this_frame:
            if detection['label'] == "Insulators":
                xmin = detection['xmin']
                xmax = detection['xmax']
                ymin = detection['ymin']
                ymax = detection['ymax']
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
                cv2.putText(frame,str(round(detection['confidence'],2)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            elif detection['label'] == "Broken insulator":
                xmin = detection['xmin']
                xmax = detection['xmax']
                ymin = detection['ymin']
                ymax = detection['ymax']
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                cv2.putText(frame,str(round(detection['confidence'],2)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    print(str(frame_number/23.0))
    cv2.imshow("frame", frame)
    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        break