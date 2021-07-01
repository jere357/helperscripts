from tqdm import tqdm
import pandas as pd
import cv2
import os
import json
import csv
import os.path
"""
folder structure:
datasetcreator.py
videos/video1.mp4
videos/video1.json
...
generates -> information.csv in the same folder as datasetcreator.py
          -> class_images/class1.jpg
                    ...
class_images folder contains all class images extracted from video frames
"""
def extract_bboxes(frame, detections, video_name, class_folder = 'class_images'):
    bbox_list = []
    try:
        os.mkdir(class_folder)
    except OSError as error:
        pass
        #nebitan error - folder alrdy exsits WOW IMAGINE
    #TODO: mozda odjebat izolatore koji su blizu ruba ekrana
    #TODO: mozda odjebat sve bboxeve koji su manji od neke dimenzije koju "zelimo"
    #NAPUNI MI SE RAM LMFAOOO ROFL HAHAHXDXD
    for index, detection in detections.iterrows():
        xmin = detection['xmin']
        xmax = detection['xmax']
        ymin = detection['ymin']
        ymax = detection['ymax']
        bbox = frame[ymin:ymax, xmin:xmax]
        bbox_list.append({'_id' : detection['_id'],
                                 'label' : detection['label'],
                                 'frame_number' : detection['frame_number'],
                                 'video_name' : video_name,
                                 'confidence' : detection['confidence']})
        imgname = class_folder + '/' + detection['_id'] + '.jpg'
        cv2.imwrite(imgname, bbox)
    return bbox_list
    
def get_files(foldername):
    json_list = []
    video_list = []
    for file in os.listdir(foldername):
        if(file.lower().endswith('.json')):
            json_list.append(file)
        elif(file.lower().endswith('.mp4')):
            video_list.append(file)
        else:
            print("unexpected file: {}".format(str(file)))
    json_list.sort()
    video_list.sort()
    return json_list, video_list
def extract_class(foldername, videofile, jsonfile, class_name = 'Insulators', csv_filename='information.csv'):
    file_exists = os.path.isfile(csv_filename)
    cap = cv2.VideoCapture(foldername + videofile)
    jsonpath = str(foldername + jsonfile)
    information_list = [] #list filled with dictionaries containing data regarding each photo saved =XD
    data = json.load(open(jsonpath))
    df = pd.DataFrame(data["classified"])
    df.sort_values(by = ['frame_number'])
    #TODO: filter sa confidence thresholdom 
    insulators_df = df[df['label'] == class_name] #filtriraj samo class_name
    for current_frame in tqdm(insulators_df['frame_number'].unique()):
        detections = insulators_df[insulators_df['frame_number'] == current_frame]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame) -1 )
        ret, frame = cap.read()
        if ret:
            frame_list = extract_bboxes(frame, detections, videofile)
        else:
            tqdm.write("error occured while trying to read {}. frame in video".format(current_frame))
        information_list.extend(frame_list)
    print("video {} had {} pictures of class {}".format(videofile, len(information_list), class_name))
    with open (csv_filename, 'a') as csvfile:
        headers = ['_id', 'label', 'frame_number', 'video_name', 'confidence']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerows(information_list)
    #zapisi information list u csv

if __name__ == "__main__":
    foldername = 'videos/'
    csv_filename = 'information.csv'
    json_list, video_list = get_files(foldername)
    for jsonfile, videofile in zip(json_list, video_list):
        print("obradujem: {} {}".format(jsonfile, videofile))
        extract_class(foldername, videofile, jsonfile)