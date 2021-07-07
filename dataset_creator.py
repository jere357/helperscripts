from tqdm import tqdm
import pandas as pd
import cv2
import os
import json
import csv
import os.path
import math
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
"""
function used to keep bounding box extraction within the frame - edge cases might be problematic with padding
"""
def contain(xmin, xmax, ymin, ymax, shape):
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > shape[1]:
        xmax = shape[1] - 1
    if ymax > shape[0]:
        ymax = shape[0] - 1
    return xmin, xmax, ymin, ymax

#TODO: mozda uopce ne triba slat lowresframe sta ja znan puca mi kurac
def extract_bboxes(lowresframe, fullresframe, detections, video_name, resolution_ratio, class_folder = 'class_images', padding = 15, test_mode = False):
    bbox_list = []
    try:
        os.mkdir(class_folder)
    except OSError as error:
        pass
        #nebitan error - folder alrdy exsits WOW IMAGINE
    #TODO: mozda odjebat izolatore koji su blizu ruba ekrana
    #TODO: mozda odjebat sve bboxeve koji su manji od neke dimenzije koju "zelimo" OVO SIGURNO PREMA PIXEL POVRSINI SLIKE
    for index, detection in detections.iterrows():
        xmin = round(detection['xmin']*resolution_ratio[0] - padding)
        xmax = round(detection['xmax']*resolution_ratio[0] + padding)
        ymin = round(detection['ymin']*resolution_ratio[0] - padding)
        ymax = round(detection['ymax']*resolution_ratio[0] + padding)
        xmin, xmax, ymin, ymax = contain(xmin,xmax,ymin,ymax, fullresframe.shape)
        bbox = fullresframe[ymin:ymax, xmin:xmax]
        bbox_list.append({'_id' : detection['_id'],
                                 'label' : detection['label'],
                                 'frame_number' : detection['frame_number'],
                                 'video_name' : video_name,
                                 'confidence' : detection['confidence']})
        imgname = class_folder + '/' + detection['_id'] + '.jpg'
        if test_mode:
            print( f'full rez: {fullresframe.shape} low rez frame: {lowresframe.shape} bbox frame: {bbox.shape}')
            print(f'bbox dimenzije : ({xmin},{ymin}):({xmax},{ymax})')
            cv2.imshow("fullres frame", fullresframe)
            cv2.imshow("lowres", lowresframe)
            cv2.imshow(imgname, bbox)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(imgname, bbox)
    return bbox_list
def get_files(foldername):
    json_list = []
    video_list = []
    video_list_fullres = []
    for file in os.listdir(foldername):
        if(file.lower().endswith('.json')):
            json_list.append(file)
        elif(file.lower().endswith('.mp4')):
            if '_lowres' in file.lower():
                video_list.append(file)
            else:
                video_list_fullres.append(file)
        else:
            print("unexpected file: {}".format(str(file)))
    json_list.sort()
    video_list.sort()
    video_list_fullres.sort()
    return json_list, video_list, video_list_fullres
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
def extract_class_fullres(foldername, lowresvideo, fullresvideo, jsonfile, full_resolution_extraction = True, class_name = 'Insulators', csv_filename='information.csv'):
    file_exists = os.path.isfile(csv_filename)
    lowrescap = cv2.VideoCapture(foldername + lowresvideo)
    fullrescap = cv2.VideoCapture(foldername + fullresvideo)
    resolution_ratio = ()
    ret, lowresframe = lowrescap.read()
    ret, fullresframe = fullrescap.read()
    if ret:
        #print(lowresframe.shape, fullresframe.shape)
        resolution_ratio = (fullresframe.shape[0]/lowresframe.shape[0], fullresframe.shape[0]/lowresframe.shape[0])
    else:
        print("idk wy doin dude")
    #print(resolution_ratio)
    jsonpath = str(foldername + jsonfile)
    information_list = [] #list filled with dictionaries containing data regarding each photo saved =XD
    data = json.load(open(jsonpath))
    df = pd.DataFrame(data["classified"])
    df.sort_values(by = ['frame_number'])
    #TODO: filter sa confidence thresholdom 
    #TODO: provjeri vrati li zapravo sliku i stae s tin tocno
    insulators_df = df[df['label'] == class_name] #filtriraj samo class_name
    for current_frame in tqdm(insulators_df['frame_number'].unique()):
        detections = insulators_df[insulators_df['frame_number'] == current_frame]
        lowrescap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame) -1 )
        fullrescap.set(cv2.CAP_PROP_POS_FRAMES, int(current_frame) -1 )
        ret1, lowresframe = lowrescap.read()
        ret2, fullresframe = fullrescap.read()
        if ret1 and ret2:
            frame_list = extract_bboxes(lowresframe, fullresframe, detections, lowresvideo, resolution_ratio)
        else:
            tqdm.write("error occured while trying to read {}. frame in video".format(current_frame))
        information_list.extend(frame_list)
    print("video {} had {} pictures of class {}".format(lowresvideo, len(information_list), class_name))
    with open (csv_filename, 'a') as csvfile:
        headers = ['_id', 'label', 'frame_number', 'video_name', 'confidence']
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerows(information_list)
    #zapisi information list u csv

if __name__ == "__main__":
    videos_foldername = 'data/'
    #fullres_videos_foldername = 'videos_fullres'
    csv_filename = 'information.csv'
    json_list, video_list, video_list_fullres = get_files(videos_foldername)
    print(json_list)
    print(video_list_fullres)
    print(video_list)
    #fullres for petljun
    for jsonfile, lowresvideo, fullresvideo in zip(json_list, video_list, video_list_fullres):
        print("obradujem: {} {} {}".format(jsonfile, fullresvideo, lowresvideo))
        #extract_class(foldername, videofile, jsonfile)
        extract_class_fullres(videos_foldername, lowresvideo, fullresvideo, jsonfile)
