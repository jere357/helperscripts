import requests
import urllib3
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import subprocess
"""
assumed folder structure:
maximo_api.py
videos/ folder containing videos

"""
def convert_video(filename, folder_name):
    video_folder_name = folder_name + '/' + folder_name + '_converted'
    try:
        os.mkdir(video_folder_name)
    except OSError as error:
        #folder alrdy exists - WHO CARES 
        pass
    command =  "ffmpeg -i " + folder_name + '/' + filename + " -filter:v scale=1000:-2 -c:v libx264 -crf 17 " + video_folder_name + '/' + filename[:-4] + '.mp4'
    a = subprocess.call(command, shell = True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
def token_request():
    maximo_response = maximo_session.post('https://192.168.91.179/visual-inspection/api/tokens',
                                         json = {'grant_type': "password", 
                                                 'username': "poc",
                                                 'password': "pcvision10!"},
                                         verify = False)
    return maximo_response.status_code, json.loads(maximo_response.text)
def detect_insulators (filename):
    with open (filename, 'rb') as f: 
            # WARNING! verify = False is here to allow an untrusted cert! 
            maximo_response = maximo_session.post(insulators_url, 
                                                  files = {'files': (filename, f), 
                                                           'genCaption': "false"
                                                           }, 
                                                  verify = False)
    return maximo_response.status_code, json.loads(maximo_response.text)
def import_dataset(filename):
    with open(filename, 'rb') as f:
        maximo_response = maximo_session.post(dataset_import_url,
                                                headers = {'X-Auth-Token' : token},
                                                files = {
                                                    'name' : "API_UPLOAD_TEST",
                                                    'files' : (filename, f)
                                                    }
                                                )
def get_inference_output (inference_url, token):
    maximo_response = maximo_session.get(inference_url,
                                         headers = {'X-Auth-Token': token,'Content-type': 'application/json'},
                                         verify = False)
    return maximo_response.status_code, json.loads(maximo_response.text)
def video_resizing (filename):
    cap = cv2.VideoCapture(filename)
    W = int(cap.get(3))
    H = int(cap.get(4))
    frame_size = (1000, int(np.ceil(1000/W*H)))
    fourcc = cv2.VideoWriter_fourcc('M','P','G','4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_filename = filename[:-4]+'_1000.MP4'
    out = cv2.VideoWriter(out_filename, fourcc, fps, frame_size)
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if ret == True:
            b = cv2.resize(frame, dsize = frame_size, interpolation = cv2.INTER_LINEAR)
            out.write(b)
            frame_no += 1
            if frame_no % 23.0 == 0:
                print(frame_no / 23.0)
        else:
            break
    out.release()
    cap.release()
    return out_filename
def str_parser (srt_filename):
    f = open(srt_filename, 'r')
    gps_data = f.read()
    gps_data_list = gps_data.splitlines()
    gps_data_seconds_strings = gps_data_list[::5]
    gps_data_seconds = [float(i)-1 for i in gps_data_seconds_strings]
    gps_data_details = gps_data_list[2::5]
    gps_longs = []
    gps_lats = []
    gps_heights = []
    for dtl in gps_data_details:
        gps_data_coordinates = dtl[dtl.find('(')+1 : dtl.find(')')]
        gps_longs.append(float(gps_data_coordinates[:gps_data_coordinates.find(',')]))
        gps_lats.append(float(gps_data_coordinates[gps_data_coordinates.find(',')+2 : gps_data_coordinates.rfind(',')]))
        gps_heights.append(float(gps_data_coordinates[gps_data_coordinates.rfind(',')+2 :]))
    gps_data_seconds_df = pd.DataFrame(data = np.array(gps_data_seconds), index = gps_data_seconds, columns = ['second'])
    gps_longs_df = pd.DataFrame(data = np.array(gps_longs), index = gps_data_seconds, columns = ['gps_long'])
    gps_lats_df = pd.DataFrame(data = np.array(gps_lats), index = gps_data_seconds, columns = ['gps_lat'])
    gps_heights_df = pd.DataFrame(data = np.array(gps_heights), index = gps_data_seconds, columns = ['gps_height'])
    gps_data_df = gps_longs_df.join(gps_lats_df)
    gps_data_df = gps_data_df.join(gps_heights_df)
    gps_data_df = gps_data_df.join(gps_data_seconds_df)
    return gps_data_df
def get_objects (video_inference_json):
    objects_list = video_inference_json['classified']
    objects_df_columns = list(objects_list[0].keys())
    objects_df_columns.append('second')
    objects_df = pd.DataFrame(columns = objects_df_columns)
    for obj in objects_list:
        obj['second'] = np.floor(obj['time_offset'] / 1000.0)
        objects_df = objects_df.append(pd.DataFrame.from_dict(obj), ignore_index = True)
        print(obj['sequence_number'])
    return objects_df
def get_export (inference_url, token):
    maximo_response = maximo_session.get(inference_url+'/export',
                                         headers = {'X-Auth-Token': token,'Content-type': 'application/json'},
                                         verify = False)
    return maximo_response
def draw_json(foldername, results_folder, filename, detections, display = False, confidence_threshold = 0.9):
    detections_in_frames = {}
    for detection in detections['classified']:
        if detection['frame_number'] not in detections_in_frames.keys():
            detections_in_frames[detection['frame_number']] = []
        detections_in_frames[detection['frame_number']].append(detection)
    cap = cv2.VideoCapture(foldername + '/' + filename)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(results_folder + '/' + str(filename[:-4])+"_drawn.mp4", fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (frame.shape[1],frame.shape[0]))
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        frame_number_as_key = str(frame_number)
        if frame_number_as_key in detections_in_frames.keys():
            detections_in_this_frame = detections_in_frames[frame_number_as_key]
            for detection in detections_in_this_frame:
                if detection['label'] == "Insulators" and detection['confidence'] > confidence_threshold:
                    xmin = detection['xmin']
                    xmax = detection['xmax']
                    ymin = detection['ymin']
                    ymax = detection['ymax']
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
                    cv2.putText(frame,str(round(detection['confidence'],2)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                #TODO: razlicite confidence thresholde za razlicite klase (mozda dictionary ime_klase: thresh)
                elif detection['label'] == "Broken insulator" and detection['confidence'] > 0.3:
                    xmin = detection['xmin']
                    xmax = detection['xmax']
                    ymin = detection['ymin']
                    ymax = detection['ymax']
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                    cv2.putText(frame,str(round(detection['confidence'],2)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        writer.write(frame)
        if display:
            cv2.imshow("frame", frame)
            k = cv2.waitKey(25) & 0xFF
            if k == 27:
                break
    cap.release()
    writer.release()
def cleanup(foldername, filename):
    filepath = str(foldername + '/' + filename)
    print("brisem: {}".format(filepath))
    if os.path.exists(filepath):
      os.remove(filepath)
    else:
      print("KAJJAZNAM BURAZ")
def inference_and_save_results(foldername, results_folder, filename):
    maximo_response_code, inference_json = detect_insulators(str(foldername + '/' + filename))
    inference_url = 'https://192.168.91.179/visual-inspection/api/inferences/' + inference_json['_id']
    maximo_response_code, video_inference_json = get_inference_output(inference_url, token)
    pbar = tqdm(total=100, desc = 'video inference')
    pbar_previous = video_inference_json['percent_complete']
    while(video_inference_json['percent_complete'] != 100.0):
        maximo_response_code, video_inference_json = get_inference_output(inference_url, token)
        pbar.update(video_inference_json['percent_complete'] - pbar_previous)
        #TODO: add an estimation of sleep so you dont spam the server with requests ;XD) -- maybe use video length as a starting point
        time.sleep(10)
        if(video_inference_json['percent_complete'] - pbar_previous < 0.05):
            pbar.write("mislim da sam se zbagao na {}".format(pbar_previous))
        pbar_previous = video_inference_json['percent_complete']
    pbar.write("gotov sam s inferencon")
    pbar.close()
    with open(results_folder + '/' + filename[:-4] + '.json', 'w') as f:
        json.dump(video_inference_json, f)
    draw_json(foldername, results_folder, filename, video_inference_json, display = False)
if __name__ == "__main__":
    #SUPRESSA SAN SVE MOGUCE WARNINGE ZIPA!
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    requests.packages.urllib3.disable_warnings()
    foldername = 'videos'
    results_folder = 'results'
    try:
        os.mkdir(results_folder)
    except:
        pass
    maximo_session = requests.Session()
    maximo_response_code, token_json = token_request()
    print ("maximo token response code =% d"% maximo_response_code)
    #print ("jsonresp:% s"% token_json)
    token = token_json['token']
    insulators_url = "https://192.168.91.179/visual-inspection/api/dlapis/b574d02f-1fb8-49f7-9ecd-1f23718d0a08"
    video_list = [file for file in os.listdir(foldername) if file.lower().endswith('.mp4')]
    for video in video_list:
        print(foldername + '/' + video)
        inference_and_save_results(foldername, results_folder, video)
        time.sleep(5)
        #cleanup(foldername, video)
    #print("infrecence done, converting to h264 now (:")
    video_list = [file for file in os.listdir(results_folder) if file.lower().endswith('.mp4')]
    for video in tqdm(video_list):
        convert_video(video, results_folder)
