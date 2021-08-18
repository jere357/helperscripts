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
import math
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
def get_inference_output (inference_url, token):
    maximo_response = maximo_session.get(inference_url,
                                         headers = {'X-Auth-Token': token,'Content-type': 'application/json'},
                                         verify = False)
    return maximo_response.status_code, json.loads(maximo_response.text)

def draw_json(foldername, results_folder, filename, detections, display = True, confidence_threshold = 0.7):
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
                if detection['label'] == "Insulator" and detection['confidence'] > confidence_threshold:
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

def inference_and_save_results(foldername, results_folder, filename, insulators_url):
    maximo_response_code, inference_json = detect_insulators(str(foldername + '/' + filename))
    inference_url = 'https://192.168.91.179/visual-inspection/api/inferences/' + inference_json['_id']
    maximo_response_code, video_inference_json = get_inference_output(inference_url, token)
    pbar = tqdm(total=100, desc = 'video inference')
    pbar_previous = video_inference_json['percent_complete']
    while(video_inference_json['percent_complete'] != 100.0):
        maximo_response_code, video_inference_json = get_inference_output(inference_url, token)
        pbar.update(math.floor(video_inference_json['percent_complete'] - pbar_previous))
        #TODO: add an estimation of sleep so you dont spam the server with requests ;XD) -- maybe use video length as a starting point
        time.sleep(15)
        if(video_inference_json['percent_complete'] - pbar_previous < 0.05):
            pbar.write("mislim da sam se zbagao na {}".format(pbar_previous))
        pbar_previous = video_inference_json['percent_complete']
    #pbar.write("gotov sam s inferencon")
    pbar.close()
    with open(results_folder + '/' + filename[:-4] + '.json', 'w') as f:
        json.dump(video_inference_json, f)
    draw_json(foldername, results_folder, filename, video_inference_json, display = False)
def ultimate_insulator_inference_function(foldername = "videos", results_folder = "results"):
    #results folder duhh
    video_list = [file for file in os.listdir(foldername) if file.lower().endswith('.mp4')]
    try:
        os.mkdir(results_folder)
    except:
        pass
    for video in video_list:
        #print(foldername + '/' + video)
        inference_and_save_results(foldername, results_folder, video, insulators_url)
        time.sleep(2)
    print("inference done, converting to h264 now (:")
    video_list_for_conversion = [file for file in os.listdir(results_folder) if file.lower().endswith('.mp4')]
    for video in tqdm(video_list_for_conversion):
        convert_video(video, results_folder)
    pass

if __name__ == "__main__":
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    requests.packages.urllib3.disable_warnings()
    foldername = "videos"
    #MAKE SURE YOU HAVE THE CORRECT MODEL URL
    insulators_url = "https://192.168.91.179/visual-inspection/api/dlapis/d953877b-cb08-467e-9ab6-19366e9e897f"
    maximo_session = requests.Session()
    maximo_response_code, token_json = token_request()
    print ("maximo token response code =% d"% maximo_response_code)
    token = token_json['token']
    t1 = time.time()
    ultimate_insulator_inference_function()
    t2 = time.time()
    print(f"vrijeme potrebno za sve: {round(t2-t1,2)}")