"""
Created on Wed May 19 10:06:43 2021
@author: antun
"""
import requests
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
"""
from getpass import getpass
from requests.auth import HTTPBasicAuth
import certifi
import urllib3
import importlib
"""
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
                                                           'genCaption': "true"
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
#koristit ffmpeg mozda zbog filesizea
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

maximo_session = requests.Session()

maximo_response_code, token_json = token_request()
print ("maximo response code =% d"% maximo_response_code)
print ("jsonresp:% s"% token_json)
token = token_json['token']
#quit()
insulators_url = "https://192.168.91.179/visual-inspection/api/dlapis/b574d02f-1fb8-49f7-9ecd-1f23718d0a08"
dataset_import_url = "https://192.168.91.179/visual-inspection/api/datasets/import"
"""
maximo_response_code, inference_json = detect_insulators('vlcsnap-2021-05-19-10h22m47s938.png')
print ("maximo response code =% d"% maximo_response_code)
print ("jsonresp:% s"% inference_json)
"""
filename = 'DJI_0690clip.mp4'
#inference_video = video_resizing(filename)
maximo_response_code, inference_json = detect_insulators('DJI_0690clip.mp4')
print ("maximo response code =% d"% maximo_response_code)
print ("jsonresp:% s"% inference_json)

inference_url = 'https://192.168.91.179/visual-inspection/api/inferences/'+inference_json['_id']
# inference_url = 'https://192.168.91.179/visual-inspection/api/inferences/54f63be3-65a7-467f-a1b9-d632adb3f9c9'
# add completeness check

maximo_response_code, video_inference_json = get_inference_output(inference_url, token)
while(video_inference_json['percent_complete'] != 100.0):
    maximo_response_code, video_inference_json = get_inference_output(inference_url, token)
    print("inference completion: ", video_inference_json['percent_complete'])
    #TODO: add an estimation of sleep so you dont spam the server with requests ;XD)
    time.sleep(5)
print("gotov sam s inferencon fala kurcu")
quit()

maximo_response_code, video_inference_json = get_inference_output(inference_url, token)
#print ("maximo response code =% d"% maximo_response_code)
#print ("jsonresp:% s"% video_inference_json)
print("inference json keys: {}".format(video_inference_json.keys()))
print("zgotovljenost: ", video_inference_json['percent_complete'])
time.sleep(5)
print("zgotovljenost: ", video_inference_json['percent_complete'])
time.sleep(5)
maximo_response_code, video_inference_json = get_inference_output(inference_url, token)
print("zgotovljenost: ", video_inference_json['percent_complete'])
quit()
objects_df = get_objects(video_inference_json)
    
srt_filename = filename[:-4]+'.srt'
gps_data_df = str_parser(srt_filename)

objects_gps_df = objects_df.join(gps_data_df, on = 'second', how = 'left', lsuffix='_caller', rsuffix='_other')

"""
filename = 'vlcsnap-2021-05-19-10h22m47s938.png'
with open (filename, 'rb') as f:
    r = requests.post(insulators_url, files = {'files': (filename, f)}, verify = False)
r.json()

token_url = 'https://192.168.91.179/visual-inspection/api/tokens'
r = requests.post(token_url, 
                  json = {'grant_type': "password",
                          'username': "poc",
                          'password': "pcvision10!"}, 
                  verify = False)

inference_url = 'https://192.168.91.179/visual-inspection/api/inferences/'+jsonresp['_id']
headers = {'X-Auth-Token': token,'Content-type': 'application/json'}
r = requests.get(inference_url,
                 headers = headers,
                 verify = False)

inference_url = 'https://192.168.91.179/visual-inspection/api/inferences/'+jsonresp['_id']
headers = {'X-Auth-Token': token,'Content-type': 'application/json'}
r = requests.get(inference_url,
                 headers = headers,
                 json = {'id': 'fake_test_DJI_0798_FPS10.MP4'},
                 verify = False)

filename = 'DJI_0667.MP4'
f = open(filename, 'rb')
data = f.read()
print(f.readline())

filename = 'DJI_0964.MP4'
cap = cv2.VideoCapture(filename)
W = int(cap.get(3))
H = int(cap.get(4))
frame_size = (1000, int(np.ceil(1000/W*H)))
fourcc = cv2.VideoWriter_fourcc('M','P','G','4')
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('DJI_0964_1000.MP4', fourcc, fps, frame_size)
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

f = open(filename[:-4]+'.srt', 'r')
gps_data = f.read()
gps_data_list = gps_data.splitlines()
gps_data_seconds_strings = gps_data_list[::5]
gps_data_seconds = [float(i) for i in gps_data_seconds_strings]
gps_data_details = gps_data_list[2::5]
gps_longs = []
gps_lats = []
gps_heights = []
for dtl in gps_data_details:
    gps_data_coordinates = dtl[dtl.find('(')+1 : dtl.find(')')]
    gps_longs.append(float(gps_data_coordinates[:gps_data_coordinates.find(',')]))
    gps_lats.append(float(gps_data_coordinates[gps_data_coordinates.find(',')+2 : gps_data_coordinates.rfind(',')]))
    gps_heights.append(float(gps_data_coordinates[gps_data_coordinates.rfind(',')+2 :]))
gps_data_df_columns = ['second', 'gps_long', 'gps_lat', 'gps_height']
gps_longs_df = pd.DataFrame(data = np.array(gps_longs), index = gps_data_seconds, columns = ['gps_long'])
gps_lats_df = pd.DataFrame(data = np.array(gps_lats), index = gps_data_seconds, columns = ['gps_lat'])
gps_heights_df = pd.DataFrame(data = np.array(gps_heights), index = gps_data_seconds, columns = ['gps_height'])
gps_data_df = gps_longs_df.join(gps_lats_df)
gps_data_df = gps_data_df.join(gps_heights_df)

"""
broken_insulators_df = objects_gps_df[objects_gps_df['label'] == 'Broken insulator']
filename = 'DJI_0964.MP4'
cap = cv2.VideoCapture(filename)
for indx in broken_insulators_df['frame_number'].index:
    print(broken_insulators_df.loc[indx, 'frame_number'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(broken_insulators_df.loc[indx, 'frame_number']))
    ret, frame = cap.read()
    fig = plt.figure()
    plt.imshow(frame)
    scatter_x = (int(4.096*broken_insulators_df.loc[indx, 'xmax']) + int(4.096*broken_insulators_df.loc[indx, 'xmin'])) / 2
    scatter_y = (int(4.096*broken_insulators_df.loc[indx, 'ymax']) + int(4.096*broken_insulators_df.loc[indx, 'ymin'])) / 2
    plt.plot(scatter_x, scatter_y, marker = 'o', markersize = 5, color = 'r')

fig = plt.figure()
plt.plot(gps_data_df['gps_long'], gps_data_df['gps_lat'], 'k')
plt.scatter(objects_gps_df[objects_gps_df['label'] == 'Insulators']['gps_long'], objects_gps_df[objects_gps_df['label'] == 'Insulators']['gps_lat'], c = 'g')
plt.scatter(objects_gps_df[objects_gps_df['label'] == 'Broken insulator']['gps_long'], objects_gps_df[objects_gps_df['label'] == 'Broken insulator']['gps_lat'], c = 'r')