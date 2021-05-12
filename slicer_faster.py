import numpy as np
import cv2 as cv
import sys
import os
import time
import math
def ranges(frame_cap, video_size):
    ranges_lists = []
    
    width_split = math.ceil(video_size[0]/(frame_cap[0] - 100))
    width_padding = math.ceil((width_split*frame_cap[0] - video_size[0]) / (width_split - 1))
    
    height_split = math.ceil(video_size[1]/(frame_cap[1] - 100))
    height_padding = math.ceil((height_split*frame_cap[1] - video_size[1]) / (height_split - 1))
    
    next_width_start = 0
    for i in range(width_split):
        next_height_start = 0
        for j in range(height_split):
            ranges_lists.append( [(next_width_start, next_width_start + frame_cap[0]), 
                                  (next_height_start, next_height_start + frame_cap[1])]
                                )
            next_height_start += frame_cap[1] - height_padding
        next_width_start += frame_cap[0] - width_padding
    return ranges_lists

#small window format is ((a,b),(c,d)) a,b - height, c,d - width, a<b,c<d
def pad_window(small_window, video_size, padding):
    #pad heights
    a = small_window[0][0]
    b = small_window[0][1]
    if(a-padding > 0):
        a-=padding
    if(b+padding < video_size[1]):
        b+=padding
    #pad widths
    c = small_window[1][0]
    d = small_window[1][1]
    if (c-padding > 0):
        c-=padding
    if (d + padding < video_size[0]):
        d+=padding
    #print("window prije: {}, window posli: {}".format(small_window, ((a,b),(c,d))))
    return ((a,b),(c,d))
    pass
"""
filename str - path to video you want to slice
frame_cap (int,int) - maximum size of window 
folder name str - name of the subfolder where the sliced videos are stored  folder_name/filenameXX.avi where X goes from 0 to X

cvSetCaptureProperty

"""
#zelin da frame_cap bude MANJI od stvarnog frame capa da se ne gubi rezolucija kada paddas + resizean
def slice_video(filename, frame_cap = (1000,600), folder_name = 'sliced_videos', time_skip = 36, duration = 0):
    print( folder_name + '/' + filename[:-4])
    try:
        os.mkdir(folder_name)
    except OSError as error:
        print(str(error) + " sliced videos folder already exists - irrelevant error")
    try:
        os.mkdir(folder_name + '/' + filename[:-4])
    except OSError as error:
        print(str(error) + " video named subfolder already exists - irrelevant error")
    cap = cv.VideoCapture(filename)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv.CAP_PROP_FPS))
    video_size = (int(cap.get(3)), int(cap.get(4)))
    pixel_ranges = ranges(frame_cap, video_size)
    cap.release()
    #sporija verzija
    #stvori sve videowriter objekte
    videowriters = []
    for counter, pixel_range in enumerate(pixel_ranges):
        if math.floor(counter/10) == 0:
            new_video_name = folder_name + '/' + filename[:-4] + '/' + filename[:-4] + 'SLICE0' + str(counter) + str(time_skip + duration) + '.avi'
        else:
            new_video_name = folder_name + '/' + filename[:-4] + '/' + filename[:-4] + 'SLICE' + str(counter) + str(time_skip + duration) + '.avi'
        videowriters.append(cv.VideoWriter(new_video_name, fourcc, fps, frame_cap))
    cap = cv.VideoCapture(filename)
    cap.set(1, time_skip*fps)
    t1 = time.time()
    #loopaj po writer objektima i tako znas
    cnt=0
    while cap.isOpened():
        cnt+=1
        if cnt == fps*duration and duration!=0:
            break 
        ret, frame = cap.read()
        if(ret == False):
            break
        for videowriter_objekt, pixel_range in zip(videowriters, pixel_ranges):
            small_frame = frame[pixel_range[1][0] : pixel_range[1][1],pixel_range[0][0] : pixel_range[0][1]]    
            videowriter_objekt.write(small_frame)
            #cv.imshow('frame', small_frame)
            #if cv.waitKey(1) == ord('q'):
            #    break
    t2 = time.time()
    print(str(round(t2-t1)) + "s for video")
    for videowriter_objekt in videowriters:
        videowriter_objekt.release()
    cap.release()
    cv.destroyAllWindows()

folder_name = 'sliced_videos'
filename = sys.argv[1]
frame_cap = (1000,600)
slice_video(filename, frame_cap, time_skip = 14, duration = 7)