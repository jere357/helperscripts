import numpy as np
import cv2 as cv
import sys
import os
import time
import math
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
"""
function used for creating splicer subframes
"""
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
"""
helper function for map(subscreen) call
"""
def subscreen(videowriter, pixel_range, image):
    videowriter.write(image[pixel_range[1][0] : pixel_range[1][1], pixel_range[0][0] : pixel_range[0][1]])
    return
"""
lets say you slice the video into N subframes of (frame_cap) resolution name slice1,slice2..., sliceN
the stitched vides gets stitched like this: slice1->slice2->...sliceN->next frame from original video->slice1,slice2... and so on.
this order will be useful when creating the final canvas that is once again in 4k resoluiton
you have segments of N frames which are actually just 1 frame from the original video
"""
def stitch_video(filename, fourcc, frame_cap = (1000, 600), folder_name = 'stitched_videos'):
    new_video_name = folder_name+ '/' + filename[:-4] + '_stitch.mp4'
    try:
        os.mkdir(folder_name)
    except OSError as error:
        pass

    cap = cv.VideoCapture(filename)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    video_size = (int(cap.get(3)), int(cap.get(4)))
    pixel_ranges = ranges(frame_cap, video_size)
    print(new_video_name)
    videowriter = cv.VideoWriter(new_video_name, fourcc, fps, frame_cap)
    cnt=0
    t1=time.time()
    while cap.isOpened():
        cnt+=1
        ret, frame = cap.read()
        if ret==False or cnt == 240:
            break
        [videowriter.write(frame[pixel_range[1][0] : pixel_range[1][1], pixel_range[0][0] : pixel_range[0][1]]) for pixel_range in pixel_ranges]
    
    t2 = time.time()
    print("vrijeme za video 10s {} ".format(round(t2-t1,5)))
    videowriter.release()
    cap.release()
"""
this function behaves in the same way as slice_video but only applies the optical flow calculation to every frame. SLOW version atm
"""
def slice_video_optical_flow(filename, fourcc, frame_cap = (1000,600), folder_name = 'sliced_videos', time_skip = 0, duration = 7):
    print( folder_name + '/' + filename[:-4])
    try:
        os.mkdir(folder_name)
    except OSError as error:
        pass
        #print(str(error) + " sliced videos folder already exists - irrelevant error")
    try:
        os.mkdir(folder_name + '/' + filename[:-4])
    except OSError as error:
        pass
        #print(str(error) + " video named subfolder already exists - irrelevant error")
    cap = cv.VideoCapture(filename)
    fps = int(cap.get(cv.CAP_PROP_FPS))
    print(fps)
    video_size = (int(cap.get(3)), int(cap.get(4)))
    cnt=0
    pixel_ranges = ranges(frame_cap, video_size)
    #sporija verzija
    for counter, pixel_range in enumerate(pixel_ranges):
        print("obradujem: {}".format(pixel_range))
        cnt=0
        if math.floor(counter/10) == 0:
            new_video_name = folder_name + '/' + filename[:-4] + '/' + filename[:-4] + 'SLICE0' + str(counter) + '.avi'
        else:
            new_video_name = folder_name + '/' + filename[:-4] + '/' + filename[:-4] + 'SLICE' + str(counter) + '.avi'
        print("new video name je: {}".format(new_video_name))
        #real frame size bi triba bit isti ka ovi u pixel rangeu !?
        out = cv.VideoWriter(new_video_name, fourcc, fps, frame_cap)
        cap = cv.VideoCapture(filename)
        cap.set(1, time_skip*fps)
    	
        ret, frame = cap.read()
        small_frame_zero = frame[pixel_range[1][0] : pixel_range[1][1],pixel_range[0][0] : pixel_range[0][1]]
        
        old_gray = cv.cvtColor(small_frame_zero, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(small_frame_zero)
        hsv[...,1] = 255

        t1 = time.time()

        font = cv.FONT_HERSHEY_SCRIPT_COMPLEX
        video_time=1.0

        while cap.isOpened():
            cnt+=1
            if cnt == fps*duration:
                break
            ret, frame = cap.read()
            small_frame = frame[pixel_range[1][0] : pixel_range[1][1],pixel_range[0][0] : pixel_range[0][1]]
            frame_gray = cv.cvtColor(small_frame, cv.COLOR_BGR2GRAY)
            # Calculate Optical Flow
            flow = cv.calcOpticalFlowFarneback(prev = old_gray,
                                                next = frame_gray, 
                                                flow = None, 
                                                pyr_scale = 0.1, 
                                                levels = 3, 
                                                winsize = 15, 
                                                iterations = 7, 
                                                poly_n = 7, 
                                                poly_sigma = 1.5, 
                                                flags = 0)
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
            # Display the demo
            img = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            img = cv.putText(np.minimum((0.05*img**1.5+1.0*small_frame), 255*(np.ones(small_frame.shape))).astype('uint8'), 
                            str(video_time/23.0), 
                            (10, 100), 
                            font, 
                            1,
                            (210, 155, 155), 
                            4, 
                            cv.LINE_8)

            out.write(img)
            cv.imshow('frame', img)

            if cv.waitKey(1) == ord('q'):
                break
            old_gray = frame_gray.copy()
            video_time+=1
        
        t2 = time.time()
        print(str(round(t2-t1)) + "s for 1 slice")
        cap.release()
        out.release()
        cv.destroyAllWindows()
"""
filename str - path to video you want to slice
frame_cap (int,int) - maximum size of window 
folder name str - name of the subfolder where the sliced videos are stored  folder_name/filenameXX.avi where X goes from 0 to X
"""
def slice_video(filename, fourcc, frame_cap = (1000,600), folder_name = 'sliced_videos', time_skip = 0, duration = 0):
    video_folder_name = folder_name+ '/' + filename[:-4] + '_' + str(time_skip) + str(duration)
    try:
        os.mkdir(folder_name)
    except OSError as error:
        pass
        #print(str(error) + " sliced videos folder already exists - irrelevant error")
    try:
        os.mkdir(video_folder_name)
    except OSError as error:
        pass
        #print(str(error) + " video named subfolder already exists - irrelevant error")
    cap = cv.VideoCapture(filename)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv.CAP_PROP_FPS))
    video_size = (int(cap.get(3)), int(cap.get(4)))
    pixel_ranges = ranges(frame_cap, video_size)
    cap.release()
    videowriters = []
    for counter, pixel_range in enumerate(pixel_ranges):
        if math.floor(counter/10) == 0:
            new_video_name = video_folder_name + '/' + filename[:-4] + 'SLICE0' + str(counter) + '_' + str(time_skip) + str(duration) + '.mp4'
        else:
            new_video_name = video_folder_name + '/' + filename[:-4] + 'SLICE' + str(counter) + '_' + str(time_skip) + str(duration) + '.mp4'
        videowriters.append(cv.VideoWriter(new_video_name, fourcc, fps, frame_cap))
    cap = cv.VideoCapture(filename)
    cap.set(1, time_skip*fps)
    t1 = time.time()
    cnt=0
    while cap.isOpened():
        cnt+=1
        if cnt == fps*duration and duration!=0:
            break 
        ret, frame = cap.read()
        if(ret == False):
            break
        #t3=time.time()
        list(map(subscreen, videowriters, pixel_ranges, repeat(frame, len(pixel_ranges))))
        #t4=time.time()
        #print("vrijeme za jedan frejm {}".format(round(t4-t3,5)))
    t2 = time.time()
    print(str(round(t2-t1)) + "s for video")
    [vw.release() for vw in videowriters]
    cap.release()
    cv.destroyAllWindows()
if __name__ == '__main__':
    folder_name = 'sliced_videos'
    filename = sys.argv[1]
    frame_cap = (1000,600)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    stitch_video(filename, fourcc, frame_cap)
"""
    slice_video(filename, fourcc, frame_cap, time_skip = 59, duration = 11)
    slice_video(filename, fourcc, frame_cap, time_skip = 75, duration = 10)
    slice_video(filename, fourcc, frame_cap, time_skip = 123, duration = 8)
    slice_video(filename, fourcc, frame_cap, time_skip = 148, duration = 13)
"""