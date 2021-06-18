import os
import sys
from tqdm import tqdm
import subprocess
def downscale_video(filename, folder_name):
    video_folder_name = folder_name + '/' + folder_name + '_lowres'
    try:
        os.mkdir(folder_name)
    except OSError as error:
        pass
        #print(str(error) + " folder already exists - irrelevant error")
    try:
        os.mkdir(video_folder_name)
    except OSError as error:
        #folder alrdy exists - WHO CARES 
        pass
    command =  "ffmpeg -i " + folder_name + '/' + filename + " -filter:v scale=1000:-2 -c:v libx264 -crf 24 " + video_folder_name + '/' + filename[:-4] + '_lowres' + '.mp4'
    #print(filename)
    #print(command)
    a = subprocess.call(command, shell = True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)

if __name__ == "__main__":
    #you have to pass the folder_name without / at the end; python3 ffmpegconverter.py myfolder/subfolder 
    folder = sys.argv[1]
    video_array = []
    [video_array.append(x) for x in os.listdir(folder) if x.lower().endswith(".mp4")]
    for video in tqdm(video_array):
        downscale_video(video, folder_name=folder)
