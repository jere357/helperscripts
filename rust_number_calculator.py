import glob
import json
import os
from typing import final
from PIL.Image import new
import numpy as np
import math
import time
from tqdm import tqdm
"""
quantization parameter determines how many "consecutive" ( they ARE necessarily consecutive :) ) frames should we bunch up into one sum
relevant_color_columns = 6 7 8 - dalmatia default
                         2 3 4 5 6 7 8 - slavonia 
"""
def calculate_rust_tower_numbers(data, quantization = 250, zone = "Slavonija" , total_columns = 12):
    keys_list = [int(i) for i in data.keys()]
    keys_list.sort()
    final_arrays = []
    corrosion_number_list = []
    tower_number_list = []
    for quant in range(math.ceil(max(keys_list)/quantization)):
        lower_limit = quant * quantization
        upper_limit = (quant + 1) * quantization
        big_frame_summed = np.zeros(total_columns)
        for key in keys_list:
            if key < lower_limit:
                continue
            if key > upper_limit:
                break
            frame_array = np.array(list(data[str(key)].values())[0])
            #print(f" key mi je : {key} frame array")
            #print(frame_array)
            big_frame_summed += np.sum(frame_array, axis = 0)
        #print("big frame")
        #print(big_frame_summed)
        final_arrays.append(big_frame_summed)
    for i in range(len(final_arrays)):
        corrosion_number_list.append(final_arrays[i][0] + final_arrays[i][1]) #CORROSION NUMBER = 1. + 2. STUPAC
        if zone == "Slavonija":
            tower_number_list.append(final_arrays[i][0] + final_arrays[i][1] + 
                final_arrays[i][2] + final_arrays[i][3] + final_arrays[i][4] + 
                final_arrays[i][5] + final_arrays[i][6] + final_arrays[i][7] +
                final_arrays[i][8]) #TOWER NUMBER = 1 + 2 + OSTALI RELEVANTNI COLOR STUPCI
        elif zone == "Dalmacija":
            tower_number_list.append(final_arrays[i][0] + final_arrays[i][1] + 
            final_arrays[i][6] + final_arrays[i][7] + final_arrays[i][8])
    rust_precentages_list = np.divide(np.array(corrosion_number_list), np.array(tower_number_list))
    #rust_precentages_list = np.array(corrosion_number_list) / np.array(tower_number_list)
    rust_precentages_list = np.nan_to_num(rust_precentages_list, nan = 0.0)
    #print(np.array(corrosion_number_list))
    #print(np.array(tower_number_list))
    #print(rust_precentages_list)
    #print("jebenti mater")
    return rust_precentages_list, quantization
if __name__ == "__main__":
    foldername = "dalmacija_results"
    json_files = [pos_json for pos_json in os.listdir(foldername) if pos_json.endswith('rust-id.json')]
    json_files.sort()
    for filename in tqdm(json_files):
        json_data = {}
        #print(filename)
        with open(foldername + '/' + filename) as f:
            data = json.load(f)
        final_array_entire_video, quantization = calculate_rust_tower_numbers(data, quantization = 1000000, zone = "Dalmacija")
        final_array, quantization = calculate_rust_tower_numbers(data, zone = "Dalmacija")
        #print(filename)
        #print(final_array)
        #print(final_array_entire_video)
        json_data['original_filename'] = filename
        json_data['rust_numbers'] = list(final_array)
        json_data['quantization'] = quantization
        json_data['average_video_rust'] = list(final_array_entire_video)
        new_filename = filename[:filename.find("-id")] + "-numbers.json"
        #print(new_filename)
        with open('dalmacija_numbers/' + new_filename, 'w+') as outfile:
            json.dump(json_data, outfile)
    #odredi postotak hrdavih piksela s obziron na stup piksele
    #napravi dictionary sa frame : postotak hrđimaule
