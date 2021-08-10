import os
import random
import cv2
import math
if __name__ =="__main__":
    broken_insulators_folder = "./samples/"
    images_folder = "./pictures/"
    augmented_folder = "./augmented/"
    scales = [1, 0.66, 0.33]
    patch_path = broken_insulators_folder + str(random.choice(os.listdir(broken_insulators_folder)))
    image_path = images_folder + str(random.choice(os.listdir(images_folder)))
    print(image_path)
    patch = cv2.imread(patch_path)
    image = cv2.imread(image_path)
    print(image.shape)
    print(patch.shape)
    print('\n')
    #generation of random new coordinates where the patch will be pasted i think this will work every time (it shouldnt crash bcs out of range indexing later), idk 
    coordinates = (random.randint(100, image.shape[0]- (patch.shape[0] + 100)), random.randint(100, image.shape[1] - (patch.shape[1] + 100)))
    print(coordinates)
    for scale in scales:
        coordinates = (random.randint(100, image.shape[0]- (patch.shape[0] + 100)), random.randint(100, image.shape[1] - (patch.shape[1] + 100)))
        patch_temp = cv2.resize(patch, ((math.floor(patch.shape[0] * scale)), math.floor(patch.shape[1] * scale)))
        print(patch_temp.shape)
        image[coordinates[0]:coordinates[0]+patch_temp.shape[0] , coordinates[1]:coordinates[1]+patch_temp.shape[1]] = patch_temp
    image = cv2.resize(image, ((1920,1080)))
    cv2.imwrite("./augmented/slikaxd.png", image)
    cv2.imshow("image", image)
    cv2.waitKey()
    
