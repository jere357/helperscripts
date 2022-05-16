from hmac import new
import albumentations as A
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.augmentations.transforms import Equalize, VerticalFlip
from albumentations.core.composition import OneOf
import cv2
import glob
#from albumentations.pytorch.transforms import ToTensorV2
import xml.etree.ElementTree as ET
import secrets
from tqdm import tqdm
#TODO: copypaste augmentaciju implementirat xd
"""
load only images that HAVE bbox files
"""



def draw_bboxes(image, bbox_list):
    for bbox in bbox_list:
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color = (255,0,0), thickness = 3)

def load_images(foldername, image_format = '.jpg'):
    image_filenames = []
    bbox_filenames = glob.glob(str(foldername) + "/*.xml")
    #bbox_list is a list. each element of that list contains a list which has all the bounding boxes found on one photo (xmin, ymin, xmax, ymax)
    bbox_list = []
    image_list = []
    size_list = []
    #put all image filenames in a list, insuring proper order of filenames
    [image_filenames.append(name[:-4] + image_format) for name in bbox_filenames]
    for bbox, image in zip(bbox_filenames, image_filenames):
        tree = ET.parse(bbox)
        image_list.append(cv2.imread(image))
        temp_bbox_list = []
        size_list.append((int(tree.getroot()[0][0].text),int(tree.getroot()[0][1].text), int(tree.getroot()[0][2].text)))
        for child in tree.getroot()[1:]:
            temp_bbox_list.append([int(child[2][0].text), int(child[2][1].text), int(child[2][2].text), int(child[2][3].text)])
        bbox_list.append(temp_bbox_list)
    return image_list, bbox_list, size_list
def write_xml_file(bbox_list, size, foldername, new_name):
    root = ET.Element("annotation")
    m1 = ET.Element("size")
    root.append (m1)
    b1 = ET.SubElement(m1, "width")
    b1.text = str(size[0])
    b2 = ET.SubElement(m1, "height")
    b2.text =  str(size[1])
    b2 = ET.SubElement(m1, "depth")
    b2.text = str(size[2])
    #mid XMLA    
    for bbox in bbox_list:
        m2 = ET.Element("object")
        root.append(m2)
        c1 = ET.SubElement(m2, "_id")
        c1.text = " "
        c2 = ET.SubElement(m2, "name")
        c2.text = str(bbox[4])

        c3 = ET.Element("bndbox")
        m2.append(c3)
        c4 = ET.SubElement(c3, "xmin")
        c4.text = str(int(bbox[0]))
        c5 = ET.SubElement(c3, "ymin")
        c5.text = str(int(bbox[1]))
        c6 = ET.SubElement(c3, "xmax")
        c6.text = str(int(bbox[2]))
        c7 = ET.SubElement(c3, "ymax")
        c7.text = str(int(bbox[3]))

        c8 = ET.SubElement(m2, "generate_type")
        c8.text = "manual"
        c9 = ET.SubElement(m2, "file_id")
        c9.text = " "
    tree = ET.ElementTree(root)
    with open (str(foldername + '/' + new_name + '.xml'), "wb") as files :
        tree.write(files)

    pass
def write_augmented_image_to_disk(image, bboxes, size, foldername):
    #convert all list elements to int - bboxes = [coord1, coord2, coord3, coord4, label] + separate label list
    new_name = secrets.token_urlsafe(20)
    #STVORI XML FAJLU
    write_xml_file(bboxes, size, foldername, new_name)
    cv2.imwrite(str(foldername + '/' + new_name + '.png'), image)
def augment_dataset(foldername, num_epochs=5):
    transform = A.Compose([
        A.OneOf([
            A.ShiftScaleRotate(p=0.2),
            A.Rotate(p=0.6),
            A.VerticalFlip(p=0.2),
            A.HorizontalFlip(p=0.4),
        ], p=0.8),
        A.OneOf([
            A.CLAHE(clip_limit=3,p=0.3),
            A.Equalize(p=0.6),
            A.RandomBrightnessContrast(p=0.3),            
        ], p=0.6),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.25),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.15),
            A.Blur(blur_limit=3, p=0.15),
        ], p=0.6),
        A.Affine(p=0.05),
    ], bbox_params=A.BboxParams(format='pascal_voc'))
    image_list, bbox_list, size_list = load_images(foldername)
    for _ in tqdm(range(num_epochs)):
        for image, bboxes, size in zip(image_list, bbox_list, size_list):
            #append a class label for every bbox bcs thats how albumentations works
            [element.append("Insulator") for element in bboxes]
            transformed = transform(image = image, bboxes = bboxes)
            draw_bboxes(transformed['image'], transformed['bboxes'])
            write_augmented_image_to_disk(transformed['image'], transformed['bboxes'], size, foldername)
            #cv2.imshow("title", cv2.resize(transformed['image'], (1000, 600)))
            #cv2.waitKey()
if __name__ == "__main__":
    augment_dataset(foldername = 'dataset')