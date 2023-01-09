from email.mime import image
import os
import logging
import cv2
import pandas as pd
import json
import os
from collections import defaultdict
from math import sqrt, atan
from tqdm import tqdm
from xml.etree import ElementTree as ET
from sklearn.model_selection import train_test_split

def distance_between_two_points(p1, p2, h , w):
    return sqrt((w*p1[0] - w*p2[0]) ** 2 + (h*p1[1] - h*p2[1]) ** 2)

def extract_images_with_4point_polygons(images):
    polygon_count = defaultdict(lambda: [])
    four_point_polygon_images = defaultdict(lambda: [])
    for image_name in images.keys():
        #print(images[image_name])
        polygon_dims = []
        for object in images[image_name]:
            polygon_dims.append(len(object[0]))
            #print(f"broj tocaka u poligonu mi je {len(object[0])} a njegov id je {object[1]}")
        polygon_count[image_name] = polygon_dims
    for image_name in polygon_count.keys():
        #print(f"slika {image_name} ima poligone : {polygon_count[image_name]}")
        if polygon_count[image_name].count(4) == len(polygon_count[image_name]):
            four_point_polygon_images[image_name] = images[image_name]
    return four_point_polygon_images

def generate_trainval_files(foldername = 'labelXml'):
    for root, dirs, files in os.walk(foldername):
        #print(files)
        print(f"ukupno iman {len(files)} labeliranih slika")
        trainval, test = train_test_split(files, test_size = 0.1)
        train, val = train_test_split(trainval, test_size = 0.15)
        with open(os.path.join("ImageSets", "train.txt"), 'w') as f:
            for line in train:
                f.write(f"{line[:-4]}\n")
        with open(os.path.join("ImageSets", "test.txt"), 'w') as f:
            for line in test:
                f.write(f"{line[:-4]}\n")
        with open(os.path.join("ImageSets", "trainval.txt"), 'w') as f:
            for line in trainval:
                f.write(f"{line[:-4]}\n")
        with open(os.path.join("ImageSets", "val.txt"), 'w') as f:
            for line in val:
                f.write(f"{line[:-4]}\n")


def write_xml_annotations(polygons, ID, width, height, labels_foldername = "labelXml"):
    anno_tree = ET.parse("empty.xml")
    anno_root = anno_tree.getroot()
    objekti = anno_root.find("HRSC_Objects")
    ID_element = anno_root.find("Img_ID")
    ID_element.text = str(ID)
    width_element = anno_root.find("Img_SizeWidth")
    width_element.text = str(width)
    height_element = anno_root.find("Img_SizeHeight")
    height_element.text = str(height)
    for poligon in polygons:
        hrsc_objekt = ET.SubElement(objekti, 'HRSC_Object')
        e = ET.SubElement(hrsc_objekt, 'difficult')
        e.text = "0"
        e.tail = "\n"
        e = ET.SubElement(hrsc_objekt, 'mbox_cx')
        e.text = poligon[0]
        e.tail = "\n"
        e = ET.SubElement(hrsc_objekt, 'mbox_cy')
        e.text = poligon[1]
        e.tail = "\n"
        e = ET.SubElement(hrsc_objekt, 'mbox_w')
        e.text = poligon[2]
        e.tail = "\n"
        e = ET.SubElement(hrsc_objekt, 'mbox_h')
        e.text = poligon[3]
        e.tail = "\n"
        e = ET.SubElement(hrsc_objekt, 'mbox_ang')
        e.text = poligon[4]
        e.tail = "\n"
        hrsc_objekt.tail = "\n"

    path = os.path.join(os.getcwd(), labels_foldername, f"{ID}.xml")
    #Path(str(path)).touch()
    anno_tree.write(path)
"""
p1 is the left upper point of the polygon 
p2 is the upper right point of the polygon
exactly how to determine the upper left and right points in the polygon is left as an exercise to the reader:))
remember that coordinate systems in images start from the top left 
"""
def determine_sign_of_angle(p1, p2):
    if p1[1] > p2[1]:
        return -1
    elif p2[1] >= p1[1]:
        return 1


"""
p1 is the left upper point of the polygon 
p2 is the upper right point of the polygon
remember that coordinate systems in images start from the "top left"
1e-6 to avoid division by 0 in some very rare edge cases :v
"""
def calculate_angle(p1,p2,h,w):
    return atan(((p2[1]-p1[1]) * h) / ((p2[0]-p1[0]) * w + 1e-6))

"""
this function is required bcs the annotations are inconsistent - sometimes the first point is the upper left but sometimes not
this function hopes to standardise the polygons point and the order theyre in
"""



#TODO: ako poligon ima 5 tocaka idk bmk odjebi ga skroz
#CVAT annotations were done by me in a consitent order: top left->bottom left->bottom right->top right
def sort_polygon_points_CVAT(points):
    sorted_points = []
    temp_points = points.split(";")
    for point_pair in temp_points:
        x = round(float(point_pair.split(',')[0]))
        y = round(float(point_pair.split(',')[1]))
        sorted_points.append((x,y))
    print(sorted_points)
    return sorted_points

#mozda drugaciji sort polygon points koji to malo bolje radi ovoga sjebu mali poligoni koji su blizu ishodista zna bit inkonzistentno
def sort_polygon_points(points, h, w):
    final_polygon_sorted = []
    point_distances = {}
    other_point_distances = {}
    for point in points:
        point_distances[tuple(point)] = distance_between_two_points((0,0), point, h, w)
    points_sorted = sorted(point_distances.items(), key=lambda item: item[1])
    top_left = points_sorted[0][0]
    #other points are the 3 remaining points that arent the top left :)
    for other_point in points_sorted[1:]:
        other_point_distances[tuple(other_point[0])] = distance_between_two_points(top_left, other_point[0], h, w)
    other_points_sorted = sorted(other_point_distances.items(), key=lambda item: item[1])
    final_polygon_sorted.append(top_left)
    [final_polygon_sorted.append(other_point[0]) for other_point in other_points_sorted]
    return final_polygon_sorted
    return [point[0] for point in points_sorted]



def draw_first_three_rectangles(img, polygon):
    cv2.rectangle(img, (int(polygon[0][0] * width), int(polygon[0][1] * height)),
                  (int(polygon[0][0] * width) + 15, int(polygon[0][1] * height) + 15),
                  (255, 255, 255), 4)
    cv2.rectangle(img, (int(polygon[1][0] * width), int(polygon[1][1] * height)),
                  (int(polygon[1][0] * width) + 15, int(polygon[1][1] * height) + 15),
                  (255, 0, 255), 4)
    cv2.rectangle(img, (int(polygon[2][0] * width), int(polygon[2][1] * height)),
                  (int(polygon[2][0] * width) + 15, int(polygon[2][1] * height) + 15),
                  (0, 255, 255), 4)
    pass



counter = 0
images_foldername = "data"
annotations = ET.parse('annotations.xml')
root = annotations.getroot()
for child in root:
    print(child.tag, child.attrib)
    if child.tag == "version" or child.tag == "meta":
        continue
    image_info = child.attrib
    image_name = image_info['name']
    image_width = image_info['width']
    image_height = image_info['height']
    img = cv2.imread(f'data/{image_name}')
    for polygon in child:
        points = polygon.attrib['points']
        points_sorted = sort_polygon_points_CVAT(points)
        cx = int(sum([point[0] for point in points_sorted])/4)
        cy = int(sum([point[1] for point in points_sorted])/4)
        ch = int(sqrt(pow(points_sorted[0][0] - points_sorted[1][0], 2) + pow(points_sorted[0][1] - points_sorted[1][1], 2)))
        cw = int(sqrt(pow(points_sorted[1][0] - points_sorted[2][0], 2) + pow(points_sorted[1][1] - points_sorted[2][1], 2)))
        #TODO: kut je nagib izmedu prve i trece tocke
        print(f"Cx, cy = ({cx}, {cy}) cw: {cw} ch:{ch}")
        print(polygon.tag, polygon.attrib)
display = False
exit()
images = defaultdict(lambda: [])
product_id_dict = defaultdict(lambda: [])
for idx, row in train_file.iterrows():
    polygons = []
    if row['ProductId'] != 1:
        #NIJE POLIGON NEGO NEKA USELESS BACKGROUND? KLASA IDK IDC
        continue
    polygons.append(json.loads(row['Polygon']))
    images[row['ImageUrl'].split('/')[-1]].append((json.loads(row['Polygon']), row['ProductId']))
    product_id_dict[row['ImageUrl'].split('/')[-1]].append(row['ProductId'])
four_point_polygon_images = extract_images_with_4point_polygons(images)
print(f"broj slika u images je {len(images.keys())}")
print(f"broj slika u four points polygons je : {len(four_point_polygon_images.keys())}")
#28 je problematican index remember
for image_name in tqdm(list(four_point_polygon_images.keys())[71:]):
    #531 je "zanimljiva za prezu amo rec"
    images_list = list(four_point_polygon_images.keys())
    images_set = set(four_point_polygon_images.keys())
    xml_annotations = []
    img = cv2.imread(f'train_both/{image_name}')
    if img is None:
        #TODO:samo ga makni iz dataseta jebe me se za 20 slika sta nije skinia
        print(f"slika {image_name} not found on disk idk what yall doin")
        continue
    height, width, c = img.shape

    #print(f'slika:{image_name} dimenzija {width},{height}')
    #something_weird refers to width being smaller than height so we just skip those images ll together since theyre a really really small part of the dataest cba
    something_weird_happened = False
    for object in images[image_name]:
        polygon = object[0]
        polygon_unsorted = object[0]
        #points gotta be sorted bcs sometimes the choice of the first point in the polygon is not consistent by the annotators
        polygon = sort_polygon_points(polygon, height, width)
        object_id = object[1]
        #imaju neke doadatne nebitne klase koje nisu police idk
        color = (255,0,0)
        #A i B decision making dio
        a = distance_between_two_points(polygon[0], polygon[2], height, width)
        b = distance_between_two_points(polygon[2], polygon[3], height, width)
        if display:
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if a < b:
            counter += 1
            something_weird_happened = True
            #tqdm.write("nesto cudno hm")
            """
            cv2.resize(img, (600,400))
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """
        cx = int(width * (polygon[0][0] + polygon[1][0] + polygon[2][0] + polygon[3][0]) / 4)
        cy = int(height * (polygon[1][1] + polygon[3][1] + polygon[2][1] + polygon[0][1]) / 4)
        #(cx,cy) krug nacrtaj idk jebe me se
        if display:
            cv2.circle(img,
                   (cx, cy),
                   5,
                   (255, 0, 255),
                   thickness=7)
        #draw_first_three_rectangles(img, polygon)
        cw = max(a, b)
        ch = min(a, b)
        angle = calculate_angle(polygon[0], polygon[2], height, width)

        """
        zero_crtano = (polygon[0][0] + cw / width, polygon[0][1])  # THIS variable is also in the fucked up format that is [0,1]*dimension instead of just a point on the image
        nasuprotna = distance_between_two_points(polygon[2], zero_crtano, height, width)
        kateta = distance_between_two_points(polygon[0], polygon[2], height, width)
        if nasuprotna > kateta:
            something_weird_happened = True
            counter += 1
            tqdm.write("sjebana slika idc uopce zasto bmk go next")
            continue
        if display:
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
        angle = asin(distance_between_two_points(polygon[2], zero_crtano, height, width) /
                     distance_between_two_points(polygon[0], polygon[2], height, width))
        angle_sign = determine_sign_of_angle(polygon[0], polygon[2])
        angle *= angle_sign

        if b > a:
            draw_first_three_rectangles(img, polygon)
            cw = max(a,b)
            ch = min(a,b)
            #TODO:sta ako 0 nije gori livo na slici :))))))))) - onda eta gege idk brate
            zero_crtano = (polygon[0][0] + cw/width, polygon[0][1]) #THIS variable is also in the fucked up format that is [0,1]*dimension instead of just a point on the image
            nasuprotna = distance_between_two_points(polygon[3], zero_crtano, height, width)
            kateta = distance_between_two_points(polygon[0], polygon[3], height, width)
            if nasuprotna > kateta:
                print("sjebana slika idc uopce zasto bmk go next")
                continue

            cv2.imshow(image_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            angle = asin(distance_between_two_points(polygon[3], zero_crtano, height, width) /
                      distance_between_two_points(polygon[0], polygon[3], height, width))
            angle_sign = determine_sign_of_angle(polygon[0], polygon[3])
            angle *= angle_sign
        elif a > b:
            draw_first_three_rectangles(img, polygon)
            cw = max(a,b)
            ch = min(a,b)
            zero_crtano = (polygon[0][0] + cw/width, polygon[0][1]) #THIS variable is also in the fucked up format that is [0,1]*dimension instead of just a point on the image
            nasuprotna = distance_between_two_points(polygon[1], zero_crtano, height, width)
            kateta = distance_between_two_points(polygon[1], polygon[0], height, width)
            if nasuprotna > kateta:
                print("sjebana slika idc uopce zasto bmk go next")
                continue
            cv2.imshow(image_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            angle = asin(distance_between_two_points(polygon[1], zero_crtano, height, width) /
                      distance_between_two_points(polygon[0], polygon[1], height, width))
            angle_sign = determine_sign_of_angle(polygon[0], polygon[1])
            angle *= angle_sign
        """
        for i in range(len(polygon)):
            if i == 0:
                cv2.circle(img,
                              (int(width * polygon[0][0]), int(height * polygon[0][1])),
                              5,
                              (255,255,0),
                           thickness=5
                              )
            if i == 1:
                cv2.circle(img,
                              (int(width * polygon[1][0]), int(height * polygon[1][1])),
                              5,
                              (0,0,255),
                           thickness=5
                              )
            if i == 2:
                cv2.circle(img,
                              (int(width * polygon[2][0]), int(height * polygon[2][1])),
                              5,
                              (0,255,255),
                           thickness=5
                             )
            cv2.line(img,
                     (int(width * polygon_unsorted[i - 1][0]), int(height * polygon_unsorted[i - 1][1])),
                     (int(width * polygon_unsorted[i][0]), int(height * polygon_unsorted[i][1])),
                     color, 3
                     )
            cv2.putText(img,
                    str(f" prvi angle {round(angle, 3)}"),
                    (int(polygon[0][0] * width), int(polygon[0][1] * height)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255,255,255),
                    2,
                    cv2.LINE_AA
                    )

        if not something_weird_happened:
            xml_annotations.append([str(cx), str(cy), str(int(cw)), str(int(ch)), str(round(angle,7))])
    #write the image i havent been drawing on to the disk =:)
    if not something_weird_happened:
        img_written = cv2.imread(f'train_both/{image_name}')
        cv2.imwrite(os.path.join(images_foldername, f"{image_name}.bmp"), img_written)
        write_xml_annotations(xml_annotations, image_name, width, height)
    if display:
        cv2.imshow(image_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
print(counter)
generate_trainval_files()
