from lxml import etree
import os
import re
from PIL import Image
import numpy as np

# 区域类，有用两个点初始化的函数和判断是否在某个框中的函数


class region:
    # 在python坐标系下的坐标
    left_x = 0
    right_x = 0
    high_y = 0
    low_y = 0

    def __init__(self, left_x, low_y, right_x, high_y):
        self.left_x = left_x
        self.high_y = high_y
        self.right_x = right_x
        self.low_y = low_y

    # 判断起点为region_x,region_y，高宽为h,w的框是否在该区域中
    def detect(self, min_x, min_y, max_x, max_y):
        if self.left_x > min_x:
            return False
        elif self.right_x < max_x:
            return False
        elif self.low_y > min_y:
            return False
        elif self.high_y < max_y:
            return False
        else:
            return True

# 将图片转化为3通道


def trans_img(img_old_dir):
    im = np.asarray(img_old_dir)
    # if len(im.shape) == 2:
    c = []
    for i in range(3):
        c.append(im)
    im = np.asarray(c)
    im = im.transpose([1, 2, 0])
    im = Image.fromarray(np.uint8(im))
    return im

# 用于剪切坐标原点在左下角的函数，返回切后的图以及该图左下角的坐标


def img_crop(img_file, w, h, cent_x, cent_y):
    im = Image.open(img_file)
    img_w = im.size[0]
    img_h = im.size[1]
    dict_x = {0: cent_x, img_w-w: img_w-cent_x, cent_x-w/2: w/2}
    x = min(dict_x, key=dict_x.get)
    dict_y = {0: cent_y, img_h-h: img_h-cent_y, cent_y-h/2: h/2}
    y = min(dict_y, key=dict_y.get)
    region = im.crop((x, y, x+w, y+h))
    return region, x, y


def write_into_json(json_file, Point_X, Point_Y, file_size, img_name ,tank):
    str_x = ''
    str_y = ''
    str_x = str_x+str(int(min(Point_X)))+','+str(int(max(Point_X)))+','+str(
        int(max(Point_X)))+','+str(int(min(Point_X)))+','+str(int(min(Point_X)))+','
    str_y = str_y+str(int(min(Point_Y)))+','+str(int(min(Point_Y)))+','+str(
        int(max(Point_Y)))+','+str(int(max(Point_Y)))+','+str(int(min(Point_Y)))+','
    str_x = '['+str_x[:-1]+']'
    str_y = '['+str_y[:-1]+']'
    result = '"'+str(os.path.basename(img_name))+str(file_size)+'":{"fileref":"","size":'+str(file_size)+',"tank_kind":'+str(tank)[2]+',"filename":"'+str(os.path.basename(
        img_name))+'","base64_img_data":"","file_attributes":{},"regions":{"0":{"shape_attributes":{"name":"polygon","all_points_x":'+str_x+',"all_points_y":'+str_y+'},"region_attributes":{}}}},'
    with open(json_file, "a+") as file_w:
        file_w.write(result)

def adjustJsonFile(json_file):
    with open(json_file ,'r') as file_adjust:
        str_input = file_adjust.read()
        print(str_input)
        str_output = '{'+str_input[:-1]+'}'
        file_adjust.close()
    output_json_dir = os.path.join(os.path.dirname(json_file),'via_region_data.json')
    with open(output_json_dir , 'w') as file_output:
        file_output.write(str_output)
        file_output.close()

        #file_adjust.write(str_output)
def clearFolder(*args):
    for folder in args[0]:
        for f in os.listdir(folder):
            path_file = os.path.join(folder,f)
            os.remove(path_file)

def trans_xml(size, file_img, xml_file, file_json, w, h):
    xml = etree.parse(xml_file)

    # 需要写入到json文件的x，y左边的字符串
    # str_x = ''
    # str_y = ''

    list_res = xml.xpath('//Pixel')
    list_tank = xml.xpath('//tank_name')
    Point_X = []
    Point_Y = []

    start_point_x = 0
    start_point_y = 0
    # 获取包含第一个目标点的图像区域，生成region对象
    region_0 = list_res[0]
    tank_kind_0 = str(list_tank[0].xpath('@tank_name'))
    for pt in region_0.xpath('Pt'):
        Point_X.append(int(pt.xpath('@LeftTopX')[0]))
        Point_Y.append(int(pt.xpath('@LeftTopY')[0]))
        pass
    start_point_x = min(Point_X)
    start_point_y = min(Point_Y)
    _, region_x, region_y = img_crop(
        file_img, w, h, min(Point_X), min(Point_Y))
    dectet_area = region(region_x, region_y, region_x+w, region_y+h)
    write_into_json(file_json,(np.array(Point_X)-region_x).tolist(),(np.array(Point_Y)-region_y).tolist(),size,file_img,tank_kind_0)
    # 依次判断之后的每个框是否在上述图像区域中
    for regions_left,tank_kind_left in zip(list_res[1:],list_tank[1:]):
        Point_X = []
        Point_Y = []
        str_x_temp = ''
        str_y_temp = ''
        for pt in regions_left.xpath('Pt'):
            Point_X.append(int(pt.xpath('@LeftTopX')[0]))
            Point_Y.append(int(pt.xpath('@LeftTopY')[0]))
        if dectet_area.detect(min(Point_X), min(Point_Y), max(Point_X), max(Point_Y)):
            write_into_json(file_json,(np.array(Point_X)-region_x).tolist(),(np.array(Point_Y)-region_y).tolist(),size,file_img,str(tank_kind_left.xpath('@tank_name')))
        pass
    pass
    
    return region_x, region_y

def transformFromFolder(root_folder_dir ,traindata_proportion):
    cwd_img=os.path.join(root_folder_dir,"images")
    cwd_xml=os.path.join(root_folder_dir,"xmls")
    cwd_output_t=os.path.join(root_folder_dir,'dataset','train')
    cwd_output_v=os.path.join(root_folder_dir,'dataset','val')
    cwd_json_t=os.path.join(cwd_output_t,"train.json")
    cwd_json_v=os.path.join(cwd_output_v,"val.json")
    clearFolder([cwd_output_t,cwd_output_v])
    for root,files,filename in os.walk(cwd_img):
        for index,name in enumerate(filename):
            file_index ,type_index=os.path.splitext(name)
            img_path = os.path.join(cwd_img,file_index+'.jpg')
            xml_path = os.path.join(cwd_xml,file_index+'.xml')
            print(img_path)
            if index < len(filename)*traindata_proportion :
                size=os.path.getsize(img_path)
                region_x,region_y=trans_xml(size,img_path,xml_path,cwd_json_t,w,h)
                img_file=Image.open(img_path)
                img=img_file.crop((region_x,region_y,region_x+w,region_y+h))
                #img=trans_img(img)
                img.save(os.path.join(cwd_output_t,os.path.basename(name)))
            else:
                size=os.path.getsize(img_path)
                region_x,region_y=trans_xml(size,img_path,xml_path,cwd_json_v,w,h)
                img_file=Image.open(img_path)
                img=img_file.crop((region_x,region_y,region_x+w,region_y+h))
                #img=trans_img(img)
                img.save(os.path.join(cwd_output_v,os.path.basename(name)))
    adjustJsonFile(cwd_json_t)
    adjustJsonFile(cwd_json_v)


root_dir=os.getcwd()
cwd_img_complete=os.path.join(root_dir,'datasets',"complete_dataset")
cwd_img_test=os.path.join(root_dir,'datasets',"mini_dataset")

[w,h]=[1024,768]
traindataset_proportion = 0.8
transformFromFolder(cwd_img_complete,traindataset_proportion)
#transformFromFolder(cwd_img_test,traindataset_proportion)
