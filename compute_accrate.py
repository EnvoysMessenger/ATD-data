import os
import sys
from mrcnn import model
from mrcnn import visualize
from mrcnn import utils
import train
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lxml import etree


root_dir=os.path.abspath('')
weight_dir=os.path.join(root_dir,'logs')
result_dir=os.path.join(root_dir,'result')
train_image_dir=os.path.join(root_dir,'dataset','train')


test_img_dir = os.path.join(root_dir,'testimage','images')
test_output_dir = os.path.join(root_dir,'testimage','results')

model_weight=os.path.join(weight_dir,"mask_rcnn_weight.h5")
sys.path.append(root_dir)

config=train.BalloonConfig()
dataset_dir=os.path.join(root_dir,'dataset')

dataset_train = train.BalloonDataset()
dataset_train.load_balloon(dataset_dir, "train")
dataset_train.prepare()

dataset_val = train.BalloonDataset()
dataset_val.load_balloon(dataset_dir, "val")
dataset_val.prepare()

source_img_w = 1474
source_img_h = 1784

output_img_width = 1024 
output_img_high = 768

class region:
    #在python坐标系下的坐标
    left_x=0
    right_x=0
    high_y=0
    low_y=0

    def __init__(self, left_x,low_y,right_x,high_y):
        self.left_x=left_x        
        self.low_y=low_y
        self.high_y=high_y
        self.right_x=right_x

    def showPoints(self):
        print(self.left_x,self.low_y,self.right_x,self.high_y)

    def saveRegion(self,Img_h,Img_w,save_img_dir):
        img_array=np.zeros([Img_h,Img_w])
        img_array[self.left_x:self.right_x,self.low_y:self.high_y]=255
        result_img = Image.fromarray(np.uint8(img_array))
        result_img.save(save_img_dir)

    #判断起点为region_x,region_y，高宽为h,w的框是否在该区域中
    def detect(self,min_x,min_y,max_x,max_y):
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


def computeLowerLeftPoint(output_img_w,output_img_h,cent_x,cent_y):
    img_w=source_img_w
    img_h=source_img_h
    dict_x={0:cent_x,img_w-output_img_w:img_w-cent_x,cent_x-output_img_w/2:output_img_w/2}
    x=min(dict_x,key=dict_x.get)
    dict_y={0:cent_y,img_h-output_img_h:img_h-cent_y,cent_y-output_img_h/2:output_img_h/2}
    y=min(dict_y,key=dict_y.get)
    return x,y

#从mask中获取所有的区域，组合为region-list
def getRegionFromArray(input_array):
    regions=[]
    for i in range(0,input_array.shape[2]):
        points = np.where(input_array[:,:,i] ==  True)
        min_x = np.min(points[0])
        max_x = np.max(points[0])
        min_y = np.min(points[1])
        max_y = np.max(points[1])
        region_true = region(min_x, min_y, max_x, max_y)
        regions.append((region_true))
    return regions

def getRegionFromXml(xml_file,output_img_h,output_img_w):
    regions = []
    xml = etree.parse(xml_file)
    list_res = xml.xpath('//Pixel')
    Point_X = []
    Point_Y = []
    region_0 = list_res[0]
    for pt in region_0.xpath('Pt'):
        Point_X.append(int(pt.xpath('@LeftTopX')[0]))
        Point_Y.append(int(pt.xpath('@LeftTopY')[0]))
        pass
    lower_left_x,lower_left_y = computeLowerLeftPoint(output_img_width, output_img_high, min(Point_X), min(Point_Y))
    detect_area = region(lower_left_x, lower_left_y, lower_left_x+output_img_width, lower_left_y+output_img_high)
    region_output_0 = region(min(Point_X), min(Point_Y), max(Point_X), max(Point_Y))
    regions.append(region_output_0)
    for regions_left in list_res[1:]:
        Point_X = []
        Point_Y = []
        for pt in regions_left.xpath('Pt'):
            Point_X.append(int(pt.xpath('@LeftTopX')[0]))
            Point_Y.append(int(pt.xpath('@LeftTopY')[0]))
            if detect_area.detect(min(Point_X), min(Point_Y), max(Point_X), max(Point_Y)):
                region_temp = region(min(Point_X), min(Point_Y), max(Point_X), max(Point_Y))
                regions.append(region_temp)
    return regions


# model_rcnn=model.MaskRCNN(mode="inference", model_dir=weight_dir, config=config)
# model_rcnn.load_weights(model_weight,by_name=True)

# class_names = ['BG','tank']

#范例：如何测试并输出结果
# file_names = next(os.walk(train_image_dir))[2]
# for x in range(len(file_names)):
#     image = skimage.io.imread(os.path.join(train_image_dir, file_names[x]))
#     output_dir=os.path.join(result_dir,file_names[x])
#     results=model_rcnn.detect([image],verbose=1)

#     r=results[0]
#     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores']).savefig(output_dir)


#单例测试
img_name = str(1)#测试文件名
image = skimage.io.imread(os.path.join(train_image_dir, '{name}.jpg'.format(name = img_name)))
test_output_img_dir = os.path.join(test_output_dir, '{name}.jpg'.format(name = img_name))
test_output_txt_dir = os.path.join(test_output_dir, '{name}.txt'.format(name = img_name))
test_read_txt_dir = os.path.join(test_output_dir, '{name}.txt.npy'.format(name = img_name))
# results=model_rcnn.detect([image],verbose=1)
# r=results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                              class_names, r['scores'])
#                             .savefig(test_output_img_dir)
# mask_test = r['masks']
# np.save(test_output_txt_dir,mask_test)
#print(mask_test.shape,np.max(mask_test),np.min(mask_test))
mask_test = np.load(test_read_txt_dir)
#print(mask_test.shape)
test_regions = getRegionFromArray(mask_test)
print(len(test_regions),test_regions)
test_regions[0].saveRegion(768,1024,test_output_img_dir)
