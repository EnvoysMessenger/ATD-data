import os
import sys
from mrcnn import model
from mrcnn import visualize
from mrcnn import utils
import train
import skimage.io
import numpy as np

root_dir=os.path.abspath('')
weight_dir=os.path.join(root_dir,'logs')
result_dir=os.path.join(root_dir,'result')
train_image_dir=os.path.join(root_dir,'dataset','train')

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

print("Train Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))


model_rcnn=model.MaskRCNN(mode="inference", model_dir=weight_dir, config=config)
model_rcnn.load_weights(model_weight,by_name=True)

class_names = ['BG','tank']


file_names = next(os.walk(train_image_dir))[2]
for x in range(len(file_names)):
    image = skimage.io.imread(os.path.join(train_image_dir, file_names[x]))
    output_dir=os.path.join(result_dir,file_names[x])
    results=model_rcnn.detect([image],verbose=1)

    r=results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores']).savefig(output_dir)
    


# image_ids = dataset_val.image_ids[:2]
# for image_id in image_ids:
#     print(image_id)
#     image = dataset_val.load_image(image_id)
#     mask, class_ids = dataset_val.load_mask(image_id)
#     pic=visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
#     result_file=file_names[image_ids]
#     out_file=os.path.join(result_dir,result_file)
#     pic.save(out_file)
# image_ids = np.random.choice(dataset_train.image_ids, 10)
# APs = []
# for image_id in image_ids:
#     # Load image and ground truth data
#     image, image_meta, gt_class_id, gt_bbox, gt_mask = \
#         model.load_image_gt(dataset_train, config,
#                                image_id, use_mini_mask=False)
#     molded_images = np.expand_dims(model.mold_image(image, config), 0)
#     # Run object detection
#     results = model_rcnn.detect([image], verbose=0)
#     r = results[0]
#     # Compute AP
#     AP, precisions, recalls, overlaps = \
#         utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                          r["rois"], r["class_ids"], r["scores"], r['masks'])
#     APs.append(AP)
    
# print("mAP: ", np.mean(APs))
