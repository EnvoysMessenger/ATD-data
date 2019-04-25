import os
import sys
from mrcnn import visualize,model
import train

root_dir=os.path.abspath('')

weight_path=os.path.join(root_dir,'logs')
weights_path=os.path.join(weight_path,'mask_rcnn_weight.h5')
sys.path.append(root_dir)

config =train.BalloonConfig()
images_dir=os.path.join(root_dir,'dataset')

dataset_train=train.BalloonDataset()
dataset_train.load_balloon(images_dir,'train')
dataset_train.prepare()

dataset_val=train.BalloonDataset()
dataset_val.load_balloon(images_dir,'val')
dataset_val.prepare()

model=model.MaskRCNN(mode="training",config=config,model_dir=weight_path)
model.load_weights(weights_path, by_name=True)
model.train(dataset_train,dataset_val,learning_rate=config.LEARNING_RATE,epochs=30,layers='all')