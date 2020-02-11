'''
Kesar TN
University of Central Florida
kesar@Knights.ucf.edu
'''
import mrcnn
from dataset import *
import os
from xml.etree import ElementTree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, mold_image
from mrcnn.visualize import display_instances, plot_precision_recall, plot_overlaps
from mrcnn.utils import extract_bboxes, compute_ap
import mrcnn.model as modellib

class IRDataset(Dataset):
	# load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "MAN")
        self.add_class("dataset", 2, "PICKUP")
        self.add_class("dataset", 3, "SUV")
        self.add_class("dataset", 4, "BTR70")
        self.add_class("dataset", 5, "BRDM2")
        self.add_class("dataset", 6, "BMP2")
        self.add_class("dataset", 7, "T72")
        self.add_class("dataset", 8, "ZSU23")
        self.add_class("dataset", 9, "2S3")
        self.add_class("dataset", 10, "D20")
        self.add_class("dataset", 11, "MTLB")
		# define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annotations/'
		# find all images
        image_count = 0
        for filename in os.listdir(images_dir):
            image_count += 1
            image_id = filename[:-4]
			# skip all images after 150 if we are building the train set
            if is_train and image_count >= 7776:
                continue
            if not is_train and image_count < 7776:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
            
	# extract bounding boxes from an annotation fil
    def extract_boxes(self, filename):
		# load and parse the file
        tree = ElementTree.parse(filename)
		# get the root of the document
        root = tree.getroot()
		# extract each bounding box
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
		# extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

	# load the masks for an image
    def load_mask(self, image_id):
		# get details of image
        info = self.image_info[image_id]
		# define box file location
        path = info['annotation']
		# load XML
        boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
        class_ids = []
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(atrdata._category_id(info['id'], i))
        return masks, np.asarray(class_ids, dtype='int32')

	# load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

class IRConfig(Config):
    NAME = "IR_cfg"
    NUM_CLASSES = 11 + 1
    BACKBONE = "resnet101"
    STEPS_PER_EPOCH = 9720
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

def show_sample(data_set, image_id):
    image = data_set.load_image(image_id)
    mask, class_ids = data_set.load_mask(image_id)
    bbox = extract_bboxes(mask)
    display_instances(image, bbox, mask, class_ids, data_set.class_names)


if __name__ == "__main__":
	
	dataset_dir = 'scaled2500_1to2'
	test_dir = 'scaled2500_25to35day'

	atrdata = AtrDataset(dataset_dir)

	# Train set
	train_set = IRDataset()
	train_set.load_dataset(dataset_dir, is_train=True)
	train_set.prepare()
	print('Train: %d' % len(train_set.image_ids))

	# Validation set
	val_set = IRDataset()
	val_set.load_dataset(dataset_dir, is_train=False)
	val_set.prepare()
	print('Validation: %d' % len(val_set.image_ids))

	# show_sample(val_set, 1876)

	# Train
	config = IRConfig()
	# Define the model
	model = MaskRCNN(mode='training', model_dir='mrcnn/logs', config=config)
	# Load weights
	model.load_weights('mrcnn/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
	# Train weights
	model.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=75, layers='heads')