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
import itertools
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix
from keras.utils import to_categorical

from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, mold_image, log
from mrcnn import visualize
from mrcnn import utils
from mrcnn.utils import extract_bboxes, compute_ap
import mrcnn.model as modellib

# Class defining dataset
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
            self.add_image('dataset', image_id=image_id,
                           path=img_path, annotation=ann_path)

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

# Prediction Configuration
class PredictionConfig(Config):
    NAME = "IR_cfg_test"
    NUM_CLASSES = 11 + 1
    BACKBONE = "resnet101"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

# Function to show image sample
def show_sample(data_set, image_id):
    image = data_set.load_image(image_id)
    mask, class_ids = data_set.load_mask(image_id)
    bbox = extract_bboxes(mask)
    visualize.display_instances(image, bbox, mask, class_ids, data_set.class_names)

# Function for prediction of 1 image
def atrpredict(image_id, image_name, dataset, model, cfg):
	# load the image and mask
    image = dataset.load_image(image_id)
    mask, _ = dataset.load_mask(image_id)
	# convert image into one sample
    sample = np.expand_dims(image, 0)
	# make prediction
    yhat = model.detect(sample, verbose=0)[0]
    plt.imshow(image)
    plt.title('Actual')
    for j in range(mask.shape[2]):
        plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
	# plot raw pixel data
    plt.imshow(image)
    ax = plt.gca()
	# plot each box
    for index, box in enumerate(yhat['rois']):
        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1
		# create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        ax.add_patch(rect)
        ax.text(x1 - 5, y1 - 5, atrdata.get_categoryname(str(yhat['class_ids'][index])) + ' - ' +
                    "{:.3f}".format(yhat['scores'][index]), color='w', size=9, backgroundcolor="none")
    
    img, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(test_set, pred_config, image_id, use_mini_mask=False)
    y_true_title = str()
    for i in gt_class_id:
        y_true = atrdata.get_categoryname(int(i))
        y_true_title = y_true_title + '  ' + y_true
    plt.title('Actual Class: ' + y_true_title)
    plt.show()
    # plt.clf()

# Function to compute mAP for multiple values
def compute_batch_ap(dataset, image_ids):
    APs = []; prec = []; rec = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, pred_config,
                                   image_id, use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps =\
            compute_ap(gt_bbox, gt_class_id, gt_mask,
							r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
        prec.append(precisions)
        rec.append(recalls)
    prec = np.array(prec)
    rec = np.array(rec)
    return APs, prec, rec

# Function for drawing bounding boxes
def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = visualize.random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})
    ax.imshow(masked_image.astype(np.uint8))
    plt.show()


# Function for drawing predictions
def draw_preds(limit):
    pillar = model.keras_model.get_layer("ROI").output
    
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
    if nms_node is None:
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
    if nms_node is None:
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")
        
    rpn = model.run_graph([image], [
		("rpn_class", model.keras_model.get_layer("rpn_class").output),
		("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
		("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
		("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
		("post_nms_anchor_ix", nms_node),
		("proposals", model.keras_model.get_layer("ROI").output),
	])
	
    # Convert back to image coordinates for display
    h, w = pred_config.IMAGE_SHAPE[:2]
    proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
    draw_boxes(image, refined_boxes=proposals)

# Function for Confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = np.expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = np.mean(APs)
	return mAP

# Function for mAP curve
def plot_precision_recall(AP, precisions, recalls):
	_, ax = plt.subplots(1)
	ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(np.mean(AP)))
	ax.set_ylim(0, 1.1)
	ax.set_xlim(0, 1.1)
	ax.plot(recalls, precisions)
	plt.show()

if __name__ == "__main__":
    # Path to dataset folers
    dataset_dir = 'scaled2500_1to2'
    test_dir = 'iter1_test'

	####################### Validation set ############################### 
    # atrdata = AtrDataset(dataset_dir)
    # val_set = IRDataset()
    # val_set.load_dataset(dataset_dir, is_train=False)
    # val_set.prepare()
    # print('Validation: %d' % len(val_set.image_ids))
    
    ####################### Test set ############################### 
    atrdata = AtrDataset(test_dir)
    test_set = IRDataset()
    test_set.load_dataset(test_dir, is_train=True)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))
    

    pred_config = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=pred_config)
    model_path = 'mrcnn/logs/ir_cfg20191119T1333/mask_rcnn_ir_cfg_0050.h5' # Path to Traind Model
    model.load_weights(model_path, by_name=True)

    # atrpredict(1440, 'cegr01925_0014_1.jpg', test_set, model, pred_config)

	####################### Display results ############################## 
    # image_id = 6452
    # show_sample(test_set, image_id)
    # image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(test_set, pred_config, image_id, use_mini_mask=False)
    # results = model.detect([image], verbose=1)
    # r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    # 							atrdata.class_names(), r['scores'],
	# 							title="Predictions")
    
    # draw_preds(15)

	####################### Plot Precision Recall graph for 1 image ##############################
    # AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask,
    #                                       r['rois'], r['class_ids'], r['scores'], r['masks'])                  
    # plot_precision_recall(AP, precisions, recalls)
    # plot_overlaps(gt_class_id, r['class_ids'], r['scores'], overlaps, atrdata.class_names())

	####################### Batch mAP ##############################
    # image_ids = list(range(10))
    # APs, Precisions, Recalls = compute_batch_ap(test_set, image_ids)
    # print(Recalls, np.ceil(np.average(Recalls, axis=0)))
    # plot_precision_recall(APs, np.ceil(np.average(Precisions, axis=0)), np.ceil(np.average(Recalls, axis=0)))
    # print("mAP @ IoU=50: ", np.mean(APs))

    ####################### Evaluate Moodel ##############################
    # Evaluate Moodel
    # val_mAP = evaluate_model(test_set, model, pred_config)
    # print(val_mAP)

    ####################### Save Predictions ##############################    
    # for i, img_name in enumerate(os.listdir(test_dir + '/images')):
    #     atrpredict(i, img_name, test_set, model, pred_config)
    #     print(i, img_name)

    ####################### Confusion Matrix ##############################
    # image_ids = len(os.listdir(test_dir + '/images/'))
    # y_true = []; y_pred = []
    # for i in range(image_ids):
    #     image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(test_set, pred_config, i, use_mini_mask=False)
    #     y_true.extend(gt_class_id)
    #     results = model.detect([image], verbose=1)
    #     r = results[0]
    #     if len(gt_class_id) == 1:
    #         gt_classes = r["class_ids"].tolist()
    #         y_pred.append(gt_classes[0])
    #     else:
    #         y_pred.extend(r["class_ids"])
        
    # plot_confusion_matrix(y_true, y_pred, classes=atrdata.class_names(),
    #                   title='Confusion matrix')
    # plt.savefig('confusion.png')