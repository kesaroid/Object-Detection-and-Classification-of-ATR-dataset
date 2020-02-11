'''
Kesar TN
University of Central Florida
kesar@Knights.ucf.edu
'''
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import json

class AtrDataset:
    def __init__(self,sample_dir):
        self.path = sample_dir +'/' + sample_dir + '.p'
        self.imagedir = sample_dir  + '/' +'images'
        self.samples = pickle.load(open(self.path, 'rb'))
        # self.info()

    def get_num_targets(self):
        num_targets = 0
        for frame in self.samples:
            num_targets += len(frame['targets'])
        return num_targets

    def _get_categories(self):
        categories = []
        for frame in self.samples:
            for target in frame['targets']:
                if target['category'] not in categories:
                    categories.append(target['category'])
        return categories

    def get_image_names(self):
        names = []
        for frame in self.samples:
            names.append(frame['name'] + '_' + frame['frame'])
        return names

    def _get_ranges(self, sample):
        rangecount = {}
        for frame in self.samples:
            if frame['range'] not in rangecount.keys():
                rangecount[frame['range']] = 1
            else:
                rangecount[frame['range']] += 1
        if sample:
            print('Range   Count')
            for range in rangecount.keys():
                print('{:<7}'.format(range), rangecount[range])

    def _get_day(self):
        day = 0
        night = 0
        for frame in self.samples:
            if frame['day']:
                day += 1
            else:
                night += 1
        print ('day \t night')
        print(day, ' \t ',night)

    def get_num_frames(self):
        return(len(self.samples))

    def info(self):
        self.number_of_frames = len(self.samples)
        self.number_of_targets = self.get_num_targets()
        self.categories = self._get_categories()
        print('frames ', self.number_of_frames)
        print('targets ', self.number_of_targets)
        print('number of classes ', len(self.categories))
        print('categories', self.categories)
        self.count_categories()
        self._get_ranges()
        self._get_day()

    def count_categories(self):
        print('Class    Count')
        for category in self.categories:
            count = 0
            for frame in self.samples:
                for target in frame['targets']:
                    if target['category']== category:
                        count += 1
            print('{:<8}'.format(category),count)

    def _category_lookup(self,name):
        lookup = {'MAN': 1, 'PICKUP': 2, 'SUV': 3, 'BTR70': 4, 'BRDM2': 5, 'BMP2': 6, 'T72': 7, 'ZSU23': 8, '2S3': 9,
                  'D20': 10, 'MTLB': 11}
        return lookup[name]
    
    def class_names(self):
        names = ['MAN', 'PICKUP', 'SUV', 'BTR70', 'BRDM2', 'BMP2', 'T72', 'ZSU23', '2S3', 'D20', 'MTLB']
        return names

    def _category_id(self, name, i):
        # Find the annotation for the image and return
        for frame in self.samples:
            if name == frame['name'] + '_' + frame['frame']:
                category = frame['targets'][i]
                class_id = self._category_lookup(category['category'])
        return class_id
    
    def get_categoryname(self,num):
        lookup = {'1': 'MAN', '2': 'PICKUP', '3': 'SUV', '4': 'BTR70', '5': 'BRDM2', '6': 'BMP2', '7': 'T72',
                  '8': 'ZSU23', '9': '2S3', '10': 'D20', '11': 'MTLB'}
        return lookup[str(num)]
    
    def get_categories(self):
        labels = [{"id": 1, "name": "MAN"}, {"id": 2, "name": "PICKUP"}, {"id": 3, "name": "SUV"}, {"id": 4, "name": "BTR70"}, 
                    {"id": 5, "name": "BRDM2"}, {"id": 6, "name": "BMP2"}, {"id": 7, "name": "T72"}, {"id": 8, "name": "ZSU23"}, 
                    {"id": 9, "name": "2S3"}, {"id": 10, "name": "D20"}, {"id": 11, "name": "MTLB"}]
        return labels

    def _image_names_dict(self):
        names_list = []
        for frame in self.samples:
            names_list.append({'filename': frame['name'] + '_' + frame['frame'] + '.jpg', 'id': frame['sample_id']})
        return names_list
    
    def _get_bbox(self, target):
        cx = target['center'][0]
        cy = target['center'][1]

        ulx = target['ul'][0]
        uly = target['ul'][1]

        bbox_width = (cx - ulx) * 2
        bbox_height = (cy - uly) * 2

        xmin = ulx
        xmax = ulx + bbox_width
        ymin = uly
        ymax = uly + bbox_height
        return xmin, ymin, xmax, ymax

    def showSample(self, idx):
        # shows sample indexed from master file and avi frame with bbox gt overlay
        sample = self.samples[idx]
        print(sample)
        imgfile = self.imagedir + '/' + sample['name'] + '_' + sample['frame'] + '.jpg'
        image = Image.open(imgfile)

        f, ax = plt.subplots(figsize=(10, 10))
        for target in sample['targets']:
            cx = target['center'][0]
            cy = target['center'][1]

            ulx = target['ul'][0]
            uly = target['ul'][1]

            w = (cx - ulx) * 2
            h = (cy - uly) * 2

            rect = patches.Rectangle((ulx, uly), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.imshow(image)
        ax.set_title(sample['name'], fontsize=30)
        plt.show()
    
    def _get_data(self):
        return self.samples
        