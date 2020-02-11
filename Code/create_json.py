'''
Kesar TN
University of Central Florida
kesar@Knights.ucf.edu
'''
import os
import json
from dataset import *
from PIL import Image

json_name = 'scaled2500_1to2'
train_dir = json_name + '/images/'

dataset = AtrDataset(json_name)
data_all = dataset._get_data()


'''
{
    'name': 'cegr02007_0014',
    'frame': '896',
    'targets': [{
        'id': '054',
        'category': 'D20',
        'center': (305, 197),
        'ul': (296, 192),
        'bbox_area': 264,
        'contrast': 16.414502164502167,
        'inst_id': '0283343'
    }, {
        'id': '405',
        'category': 'MTLB',
        'center': (319, 198),
        'ul': (309, 193),
        'bbox_area': 312,
        'contrast': 16.89242788461538,
        'inst_id': '0283344'
    }],
    'sample_id': '0211486',
    'range': 2,
    'day': 1,
    'test': 0
}
'''
###############################################################
# Combine Pickle files

# with open('scaled2500_1to2\scaled2500_1to2.p', 'rb') as f:
#     my_dict_final = pickle.load(f)
# with open('scaled2500_25to35day\scaled2500_25to35day.p', 'rb') as f:
#     my_dict_final.extend(pickle.load(f)) 

# with open('data_variants\iter1\iter1.p', 'wb') as f:
#     pickle.dump(my_dict_final, f)

###############################################################
# Create JSON format

# data = {}
# data['annotations'] = []
# for frame in data_all:
#     frame.update(image_name = frame['name'] + '_' + frame['frame'])
#     data['annotations'].append(frame)

# with open(json_name + '.json', 'w') as outfile:
#     json.dump(data, outfile)

###############################################################
# # Create xml files
# from pascal_voc_writer import Writer

# outfile = json_name + '/annotations/'

# for frame in data_all:
#     image_name = frame['name'] + '_' + frame['frame'] + '.jpg'
#     if image_name in os.listdir(train_dir):
#         image = Image.open(train_dir + image_name)
#         image = np.asarray(image)
#         writer = Writer(image_name, image.shape[1], image.shape[0])
#         for targets in frame['targets']:
#             xmin, ymin, xmax, ymax = dataset._get_bbox(targets)
#             writer.addObject(targets['category'], xmin, ymin, xmax, ymax)
#             writer.save(outfile + image_name[:-4] + '.xml')
# print("Process Complete")

###############################################################
# # Seperate Knight and Day
# night, day = 0, 0
# for frame in data_all:
#     img = Image.open(train_dir + frame['name'] + '_' + frame['frame'] + '.jpg')
#     if frame['day']:
#         img.save('day/' + frame['name'] + '_' + frame['frame'] + '.jpg')
#         day += 1
#     else:
#         img.save('night/' + frame['name'] + '_' + frame['frame'] + '.jpg')
#         night += 1

# print ('Total saved\nday \t night')
# print(day, ' \t ',night)

###############################################################
# # Seperate ranges
# from shutil import copy2

# for frame in data_all:
#     img = Image.open(train_dir + frame['name'] + '_' + frame['frame'] + '.jpg')
#     if frame['range'] == 1:
#         img.save('data_variants/range/1000/images/' + frame['name'] + '_' + frame['frame'] + '.jpg')
#         copy2(json_name + '/annotations/' + frame['name'] + '_' + frame['frame'] + '.xml', 'data_variants/range/1000/annotations/')
#     elif frame['range'] == 1.5:
#         img.save('data_variants/range/1500/images/' + frame['name'] + '_' + frame['frame'] + '.jpg')
#         copy2(json_name + '/annotations/' + frame['name'] + '_' + frame['frame'] + '.xml', 'data_variants/range/1500/annotations/')
#     elif frame['range'] == 2:
#         img.save('data_variants/range/2000/images/' + frame['name'] + '_' + frame['frame'] + '.jpg')
#         copy2(json_name + '/annotations/' + frame['name'] + '_' + frame['frame'] + '.xml', 'data_variants/range/2000/annotations/')
#     elif frame['range'] == 2.5:
#         img.save('data_variants/range/2500/images/' + frame['name'] + '_' + frame['frame'] + '.jpg')
#         copy2(json_name + '/annotations/' + frame['name'] + '_' + frame['frame'] + '.xml', 'data_variants/range/2500/annotations/')
#     elif frame['range'] == 3:
#         img.save('data_variants/range/3000/images/' + frame['name'] + '_' + frame['frame'] + '.jpg')
#         copy2(json_name + '/annotations/' + frame['name'] + '_' + frame['frame'] + '.xml', 'data_variants/range/3000/annotations/')
#     elif frame['range'] == 3.5:
#         img.save('data_variants/range/3500/images/' + frame['name'] + '_' + frame['frame'] + '.jpg')
#         copy2(json_name + '/annotations/' + frame['name'] + '_' + frame['frame'] + '.xml', 'data_variants/range/3500/annotations/')