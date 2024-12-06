import os
import json
import shutil
import numpy as np
import random


# create three new directories
train_folder = 'train_folder/'
val_folder = 'val_folder/'
test_folder = 'test_folder/'
if not os.path.exists(train_folder):
    os.mkdir(train_folder)
if not os.path.exists(val_folder):
    os.mkdir(val_folder)
if not os.path.exists(test_folder):
    os.mkdir(test_folder)

# load json file
old_folder = 'train_set/'
label_files = [os.path.join(old_folder, f"label_data_{suffix}.json")
               for suffix in ['0313', '0531', '0601']]

labels_collection = []
for label_file in label_files:
    with open(label_file, 'r') as f:
        labels = [json.loads(line) for line in f]

        for label in labels:
            labels_collection.append(label)

collection_size = len(labels_collection)

# iterate for each labelled data
count = 0
index = np.arange(collection_size)
random.shuffle(index)

# split to three datasets
train_size = int(collection_size * 0.7)
val_size = int(collection_size * 0.15)

train_ind = index[:train_size]
val_ind = index[train_size:train_size + val_size]
test_ind = index[train_size + val_size:]

for i in range(3):
    new_labels_collection = []

    if i == 0:
        ind_collection = train_ind
        folder_path = train_folder
        new_json_file = 'train_label_data.json'
    elif i == 1:
        ind_collection = val_ind
        folder_path = val_folder
        new_json_file = 'val_label_data.json'
    elif i == 2:
        ind_collection = test_ind
        folder_path = test_folder
        new_json_file = 'test_label_data.json'

    for ind in ind_collection:
        label = labels_collection[ind]

        old_img_path = os.path.join(old_folder, label['raw_file'])

        # new_image_path
        new_name = label['raw_file'].replace('clips/', '')
        new_name = new_name.replace('/', '_')
        new_img_path = os.path.join(folder_path, new_name)

        # copy image
        shutil.copy(old_img_path, new_img_path)

        # label file
        label['raw_file'] = new_name
        new_labels_collection.append(label)

    # load the new json file
    with open(os.path.join(folder_path, new_json_file), 'a') as f:
        for label in new_labels_collection:
            json.dump(label, f)
            f.write('\n')
