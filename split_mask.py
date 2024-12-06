import os
import json
import shutil
import numpy as np
import random
import cv2
import matplotlib.image


# create three new directories
train_folder = 'train_masks2/'
val_folder = 'val_masks2/'
test_folder = 'test_masks2/'

train_json_folder = 'train_folder/'
val_json_folder = 'val_folder/'
test_json_folder = 'test_folder/'

if not os.path.exists(train_folder):
    os.mkdir(train_folder)
if not os.path.exists(val_folder):
    os.mkdir(val_folder)
if not os.path.exists(test_folder):
    os.mkdir(test_folder)

for i in range(0, 3):
    labels_collection = []

    if i == 0:
        folder_path = train_folder
        json_folder = train_json_folder
        json_file = 'train_label_data.json'
    elif i == 1:
        folder_path = val_folder
        json_folder = val_json_folder
        json_file = 'val_label_data.json'
    elif i == 2:
        folder_path = test_folder
        json_folder = test_json_folder
        json_file = 'test_label_data.json'

    with open(os.path.join(json_folder, json_file)) as f:
        labels = [json.loads(line) for line in f]

        for label in labels:
            mask = np.zeros((720, 1280), dtype=np.uint8)

            lanes = label['lanes']
            h_samples = label['h_samples']
            raw_file = label['raw_file']

            for lane in lanes:
                if len(lane) > 0:
                    points = [(x, y) for (x, y) in zip(lane, h_samples) if x >= 0]
                    for j in range(len(points) - 1):
                        cv2.line(mask, points[j], points[j + 1], 255, 5)

            unique_value = np.unique(mask)
            if len(unique_value) > 2:
                print("stop")

            # save the mask
            mask_name,_ = os.path.splitext(raw_file)
            mask_name = mask_name + '_mask.jpg'
            # cv2.imwrite(os.path.join(folder_path, mask_name), mask)
            matplotlib.image.imsave(os.path.join(folder_path, mask_name), mask)
            print(f"saved mask {mask_name}")

