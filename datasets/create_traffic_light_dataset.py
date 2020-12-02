import os
import json
import glob
import sys
sys.setrecursionlimit(10**8)
import shutil
import numpy as np

# configuration
DATASET_FOLDER = './traffic_lights/'
ANNO_FOLDER_NAME = DATASET_FOLDER + 'annotations/'
TRAIN_JSON = 'instances4_train.json'
TRAIN_FOLDER_NAME = DATASET_FOLDER + 'train/'
TEST_JSON = 'instances4_test.json'
TEST_FOLDER_NAME = DATASET_FOLDER + 'test/'

H, W = 512, 512
TL_LABEL = 18
POLE_LABEL = 5
CAR_LABEL = 10
SIGN_LABEL = 12
MIN_AREA = 100

OTHER_LABELS = [(3, SIGN_LABEL), (4, CAR_LABEL)]
# OTHER_LABELS = [(3, POLE_LABEL), (4, SIGN_LABEL), (5, CAR_LABEL)]
# OTHER_LABELS = []

INPUT_FOLDER = './rgb_images/'
GT_FOLDER = './semantic_images/'
NUM_TEST = 2000

def _dfs(label, visited, y, x, area=0,
         y_range=(float('inf'), float('-inf')),
         x_range=(float('inf'), float('-inf')),
         ref_label = TL_LABEL):
    if visited[y][x]:
        return False, 0, (0, 0), (0, 0)
    visited[y][x] = True

    if label[y][x] == ref_label:
        area += 1
        y_range = (min(y_range[0], y), max(y_range[1], y))
        x_range = (min(x_range[0], x), max(x_range[1], x))

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny][nx]:
                _s, _a, _yr, _xr = _dfs(label, visited, ny, nx, area=area,
                                        y_range=y_range, x_range=x_range,
                                        ref_label=ref_label)
                if _s:
                    area = _a
                    y_range = _yr
                    x_range = _xr

        return True, area, y_range, x_range
    return False, area, y_range, x_range


if not os.path.exists(ANNO_FOLDER_NAME):
    os.makedirs(ANNO_FOLDER_NAME)
if not os.path.exists(TRAIN_FOLDER_NAME):
    os.makedirs(TRAIN_FOLDER_NAME)
if not os.path.exists(TEST_FOLDER_NAME):
    os.makedirs(TEST_FOLDER_NAME)

input_ = glob.glob(INPUT_FOLDER + '*_[0-9].*')
gt = glob.glob(GT_FOLDER + '*_[0-9].*')

print(len(input_))

assert len(input_) == len(gt), '{} != {}'/format(len(input_), len(gt))

train_split, test_split = input_[:-NUM_TEST], input_[-NUM_TEST:]

train_json, test_json = {}, {}
train_json['info'] = {
    "description": "Traffic Light Train",
    "url": "",
    "version": "1.0",
    "year": 2020,
    "contributor": "",
    "date_created": "2020/11/01",
}
test_json['info'] = {
    "description": "Traffic Light Test",
    "url": "",
    "version": "1.0",
    "year": 2020,
    "contributor": "",
    "date_created": "2020/11/01",
}
train_json['info'] = [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    },
]
test_json['info'] = [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    },
]
train_json['categories'] = [
    {"supercategory": "lights","id": 1,"name": "Green"},
    {"supercategory": "lights","id": 2,"name": "Red"},
    # {"supercategory": "others","id": 3,"name": "Pole"},
    {"supercategory": "others","id": 3,"name": "Sign"},
    {"supercategory": "others","id": 4,"name": "Car"},
]
test_json['categories'] = [
    {"supercategory": "lights","id": 1,"name": "Green"},
    {"supercategory": "lights","id": 2,"name": "Red"},
    # {"supercategory": "others","id": 3,"name": "Pole"},
    {"supercategory": "others","id": 3,"name": "Sign"},
    {"supercategory": "others","id": 4,"name": "Car"},
]
train_json['images'] = []
test_json['images'] = []
train_json['annotations'] = []
test_json['annotations'] = []

i, j = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
idx_arr = np.stack((i, j), axis=-1)
ANNO_IDX = 0

for idx, img_path in enumerate(train_split):
    prefix, name = img_path.rsplit('/', 1)
    frame, dist, gt_color = name[:-4].split('_')
    dist = float(dist)

#     print(img_path)

    train_json['images'].append({
        "license": 1,
        "file_name": name,
        "height": H,
        "width": W,
        "id": idx,
    })

    label = np.load(GT_FOLDER + name[:-3] + 'npy')
    visited = np.zeros_like(label, dtype=np.bool)
    max_area = float('-inf')

    for y, x in idx_arr[label == TL_LABEL]:
        status, area, y_range, x_range = _dfs(label, visited, y, x, area=0)
        if status and area > max_area:
            max_area = area
            at = (y_range, x_range)
    print(idx, name, at, max_area)

    x1, y1, dx, dy = at[1][0], at[0][0], at[1][1] - at[1][0], at[0][1] - at[0][0]
    x1, y1, dx, dy = int(x1), int(y1), int(dx), int(dy)

    train_json['annotations'].append({
        "area": max_area,
        "bbox": [x1, y1, dx, dy],
        "iscrowd": 0,
        "image_id": idx,
        "category_id": 1 if gt_color == 'Green' else 2,
        "id": ANNO_IDX,
    })
    ANNO_IDX += 1

    # for other label
    for CAT, LABEL in OTHER_LABELS:
        visited = np.zeros_like(label, dtype=np.bool)
        for y, x in idx_arr[label == LABEL]:
            status, area, y_range, x_range = _dfs(label, visited, y, x, area=0, ref_label=LABEL)
            # print(len(visited[visited == 1]))
            if status and area > MIN_AREA:
                at = (y_range, x_range)
                print(CAT, LABEL, idx, name, at, area)

                x1, y1, dx, dy = at[1][0], at[0][0], at[1][1] - at[1][0], at[0][1] - at[0][0]
                x1, y1, dx, dy = int(x1), int(y1), int(dx), int(dy)

                train_json['annotations'].append({
                    "area": area,
                    "bbox": [x1, y1, dx, dy],
                    "iscrowd": 0,
                    "image_id": idx,
                    "category_id": CAT,
                    "id": ANNO_IDX,
                })
                ANNO_IDX += 1


    # shutil.copyfile(img_path, TRAIN_FOLDER_NAME + name)


ANNO_IDX = 0
for idx, img_path in enumerate(test_split):
    prefix, name = img_path.rsplit('/', 1)
    frame, dist, gt_color = name[:-4].split('_')
    dist = float(dist)

    print(img_path)

    test_json['images'].append({
        "license": 1,
        "file_name": name,
        "height": H,
        "width": W,
        "id": idx
    })

    label = np.load(GT_FOLDER + name[:-3] + 'npy')
    visited = np.zeros_like(label, dtype=np.bool)
    max_area = float('-inf')

    for y, x in idx_arr[label == TL_LABEL]:
        status, area, y_range, x_range = _dfs(label, visited, y, x, area=0)
        if status and area > max_area:
            max_area = area
            at = (y_range, x_range)
    print(idx, name, at, max_area)

    x1, y1, dx, dy = at[1][0], at[0][0], at[1][1] - at[1][0], at[0][1] - at[0][0]
    x1, y1, dx, dy = int(x1), int(y1), int(dx), int(dy)

    test_json['annotations'].append({
        "area": max_area,
        "bbox": [x1, y1, dx, dy],
        "iscrowd": 0,
        "image_id": idx,
        "category_id": 1 if gt_color == 'Green' else 2,
        "id": ANNO_IDX,
    })
    ANNO_IDX += 1

    # for other label
    for CAT, LABEL in OTHER_LABELS:
        visited = np.zeros_like(label, dtype=np.bool)
        for y, x in idx_arr[label == LABEL]:
            status, area, y_range, x_range = _dfs(label, visited, y, x, area=0, ref_label=LABEL)
            if status and area > MIN_AREA:
                at = (y_range, x_range)
                print(CAT, LABEL, idx, name, at, area)

                x1, y1, dx, dy = at[1][0], at[0][0], at[1][1] - at[1][0], at[0][1] - at[0][0]
                x1, y1, dx, dy = int(x1), int(y1), int(dx), int(dy)

                test_json['annotations'].append({
                    "area": area,
                    "bbox": [x1, y1, dx, dy],
                    "iscrowd": 0,
                    "image_id": idx,
                    "category_id": CAT,
                    "id": ANNO_IDX,
                })
                ANNO_IDX += 1

    # shutil.copyfile(img_path, TEST_FOLDER_NAME + name)

with open(ANNO_FOLDER_NAME + TRAIN_JSON, 'w') as f:
    json.dump(train_json, f)
with open(ANNO_FOLDER_NAME + TEST_JSON, 'w') as f:
    json.dump(test_json, f)

print('DONE')
