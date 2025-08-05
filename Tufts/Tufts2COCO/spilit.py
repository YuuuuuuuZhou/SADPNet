import os
import json
import random
import shutil

data_dir = '../Radiographs'
train_dir = '../train'
val_dir = '../val'
test_dir = '../test'
json_file = '../teeth_polygon.json'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

os.makedirs('../annotations', exist_ok=True)

images = [f for f in os.listdir(data_dir) if f.endswith(('jpg', 'png', 'jpeg', 'JPG'))]

random.shuffle(images)

total_images = len(images)
train_size = int(total_images * 0.7)
val_size = int(total_images * 0.1)

train_images = images[:train_size]
val_images = images[train_size:train_size + val_size]
test_images = images[train_size + val_size:]

for img in train_images:
    base_name, ext = os.path.splitext(img)
    new_name = base_name + ext.lower()
    shutil.copy(os.path.join(data_dir, img), os.path.join(train_dir, new_name))

for img in val_images:
    base_name, ext = os.path.splitext(img)
    new_name = base_name + ext.lower()
    shutil.copy(os.path.join(data_dir, img), os.path.join(val_dir, new_name))

for img in test_images:
    base_name, ext = os.path.splitext(img)
    new_name = base_name + ext.lower()
    shutil.copy(os.path.join(data_dir, img), os.path.join(test_dir, new_name))

if not os.path.exists(json_file):
    raise FileNotFoundError(f"The JSON file '{json_file}' does not exist.")

with open(json_file, 'r') as f:
    data = json.load(f)

instances_train = []
instances_val = []
instances_test = []


train_images_lower = [img.lower() for img in train_images]
val_images_lower = [img.lower() for img in val_images]
test_images_lower = [img.lower() for img in test_images]

for instance in data:
    external_id = instance['External ID'].lower()
    if external_id in train_images_lower:
        instances_train.append(instance)
    elif external_id in val_images_lower:
        instances_val.append(instance)
    elif external_id in test_images_lower:
        instances_test.append(instance)

if not instances_train or not instances_val or not instances_test:
    raise ValueError("One or more splits are empty.")

with open('../annotations/instances_train_source.json', 'w') as f:
    json.dump(instances_train, f)
with open('../annotations/instances_val_source.json', 'w') as f:
    json.dump(instances_val, f)
with open('../annotations/instances_test_source.json', 'w') as f:
    json.dump(instances_test, f)
