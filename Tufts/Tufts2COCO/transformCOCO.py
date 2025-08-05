import json

with open('../annotations/instances_train_smooth.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

coco_format = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": i,
            "name": (
                str(18 - (i - 1)) if 1 <= i <= 8 else
                str(21 + (i - 9)) if 9 <= i <= 16 else
                str(38 - (i - 17)) if 17 <= i <= 24 else
                str(41 + (i - 25)) if 25 <= i <= 32 else
                str(55 - (i - 33)) if 33 <= i <= 37 else
                str(61 + (i - 38)) if 38 <= i <= 42 else
                str(75 - (i - 43)) if 43 <= i <= 47 else
                str(81 + (i - 48))
            ),
            "supercategory": "teeth"
        }
        for i in range(1, 53)
    ]
}

annotation_id = 1

def get_category_id(title):
    if title.isdigit():
        return int(title)
    elif title.isalpha() and 'A' <= title <= 'T':
        return ord(title) - ord('A') + 33
    else:
        raise ValueError(f"Invalid title value: {title}")

def flatten_polygons(polygons):
    flat_polygon = []
    for polygon in polygons:
        flat_polygon.extend(polygon)
    return flat_polygon

for item in data:
    external_id = item['External ID']
    file_name = external_id
    image_id = int(external_id.split('.')[0])
    coco_format['images'].append({
        'file_name': file_name,
        'height': 840,
        'width': 1615,
        'id': image_id
    })

    for obj in item['Label']['objects']:
        segmentation = flatten_polygons(obj['polygons'])
        segmentation = [coordinate for sublist in segmentation for coordinate in sublist]
        bbox = obj['bounding box']
        y0, x0, y1, x1 = bbox
        w, h = x1 - x0, y1 - y0
        area = w * h
        category_id = get_category_id(obj['title'])
        coco_format['annotations'].append({
            'segmentation': [segmentation],
            'area': area,
            'iscrowd': 0,
            'image_id': image_id,
            'bbox': [x0, y0, w, h],
            'category_id': category_id,
            'id': annotation_id
        })
        annotation_id += 1

with open('../annotations/instances_train.json', 'w', encoding='utf-8') as f:
    json.dump(coco_format, f, ensure_ascii=False)

print("Conversion complete! ")
