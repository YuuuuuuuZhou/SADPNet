import json
import cv2
import numpy as np
import os

annotation_path = '../annotations/instances_train_source.json'
image_dir = '../train'
output_annotation_path = '../annotations/instances_train_smooth.json'


with open(annotation_path, 'r') as f:
    annotations = json.load(f)

for idx, item in enumerate(annotations):
    external_id = item.get("External ID")
    image_path = os.path.join(image_dir, external_id)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    height, width, _ = image.shape

    objects = item.get("Label", {}).get("objects", [])
    for obj_idx, obj in enumerate(objects):
        title = obj['title']
        polygons = obj.get('polygons', [])
        longest_polygon = max(polygons, key=len, default=None)

        if longest_polygon:
            mask = np.zeros((height, width), dtype=np.uint8)
            longest_polygon = np.array(longest_polygon, np.int32)
            cv2.fillPoly(mask, [longest_polygon], 255)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            smoothed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            smoothed_mask = cv2.GaussianBlur(smoothed_mask, (29, 29), 0)
            _, binary_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                obj['bounding box'] = [y, x, y + h, x + w]

                smooth_polygons = [contour.reshape(-1, 2).tolist() for contour in contours]
                obj['polygons'] = smooth_polygons


with open(output_annotation_path, 'w') as f:
    json.dump(annotations, f)

print(f"Updated annotations saved to {output_annotation_path}")
