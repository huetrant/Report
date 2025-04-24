import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def load_annotations(xml_path):
    """Load annotations từ file XML."""
    annotations = {}
    try:
        # Kiểm tra xem file có tồn tại không
        if not os.path.isfile(xml_path):
            print(f"Error: Annotations file not found at {xml_path}")
            return annotations

        print(f"Loading annotations from {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        print(f"Found {len(root.findall('.//image'))} images in XML file.")

        for image in root.findall('.//image'):
            filename = image.get('name')
            box_elem = image.find('.//box')
            if box_elem is not None:
                print(f"Found box for {filename} with attributes: {box_elem.attrib}")
                # Kiểm tra xem có các thuộc tính xtl, ytl, xbr, ybr không
                if all(attr in box_elem.attrib for attr in ['xtl', 'ytl', 'xbr', 'ybr']):
                    xmin = float(box_elem.get('xtl'))
                    ymin = float(box_elem.get('ytl'))
                    xmax = float(box_elem.get('xbr'))
                    ymax = float(box_elem.get('ybr'))
                    print(f"Using xtl, ytl, xbr, ybr attributes for {filename}")
                # Kiểm tra xem có các thuộc tính xmin, ymin, xmax, ymax không
                elif all(attr in box_elem.attrib for attr in ['xmin', 'ymin', 'xmax', 'ymax']):
                    xmin = float(box_elem.get('xmin'))
                    ymin = float(box_elem.get('ymin'))
                    xmax = float(box_elem.get('xmax'))
                    ymax = float(box_elem.get('ymax'))
                    print(f"Using xmin, ymin, xmax, ymax attributes for {filename}")
                else:
                    print(f"Warning: Box for {filename} does not have required attributes.")
                    continue

                annotations[filename] = {
                    'box': [xmin, ymin, xmax, ymax]
                }
            else:
                print(f"No box found for {filename}")

        print(f"Loaded {len(annotations)} annotations from XML file.")
        # In ra một vài annotations để kiểm tra
        if annotations:
            print("Sample annotations:")
            for i, (filename, annotation) in enumerate(list(annotations.items())[:3]):
                print(f"  {i+1}. {filename}: {annotation}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading annotations: {e}")

    return annotations

# Thêm import os
import os

def draw_box(draw, box, color):
    """Vẽ bounding box lên ảnh."""
    draw.rectangle(
        [(box[0], box[1]), (box[2], box[3])],
        outline=color,
        width=5
    )

def calculate_iou(box1, box2):
    """Tính IoU giữa hai bounding box."""
    # Tọa độ của hình chữ nhật giao nhau
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Kiểm tra xem hai hình chữ nhật có giao nhau không
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Tính diện tích giao nhau
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Tính diện tích của từng hình chữ nhật
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Tính IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou

def calculate_center_distance(box1, box2):
    """Tính khoảng cách giữa tâm của hai bounding box."""
    # Tính tọa độ tâm của box1
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2

    # Tính tọa độ tâm của box2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2

    # Tính khoảng cách Euclidean
    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

    return distance
