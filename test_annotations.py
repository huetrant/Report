import xml.etree.ElementTree as ET
import os

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

if __name__ == "__main__":
    # Đường dẫn đến file annotations.xml
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'annotations.xml')
    print(f"XML path: {xml_path}")
    print(f"File exists: {os.path.isfile(xml_path)}")

    try:
        annotations = load_annotations(xml_path)
        print(f"Total annotations: {len(annotations)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
