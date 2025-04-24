import os
import pandas as pd
from PIL import Image, ImageDraw
import io
import base64
from .face_detection import load_annotations, draw_box

# Đường dẫn đến thư mục dữ liệu
DATA_DIR = "./data/image_customer"
CUSTOMER_ROOT = os.path.join(DATA_DIR)

def get_detection_methods():
    """Trả về danh sách các phương pháp nhận diện có sẵn."""
    return {
        "MTCNN": "MTCNN",
        "Yolo": "Yolo",
        "Haar": "Haar Cascade",
        "Retina": "RetinaFace"
    }

def get_csv_files():
    """Trả về danh sách các file CSV tương ứng với các phương pháp nhận diện."""
    return {
        "MTCNN": "./result/MTCNN_face_detection.csv",
        "Yolo": "./result/Yolo_face_detection.csv",
        "Haar": "./result/Haar_face_detection.csv",
        "Retina": "./result/RetinaFace_face_detection.csv"
    }

def load_predictions(detection_method):
    """Đọc dữ liệu từ file CSV tương ứng với phương pháp nhận diện."""
    csv_files = get_csv_files()

    try:
        predictions_df = pd.read_csv(csv_files[detection_method])
        # Đảm bảo tên cột phù hợp
        if 'filename' in predictions_df.columns and 'file_name' not in predictions_df.columns:
            predictions_df = predictions_df.rename(columns={'filename': 'file_name'})
        return predictions_df, None
    except Exception as e:
        return None, f"Không thể đọc file {csv_files[detection_method]}: {e}"

def calculate_stats(predictions_df):
    """Tính toán các thống kê từ dữ liệu dự đoán."""
    try:
        if predictions_df.empty:
            return {
                'iou': {'mean': '0.0000', 'min': '0.0000', 'max': '0.0000', 'std': '0.0000'},
                'center_dist': {'mean': '0.00', 'min': '0.00', 'max': '0.00', 'std': '0.00'},
                'time': {'mean': '0.0000', 'min': '0.0000', 'max': '0.0000', 'std': '0.0000'}
            }

        # Đảm bảo các cột tồn tại
        required_columns = ['IoU', 'center_distance', 'inference_time']
        for col in required_columns:
            if col not in predictions_df.columns:
                raise ValueError(f"Missing required column: {col}")

        return {
            'iou': {
                'mean': f"{predictions_df['IoU'].mean():.4f}",
                'min': f"{predictions_df['IoU'].min():.4f}",
                'max': f"{predictions_df['IoU'].max():.4f}",
                'std': f"{predictions_df['IoU'].std():.4f}"
            },
            'center_dist': {
                'mean': f"{predictions_df['center_distance'].mean():.2f}",
                'min': f"{predictions_df['center_distance'].min():.2f}",
                'max': f"{predictions_df['center_distance'].max():.2f}",
                'std': f"{predictions_df['center_distance'].std():.2f}"
            },
            'time': {
                'mean': f"{predictions_df['inference_time'].mean():.4f}",
                'min': f"{predictions_df['inference_time'].min():.4f}",
                'max': f"{predictions_df['inference_time'].max():.4f}",
                'std': f"{predictions_df['inference_time'].std():.4f}"
            }
        }
    except Exception as e:
        print(f"Error calculating stats: {e}")
        # Trả về giá trị mặc định nếu có lỗi
        return {
            'iou': {'mean': 'N/A', 'min': 'N/A', 'max': 'N/A', 'std': 'N/A'},
            'center_dist': {'mean': 'N/A', 'min': 'N/A', 'max': 'N/A', 'std': 'N/A'},
            'time': {'mean': 'N/A', 'min': 'N/A', 'max': 'N/A', 'std': 'N/A'}
        }

def calculate_all_models_stats():
    """Tính toán thống kê cho tất cả các mô hình."""
    all_stats = {}
    detection_methods = get_detection_methods()
    csv_files = get_csv_files()
    
    for method_key, method_name in detection_methods.items():
        try:
            predictions_df = pd.read_csv(csv_files[method_key])
            
            # Tính toán thống kê
            zero_iou_count = len(predictions_df[predictions_df['IoU'] == 0])
            poor_iou_count = len(predictions_df[(predictions_df['IoU'] > 0) & (predictions_df['IoU'] < 0.5)])

            all_stats[method_key] = {
                'name': method_name,
                'iou_mean': f"{predictions_df['IoU'].mean():.4f}",
                'center_dist_mean': f"{predictions_df['center_distance'].mean():.2f}",
                'time_mean': f"{predictions_df['inference_time'].mean():.4f}",
                'zero_iou_count': zero_iou_count,
                'poor_iou_count': poor_iou_count,
                'zero_iou_percent': f"{zero_iou_count / len(predictions_df) * 100:.1f}%",
                'poor_iou_percent': f"{poor_iou_count / len(predictions_df) * 100:.1f}%"
            }
        except Exception as e:
            print(f"Không thể đọc file {csv_files[method_key]}: {e}")
            all_stats[method_key] = {
                'name': method_name,
                'iou_mean': 'N/A',
                'center_dist_mean': 'N/A',
                'time_mean': 'N/A',
                'zero_iou_count': 0,
                'poor_iou_count': 0,
                'zero_iou_percent': '0.0%',
                'poor_iou_percent': '0.0%'
            }

    return all_stats

def get_customer_folders(page, customers_per_page=4):
    """Lấy danh sách thư mục khách hàng cho trang hiện tại."""
    # Lấy danh sách khách hàng
    customer_folders = [d for d in os.listdir(CUSTOMER_ROOT)
                       if os.path.isdir(os.path.join(CUSTOMER_ROOT, d)) and d.isdigit()]
    customer_folders.sort(key=lambda x: int(x))

    # Tính tổng số trang
    total_pages = (len(customer_folders) + customers_per_page - 1) // customers_per_page

    # Lấy khách hàng cho trang hiện tại
    start_idx = page * customers_per_page
    end_idx = min(start_idx + customers_per_page, len(customer_folders))
    current_customers = customer_folders[start_idx:end_idx]

    return current_customers, total_pages

def create_customers_data(customer_folders, predictions_df):
    """Tạo dữ liệu khách hàng cho hiển thị."""
    customers_data = []

    # Đọc groundtruth từ file annotations.xml
    annotations_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'annotations.xml'))
    print(f"Loading annotations from: {annotations_path}")
    print(f"File exists: {os.path.isfile(annotations_path)}")

    # Nếu file annotations.xml tồn tại, sử dụng nó để lấy groundtruth
    if os.path.isfile(annotations_path):
        annotations = load_annotations(annotations_path)
        print(f"Loaded {len(annotations)} annotations from XML file")
    else:
        # Nếu không tìm thấy file annotations.xml, tạo groundtruth giả từ dữ liệu trong file CSV
        print("Annotations file not found. Creating fake annotations from CSV data.")
        annotations = {}
        for _, row in predictions_df.iterrows():
            rel_path = row['file_name']
            if all(col in row.index for col in ['x1', 'y1', 'x2', 'y2']):
                # Tạo groundtruth là hình chữ nhật hơi lớn hơn prediction
                x1 = float(row['x1']) - 5
                y1 = float(row['y1']) - 5
                x2 = float(row['x2']) + 5
                y2 = float(row['y2']) + 5
                annotations[rel_path] = {'box': [x1, y1, x2, y2]}
        print(f"Created {len(annotations)} fake annotations")

    for customer_folder in customer_folders:
        customer_id = int(customer_folder)
        customer_data = {'id': customer_id, 'images': []}

        image_dir = os.path.join(CUSTOMER_ROOT, customer_folder)
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

            if len(images) == 4:
                for img_name in sorted(images):
                    rel_path = f"image_customer/{customer_folder}/{img_name}"
                    img_path = os.path.join(CUSTOMER_ROOT, customer_folder, img_name)

                    if os.path.isfile(img_path):
                        # Mở ảnh bằng PIL để vẽ prediction
                        img = Image.open(img_path)
                        draw = ImageDraw.Draw(img)

                        # Vẽ ground truth (green) nếu có
                        if rel_path in annotations:
                            gt_box = annotations[rel_path]['box']
                            draw_box(draw, gt_box, "green")

                        # Lấy metrics và vẽ prediction (red) nếu có
                        pred_row = predictions_df[predictions_df['file_name'] == rel_path]
                        metrics = {}

                        if not pred_row.empty:
                            # Vẽ prediction box (red)
                            if all(col in pred_row.columns for col in ['x1', 'y1', 'x2', 'y2']):
                                pred_box = [
                                    pred_row['x1'].iloc[0],
                                    pred_row['y1'].iloc[0],
                                    pred_row['x2'].iloc[0],
                                    pred_row['y2'].iloc[0]
                                ]
                                draw_box(draw, pred_box, "red")

                            # Lấy metrics
                            metrics = {
                                'iou': f"{pred_row['IoU'].iloc[0]:.3f}",
                                'center_dist': f"{pred_row['center_distance'].iloc[0]:.1f}",
                                'inference_time': f"{pred_row['inference_time'].iloc[0]:.3f}"
                            }

                        # Chuyển ảnh thành base64
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        buf.seek(0)
                        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

                        customer_data['images'].append({
                            'name': img_name,
                            'base64': img_base64,
                            'metrics': metrics
                        })

        customers_data.append(customer_data)

    return customers_data

def create_customer_view_data(page, customers_per_page=8):
    """Tạo dữ liệu cho trang xem khách hàng."""
    # Lấy danh sách khách hàng
    customer_folders = [d for d in os.listdir(CUSTOMER_ROOT)
                       if os.path.isdir(os.path.join(CUSTOMER_ROOT, d)) and d.isdigit()]
    customer_folders.sort(key=lambda x: int(x))

    # Tính tổng số trang
    total_pages = (len(customer_folders) + customers_per_page - 1) // customers_per_page

    # Lấy khách hàng cho trang hiện tại
    start_idx = page * customers_per_page
    end_idx = min(start_idx + customers_per_page, len(customer_folders))
    current_customers = customer_folders[start_idx:end_idx]

    # Tạo dữ liệu khách hàng
    customers_data = []
    for customer_folder in current_customers:
        customer_id = int(customer_folder)
        customer_data = {'id': customer_id, 'images': []}

        image_dir = os.path.join(CUSTOMER_ROOT, customer_folder)
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

            for img_name in sorted(images):
                img_path = os.path.join(CUSTOMER_ROOT, customer_folder, img_name)

                if os.path.isfile(img_path):
                    # Chuyển ảnh thành base64
                    with open(img_path, "rb") as img_file:
                        img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

                    customer_data['images'].append({
                        'name': img_name,
                        'base64': img_base64
                    })

        customers_data.append(customer_data)

    return customers_data, total_pages






