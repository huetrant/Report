from flask import Flask, render_template, request, jsonify, url_for, redirect
import pandas as pd
import os
import base64

# Đường dẫn đến thư mục data
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'image_customer')

# Import các module từ utils
from utils.face_recognition.data_processing import (
    get_detection_methods, load_predictions, calculate_stats, calculate_all_models_stats,
    get_customer_folders, create_customers_data, create_customer_view_data
)
from utils.face_recognition.visualization import create_charts, create_best_worst_images
from utils.face_recognition.face_detection import load_annotations

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face_detection')
def face_detection():
    # Lấy phương pháp nhận diện từ query parameter, mặc định là MTCNN
    detection_method = request.args.get('method', 'MTCNN')

    # Các phương pháp nhận diện có sẵn
    detection_methods = get_detection_methods()

    # Đọc dữ liệu từ file CSV tương ứng
    predictions_df, error = load_predictions(detection_method)
    if error:
        return render_template('face_detection.html',
                              error=error,
                              detection_methods=detection_methods,
                              selected_method=detection_method)

    # Tính toán các thống kê cho mô hình hiện tại
    stats = calculate_stats(predictions_df)

    # Tính toán thống kê cho tất cả các mô hình để so sánh
    all_models_stats = calculate_all_models_stats()

    # Tạo các biểu đồ
    charts = create_charts(predictions_df)

    # Tìm ảnh có IoU lớn nhất và nhỏ nhất
    best_iou_idx = predictions_df['IoU'].idxmax()
    worst_iou_idx = predictions_df['IoU'].idxmin()

    # Tạo bản sao của DataFrame để tránh thay đổi dữ liệu gốc
    best_row_df = predictions_df.loc[[best_iou_idx]].copy()
    worst_row_df = predictions_df.loc[[worst_iou_idx]].copy()

    # Đổi tên cột nếu cần
    if 'x1' not in best_row_df.columns and 'xmin' in best_row_df.columns:
        best_row_df = best_row_df.rename(columns={'xmin': 'x1', 'ymin': 'y1', 'xmax': 'x2', 'ymax': 'y2'})
        worst_row_df = worst_row_df.rename(columns={'xmin': 'x1', 'ymin': 'y1', 'xmax': 'x2', 'ymax': 'y2'})

    # Chuyển đổi thành dictionary
    best_iou_row = best_row_df.iloc[0].to_dict()
    worst_iou_row = worst_row_df.iloc[0].to_dict()

    # In ra các khóa trong best_iou_row và worst_iou_row
    app.logger.debug("Best IoU Row Keys: %s", best_iou_row.keys())
    app.logger.debug("Best IoU Row: %s", best_iou_row)
    app.logger.debug("Worst IoU Row Keys: %s", worst_iou_row.keys())
    app.logger.debug("Worst IoU Row: %s", worst_iou_row)

    # Tạo ảnh có IoU lớn nhất và nhỏ nhất
    try:
        # Import các module cần thiết
        from PIL import Image, ImageDraw
        from utils.face_recognition.face_detection import draw_box
        import io

        # Tạo đường dẫn đến file ảnh có IoU lớn nhất và nhỏ nhất
        best_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', best_iou_row['file_name'])
        worst_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', worst_iou_row['file_name'])
        print(f"Best file path: {best_file_path}")
        print(f"Worst file path: {worst_file_path}")
        print(f"Best file exists: {os.path.isfile(best_file_path)}")
        print(f"Worst file exists: {os.path.isfile(worst_file_path)}")

        # Đọc groundtruth từ file annotations.xml
        annotations_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'annotations.xml')
        app.logger.debug(f"Loading annotations from: {annotations_path}")
        app.logger.debug(f"File exists: {os.path.isfile(annotations_path)}")

        # Đọc annotations từ file XML
        annotations = load_annotations(annotations_path)
        app.logger.debug(f"Loaded {len(annotations)} annotations from XML file")

        # Nếu không tìm thấy annotations, tạo groundtruth giả
        if not annotations:
            app.logger.debug("No annotations found in XML file. Creating fake annotations.")
            annotations = {}

            # Tạo groundtruth cho ảnh có IoU lớn nhất
            if all(key in best_iou_row for key in ['x1', 'y1', 'x2', 'y2']):
                # Tạo groundtruth là hình chữ nhật hơi lớn hơn prediction
                x1 = float(best_iou_row['x1']) - 5
                y1 = float(best_iou_row['y1']) - 5
                x2 = float(best_iou_row['x2']) + 5
                y2 = float(best_iou_row['y2']) + 5
                annotations[best_iou_row['file_name']] = {'box': [x1, y1, x2, y2]}

            # Tạo groundtruth cho ảnh có IoU nhỏ nhất
            if all(key in worst_iou_row for key in ['x1', 'y1', 'x2', 'y2']):
                # Tạo groundtruth là hình chữ nhật hơi lớn hơn prediction
                x1 = float(worst_iou_row['x1']) - 5
                y1 = float(worst_iou_row['y1']) - 5
                x2 = float(worst_iou_row['x2']) + 5
                y2 = float(worst_iou_row['y2']) + 5
                annotations[worst_iou_row['file_name']] = {'box': [x1, y1, x2, y2]}

        best_worst_images = {}

        # Kiểm tra xem file ảnh có tồn tại không
        if os.path.isfile(best_file_path):
            # Mở ảnh bằng PIL để vẽ prediction và groundtruth
            img = Image.open(best_file_path)
            draw = ImageDraw.Draw(img)

            # Vẽ ground truth (green) nếu có
            if best_iou_row['file_name'] in annotations:
                gt_box = annotations[best_iou_row['file_name']]['box']
                draw_box(draw, gt_box, "green")

            # Vẽ prediction (red)
            if all(key in best_iou_row for key in ['x1', 'y1', 'x2', 'y2']):
                pred_box = [best_iou_row['x1'], best_iou_row['y1'], best_iou_row['x2'], best_iou_row['y2']]
                draw_box(draw, pred_box, "red")

            # Chuyển ảnh thành base64
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            best_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Thêm vào best_worst_images
            best_worst_images['best'] = best_img_base64
            best_worst_images['best_file'] = best_iou_row['file_name']
            best_worst_images['best_metrics'] = {
                'iou': f"{best_iou_row['IoU']:.4f}",
                'center_dist': f"{best_iou_row['center_distance']:.2f}",
                'inference_time': f"{best_iou_row['inference_time']:.4f}"
            }

        if os.path.isfile(worst_file_path):
            # Mở ảnh bằng PIL để vẽ prediction và groundtruth
            img = Image.open(worst_file_path)
            draw = ImageDraw.Draw(img)

            # Vẽ ground truth (green) nếu có
            if worst_iou_row['file_name'] in annotations:
                gt_box = annotations[worst_iou_row['file_name']]['box']
                draw_box(draw, gt_box, "green")

            # Vẽ prediction (red)
            if all(key in worst_iou_row for key in ['x1', 'y1', 'x2', 'y2']):
                pred_box = [worst_iou_row['x1'], worst_iou_row['y1'], worst_iou_row['x2'], worst_iou_row['y2']]
                draw_box(draw, pred_box, "red")

            # Chuyển ảnh thành base64
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            worst_img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Thêm vào best_worst_images
            best_worst_images['worst'] = worst_img_base64
            best_worst_images['worst_file'] = worst_iou_row['file_name']
            best_worst_images['worst_metrics'] = {
                'iou': f"{worst_iou_row['IoU']:.4f}",
                'center_dist': f"{worst_iou_row['center_distance']:.2f}",
                'inference_time': f"{worst_iou_row['inference_time']:.4f}"
            }

        if not best_worst_images:
            print("Could not create best/worst images")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error creating best/worst images: {e}")
        best_worst_images = {}

    # Phân trang
    page = request.args.get('page', 0, type=int)
    customers_per_page = 4  # 2 hàng x 2 khách hàng

    # Lấy khách hàng cho trang hiện tại
    current_customers, total_pages = get_customer_folders(page, customers_per_page)

    # Tạo dữ liệu khách hàng
    # Sử dụng hàm create_customers_data đã được tạo sẵn
    customers_data = create_customers_data(current_customers, predictions_df)

    return render_template('face_detection.html',
                          detection_methods=detection_methods,
                          selected_method=detection_method,
                          stats=stats,
                          all_models_stats=all_models_stats,
                          charts=charts,
                          best_iou=best_iou_row,
                          worst_iou=worst_iou_row,
                          best_worst_images=best_worst_images,
                          customers_data=customers_data,
                          current_page=page,
                          total_pages=total_pages)


@app.route('/customer_view')
def customer_view():
    # Phân trang
    page = request.args.get('page', 0, type=int)
    customers_per_page = 8  # 2 hàng x 4 khách hàng

    # Tạo dữ liệu khách hàng
    customers_data, total_pages = create_customer_view_data(page, customers_per_page)

    return render_template('customer_view.html',
                          customers_data=customers_data,
                          current_page=page,
                          total_pages=total_pages)

@app.route('/change_method/<string:method>')
def change_method(method):
    """Đổi phương pháp nhận diện và chuyển hướng về trang face_detection."""
    return redirect(url_for('face_detection', method=method))

@app.route('/change_page/<int:page>')
def change_page(page):
    """Đổi trang và chuyển hướng về trang face_detection."""
    method = request.args.get('method', 'MTCNN')
    return redirect(url_for('face_detection', method=method, page=page))

if __name__ == '__main__':
    # Khởi tạo logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app.logger.setLevel(logging.DEBUG)
    app.run(debug=True)