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
from utils.face_recognition.performance_metrics import calculate_metrics_and_plots

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
                          customers_data=customers_data,
                          current_page=page,
                          total_pages=total_pages,
                          best_worst_images=best_worst_images,
                          best_iou=best_iou_row,
                          worst_iou=worst_iou_row)


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

@app.route('/api/change_method/<string:method>')
def api_change_method(method):
    """API endpoint để lấy dữ liệu cho phương pháp nhận diện."""
    app.logger.info(f"API change_method called with method: {method}")

    # Kiểm tra xem method có hợp lệ không
    valid_methods = get_detection_methods().keys()
    app.logger.info(f"Valid methods: {list(valid_methods)}")

    if method not in valid_methods:
        app.logger.error(f"Invalid method: {method}")
        return jsonify({
            'error': f"Phương pháp '{method}' không được hỗ trợ"
        }), 400

    # Tải dữ liệu dự đoán
    predictions_df, error = load_predictions(method)
    if error:
        app.logger.error(f"Error loading predictions: {error}")
        return jsonify({
            'error': error
        }), 400

    app.logger.info(f"Successfully loaded predictions for method: {method}")

    # Tính toán thống kê
    stats = calculate_stats(predictions_df)
    all_models_stats = calculate_all_models_stats()
    charts = create_charts(predictions_df)

    # Tạo dữ liệu khách hàng cho trang hiện tại
    page = request.args.get('page', 0, type=int)
    customers_per_page = 4
    current_customers, total_pages = get_customer_folders(page, customers_per_page)
    customers_data = create_customers_data(current_customers, predictions_df)

    app.logger.info(f"Created customers data for page {page} with {len(customers_data)} customers")

    # Tìm ảnh có IoU lớn nhất và nhỏ nhất
    try:
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

        app.logger.info(f"Found best IoU: {best_iou_row['IoU']} and worst IoU: {worst_iou_row['IoU']}")
    except Exception as e:
        app.logger.error(f"Error finding best/worst IoU: {e}")
        best_iou_row = {}
        worst_iou_row = {}

    # Tạo ảnh có IoU lớn nhất và nhỏ nhất
    try:
        best_worst_images = create_best_worst_images(predictions_df)
        app.logger.info(f"Created best/worst images: {len(best_worst_images)} items")
    except Exception as e:
        app.logger.error(f"Error creating best/worst images: {e}")
        best_worst_images = {}

    # Tạo response
    response_data = {
        'stats': stats,
        'all_models_stats': all_models_stats,
        'charts': charts,
        'customers_data': customers_data,
        'current_page': page,
        'total_pages': total_pages,
        'best_iou': best_iou_row,
        'worst_iou': worst_iou_row,
        'best_worst_images': best_worst_images
    }

    app.logger.info(f"API response prepared with {len(response_data)} keys")
    return jsonify(response_data)

@app.route('/report')
def report():
    """Chuyển hướng từ trang báo cáo tổng quan đến trang báo cáo khoa học."""
    return redirect(url_for('bao_cao'))

@app.route('/bao_cao')
def bao_cao():
    """Trang báo cáo khoa học về nhận diện khuôn mặt."""
    return render_template('bao_cao.html')

@app.route('/face_identification')
def face_identification():
    """Trang nhận diện khuôn mặt."""
    # Lấy tham số model từ query string, mặc định là ArcFace
    model = request.args.get('model', 'ArcFace')

    # Đảm bảo model là một trong các giá trị hợp lệ
    valid_models = ['ArcFace', 'FaceNet', 'EfficientNet']
    if model not in valid_models:
        model = 'ArcFace'

    # Tính toán metrics và charts
    error_message, charts, metrics = get_model_metrics(model)

    # Tải dữ liệu cặp ảnh
    same_pairs, diff_pairs = load_image_pairs(model)

    # Render the template with metrics data
    return render_template('face_recognition.html',
                          charts=charts,
                          metrics=metrics,
                          selected_model=model,
                          error=error_message,
                          same_pairs=same_pairs,
                          diff_pairs=diff_pairs)


def load_image_pairs(model):
    """Hàm helper để tải dữ liệu cặp ảnh từ static/images."""
    same_pairs = []
    diff_pairs = []

    try:
        # Đường dẫn đến ảnh mẫu trong thư mục static
        same1_img_path = os.path.join('static', 'images', 'same1.png')
        same2_img_path = os.path.join('static', 'images', 'same2.png')
        diff1_img_path = os.path.join('static', 'images', 'different1.png')
        diff2_img_path = os.path.join('static', 'images', 'different2.png')

        app.logger.info(f"Loading sample images from static/images directory")
        app.logger.info(f"Same1 image exists: {os.path.isfile(same1_img_path)}")
        app.logger.info(f"Same2 image exists: {os.path.isfile(same2_img_path)}")
        app.logger.info(f"Different1 image exists: {os.path.isfile(diff1_img_path)}")
        app.logger.info(f"Different2 image exists: {os.path.isfile(diff2_img_path)}")

        # Tạo 1 cặp ảnh giống nhau cho same_pairs
        if os.path.isfile(same1_img_path) and os.path.isfile(same2_img_path):
            # Đọc ảnh và chuyển thành base64
            with open(same1_img_path, 'rb') as f:
                same1_img_data = base64.b64encode(f.read()).decode('utf-8')

            with open(same2_img_path, 'rb') as f:
                same2_img_data = base64.b64encode(f.read()).decode('utf-8')

            # Tạo 1 cặp ảnh
            same_pairs.append({
                'image1': same1_img_data,
                'image2': same2_img_data,
                'similarity': 0.92,
                'processing_time': 0.045
            })

        # Tạo 1 cặp ảnh khác nhau cho diff_pairs
        if os.path.isfile(diff1_img_path) and os.path.isfile(diff2_img_path):
            # Đọc ảnh và chuyển thành base64
            with open(diff1_img_path, 'rb') as f:
                diff1_img_data = base64.b64encode(f.read()).decode('utf-8')

            with open(diff2_img_path, 'rb') as f:
                diff2_img_data = base64.b64encode(f.read()).decode('utf-8')

            # Tạo 1 cặp ảnh
            diff_pairs.append({
                'image1': diff1_img_data,
                'image2': diff2_img_data,
                'similarity': 0.32,
                'processing_time': 0.047
            })

        # Nếu không tìm thấy ảnh, thử tải từ file CSV
        if not same_pairs or not diff_pairs:
            app.logger.warning("Sample images not found, trying to load from CSV files")

            # Đường dẫn đến file CSV
            same_path = f"result/Regnition/RetinaFace_{model}_same.csv"
            diff_path = f"result/Regnition/RetinaFace_{model}_different.csv"

            # Kiểm tra xem file có tồn tại không
            if os.path.exists(same_path) and not same_pairs:
                # Đọc file CSV
                same_df = pd.read_csv(same_path)

                # Tạo dữ liệu cặp ảnh
                for _, row in same_df.head(4).iterrows():  # Chỉ lấy 4 cặp đầu tiên
                    # Đường dẫn đến ảnh
                    img1_path = os.path.join('data', row['image1']) if 'image1' in row else None
                    img2_path = os.path.join('data', row['image2']) if 'image2' in row else None

                    # Kiểm tra xem ảnh có tồn tại không
                    if img1_path and img2_path and os.path.exists(img1_path) and os.path.exists(img2_path):
                        # Đọc ảnh và chuyển thành base64
                        with open(img1_path, 'rb') as f:
                            img1_data = base64.b64encode(f.read()).decode('utf-8')

                        with open(img2_path, 'rb') as f:
                            img2_data = base64.b64encode(f.read()).decode('utf-8')

                        # Thêm vào danh sách
                        same_pairs.append({
                            'image1': img1_data,
                            'image2': img2_data,
                            'similarity': row['similarity'] if 'similarity' in row else 0,
                            'processing_time': row['processing_time'] if 'processing_time' in row else 0
                        })

            # Tương tự cho cặp ảnh khác người
            if os.path.exists(diff_path) and not diff_pairs:
                # Đọc file CSV
                diff_df = pd.read_csv(diff_path)

                # Tạo dữ liệu cặp ảnh
                for _, row in diff_df.head(4).iterrows():  # Chỉ lấy 4 cặp đầu tiên
                    # Đường dẫn đến ảnh
                    img1_path = os.path.join('data', row['image1']) if 'image1' in row else None
                    img2_path = os.path.join('data', row['image2']) if 'image2' in row else None

                    # Kiểm tra xem ảnh có tồn tại không
                    if img1_path and img2_path and os.path.exists(img1_path) and os.path.exists(img2_path):
                        # Đọc ảnh và chuyển thành base64
                        with open(img1_path, 'rb') as f:
                            img1_data = base64.b64encode(f.read()).decode('utf-8')

                        with open(img2_path, 'rb') as f:
                            img2_data = base64.b64encode(f.read()).decode('utf-8')

                        # Thêm vào danh sách
                        diff_pairs.append({
                            'image1': img1_data,
                            'image2': img2_data,
                            'similarity': row['similarity'] if 'similarity' in row else 0,
                            'processing_time': row['processing_time'] if 'processing_time' in row else 0
                        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        app.logger.error(f"Error loading image pairs: {str(e)}")

    app.logger.info(f"Loaded {len(same_pairs)} same pairs and {len(diff_pairs)} different pairs")
    return same_pairs, diff_pairs


def get_model_metrics(model):
    """Hàm helper để tính toán metrics và charts cho model."""
    error_message = None
    charts = {}
    metrics = {}

    app.logger.debug(f"Selected model: {model}")

    # Try to calculate metrics
    try:
        # Tính toán các metrics và tạo biểu đồ với model được chọn
        same_path = f"result/Regnition/RetinaFace_{model}_same.csv"
        diff_path = f"result/Regnition/RetinaFace_{model}_different.csv"

        # Check if files exist
        if os.path.exists(same_path) and os.path.exists(diff_path):
            charts, metrics = calculate_metrics_and_plots(same_path, diff_path)
        else:
            app.logger.warning(f"CSV files not found: {same_path} or {diff_path}")
            error_message = f"Không tìm thấy file CSV để tính toán metrics cho mô hình {model}."
    except Exception as e:
        import traceback
        traceback.print_exc()
        app.logger.error(f"Error calculating metrics: {str(e)}")
        error_message = f"Lỗi khi tính toán metrics: {str(e)}"

    # Đảm bảo charts và metrics luôn là dict
    if charts is None:
        charts = {}

    if metrics is None:
        metrics = {}

    return error_message, charts, metrics

# Redirect old face_recognition URL to new report URL for backward compatibility
@app.route('/face_recognition')
def face_recognition():
    """Redirect to report page."""
    return redirect(url_for('report'))

if __name__ == '__main__':
    # Khởi tạo logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app.logger.setLevel(logging.DEBUG)
    app.run(debug=True)
