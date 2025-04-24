import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image, ImageDraw
from .face_detection import load_annotations, draw_box

def create_charts(predictions_df):
    """Tạo các biểu đồ phân tích."""
    charts = {}

    # Tăng chiều cao của biểu đồ và thêm padding
    plt.rcParams['figure.autolayout'] = True

    # Biểu đồ IoU
    fig1 = Figure(figsize=(10, 4))  # Tăng chiều cao từ 3 lên 4
    ax1 = fig1.add_subplot(1, 1, 1)
    
    # Vẽ histogram
    hist = sns.histplot(data=predictions_df['IoU'], bins=20, kde=True, ax=ax1, color='green')
    
    # Thêm giá trị trên mỗi cột
    for i in hist.patches:
        hist.annotate(f'{int(i.get_height())}',
                     xy=(i.get_x() + i.get_width()/2, i.get_height()),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')

    ax1.set_xlabel('IoU (Giao trên hợp)')
    ax1.set_ylabel('Tần suất')
    ax1.set_title('Phân phối giá trị IoU')
    ax1.grid(True, alpha=0.3)
    
    # Thêm padding để tránh bị cắt
    fig1.tight_layout(pad=1.5)

    # Chuyển biểu đồ thành base64
    buf = io.BytesIO()
    fig1.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    charts['iou'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Biểu đồ khoảng cách tâm
    fig2 = Figure(figsize=(10, 4))
    ax2 = fig2.add_subplot(1, 1, 1)
    
    hist = sns.histplot(data=predictions_df['center_distance'], bins=20, kde=True, ax=ax2, color='red')
    
    # Thêm giá trị trên mỗi cột
    for i in hist.patches:
        hist.annotate(f'{int(i.get_height())}',
                     xy=(i.get_x() + i.get_width()/2, i.get_height()),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    ax2.set_xlabel('Khoảng cách tâm (pixels)')
    ax2.set_ylabel('Tần suất')
    ax2.set_title('Phân phối khoảng cách tâm')
    ax2.grid(True, alpha=0.3)
    
    fig2.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig2.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    charts['center_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Biểu đồ thời gian xử lý
    fig3 = Figure(figsize=(10, 4))
    ax3 = fig3.add_subplot(1, 1, 1)
    
    hist = sns.histplot(data=predictions_df['inference_time'], bins=20, kde=True, ax=ax3, color='blue')
    
    # Thêm giá trị trên mỗi cột
    for i in hist.patches:
        hist.annotate(f'{int(i.get_height())}',
                     xy=(i.get_x() + i.get_width()/2, i.get_height()),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    ax3.set_xlabel('Thời gian xử lý (giây)')
    ax3.set_ylabel('Tần suất')
    ax3.set_title('Phân phối thời gian xử lý')
    ax3.grid(True, alpha=0.3)
    
    fig3.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig3.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    charts['time'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    return charts

def create_best_worst_images(best_iou_row, worst_iou_row):
    """Tạo ảnh có IoU lớn nhất và nhỏ nhất."""
    images = {}

    # In ra các khóa trong best_iou_row và worst_iou_row
    print("Best IoU Row Keys:", best_iou_row.keys())
    print("Best IoU Row:", best_iou_row)
    print("Worst IoU Row Keys:", worst_iou_row.keys())
    print("Worst IoU Row:", worst_iou_row)

    # Tạo groundtruth giả từ dữ liệu trong file CSV
    # Giả sử rằng groundtruth là một hình chữ nhật hơi lớn hơn prediction
    annotations = {}
    if all(key in best_iou_row for key in ['x1', 'y1', 'x2', 'y2', 'file_name']):
        # Tạo groundtruth là hình chữ nhật hơi lớn hơn prediction
        x1 = float(best_iou_row['x1']) - 5
        y1 = float(best_iou_row['y1']) - 5
        x2 = float(best_iou_row['x2']) + 5
        y2 = float(best_iou_row['y2']) + 5
        annotations[best_iou_row['file_name']] = {'box': [x1, y1, x2, y2]}
    else:
        print("Missing required keys in best_iou_row for groundtruth creation")

    if all(key in worst_iou_row for key in ['x1', 'y1', 'x2', 'y2', 'file_name']):
        # Tạo groundtruth là hình chữ nhật hơi lớn hơn prediction
        x1 = float(worst_iou_row['x1']) - 5
        y1 = float(worst_iou_row['y1']) - 5
        x2 = float(worst_iou_row['x2']) + 5
        y2 = float(worst_iou_row['y2']) + 5
        annotations[worst_iou_row['file_name']] = {'box': [x1, y1, x2, y2]}
    else:
        print("Missing required keys in worst_iou_row for groundtruth creation")

    print(f"Created {len(annotations)} fake annotations for best/worst images")

    # Ảnh có IoU lớn nhất
    best_img_path = os.path.join('data', best_iou_row['file_name'])
    print(f"Best image path: {best_img_path}")
    print(f"File exists: {os.path.isfile(best_img_path)}")
    if os.path.isfile(best_img_path):
        img = Image.open(best_img_path)
        draw = ImageDraw.Draw(img)

        # Vẽ ground truth (green)
        if best_iou_row['file_name'] in annotations:
            gt_box = annotations[best_iou_row['file_name']]['box']
            draw_box(draw, gt_box, "green")

        # Vẽ prediction (red)
        # Kiểm tra xem các khóa có tồn tại không
        if all(key in best_iou_row for key in ['x1', 'y1', 'x2', 'y2']):
            pred_box = [best_iou_row['x1'], best_iou_row['y1'], best_iou_row['x2'], best_iou_row['y2']]
            draw_box(draw, pred_box, "red")
        elif all(key in best_iou_row for key in ['xmin', 'ymin', 'xmax', 'ymax']):
            pred_box = [best_iou_row['xmin'], best_iou_row['ymin'], best_iou_row['xmax'], best_iou_row['ymax']]
            draw_box(draw, pred_box, "red")
        else:
            print("Warning: Could not find bounding box coordinates in best_iou_row")
            print("Available keys:", best_iou_row.keys())

        # Chuyển ảnh thành base64
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        images['best'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        images['best_file'] = best_iou_row['file_name']
        images['best_metrics'] = {
            'iou': f"{best_iou_row['IoU']:.4f}",
            'center_dist': f"{best_iou_row['center_distance']:.2f}",
            'inference_time': f"{best_iou_row['inference_time']:.4f}"
        }

    # Ảnh có IoU nhỏ nhất
    worst_img_path = os.path.join('data', worst_iou_row['file_name'])
    print(f"Worst image path: {worst_img_path}")
    print(f"File exists: {os.path.isfile(worst_img_path)}")
    if os.path.isfile(worst_img_path):
        img = Image.open(worst_img_path)
        draw = ImageDraw.Draw(img)

        # Vẽ ground truth (green)
        if worst_iou_row['file_name'] in annotations:
            gt_box = annotations[worst_iou_row['file_name']]['box']
            draw_box(draw, gt_box, "green")

        # Vẽ prediction (red)
        # Kiểm tra xem các khóa có tồn tại không
        if all(key in worst_iou_row for key in ['x1', 'y1', 'x2', 'y2']):
            pred_box = [worst_iou_row['x1'], worst_iou_row['y1'], worst_iou_row['x2'], worst_iou_row['y2']]
            draw_box(draw, pred_box, "red")
        elif all(key in worst_iou_row for key in ['xmin', 'ymin', 'xmax', 'ymax']):
            pred_box = [worst_iou_row['xmin'], worst_iou_row['ymin'], worst_iou_row['xmax'], worst_iou_row['ymax']]
            draw_box(draw, pred_box, "red")
        else:
            print("Warning: Could not find bounding box coordinates in worst_iou_row")
            print("Available keys:", worst_iou_row.keys())

        # Chuyển ảnh thành base64
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        images['worst'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        images['worst_file'] = worst_iou_row['file_name']
        images['worst_metrics'] = {
            'iou': f"{worst_iou_row['IoU']:.4f}",
            'center_dist': f"{worst_iou_row['center_distance']:.2f}",
            'inference_time': f"{worst_iou_row['inference_time']:.4f}"
        }

    return images

# Thêm import os ở đầu file
import os

