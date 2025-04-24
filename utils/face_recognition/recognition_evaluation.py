import pandas as pd
import os
from PIL import Image, ImageDraw
import io
import base64
from .face_detection import draw_box
import logging

# Thiết lập logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_recognition_pairs(num_pairs=4, model="ArcFace"):
    """Load và xử lý các cặp ảnh từ tập same và different.

    Args:
        num_pairs: Số cặp ảnh cần tải
        model: Tên mô hình (ArcFace, FaceNet, EfficientNet)
    """
    try:
        # Xác định đường dẫn file CSV dựa trên mô hình được chọn
        same_csv_path = f"result/Regnition/RetinaFace_{model}_same.csv"
        diff_csv_path = f"result/Regnition/RetinaFace_{model}_different.csv"

        logger.debug(f"Selected model: {model}")
        logger.debug(f"Reading CSV files from: {same_csv_path} and {diff_csv_path}")
        logger.debug(f"Current working directory: {os.getcwd()}")

        same_df = pd.read_csv(same_csv_path)
        diff_df = pd.read_csv(diff_csv_path)

        logger.debug(f"Sample of same_df paths:\n{same_df[['Image1', 'Image2']].head()}")
        logger.debug(f"Sample of diff_df paths:\n{diff_df[['Image1', 'Image2']].head()}")

        same_pairs = []
        diff_pairs = []

        for i in range(num_pairs):
            if i < len(same_df):
                same_row = same_df.iloc[i]
                logger.debug(f"Processing same pair {i+1}:")
                logger.debug(f"Image1: {same_row['Image1']}")
                logger.debug(f"Image2: {same_row['Image2']}")
                same_pairs.append(process_image_pair(same_row))

            if i < len(diff_df):
                diff_row = diff_df.iloc[i]
                logger.debug(f"Processing different pair {i+1}:")
                logger.debug(f"Image1: {diff_row['Image1']}")
                logger.debug(f"Image2: {diff_row['Image2']}")
                diff_pairs.append(process_image_pair(diff_row))

        return same_pairs, diff_pairs
    except Exception as e:
        logger.error(f"Error in load_recognition_pairs: {str(e)}", exc_info=True)
        return [], []

def process_image_pair(row):
    """Xử lý một cặp ảnh và trả về thông tin cần thiết."""
    def load_and_draw_box(image_path, x, y, w, h):
        try:
            # Kiểm tra xem đường dẫn có phải là đường dẫn tuyệt đối không
            abs_path = os.path.abspath(image_path)
            logger.debug(f"Attempting to load image from: {abs_path}")
            logger.debug(f"File exists: {os.path.exists(abs_path)}")

            if not os.path.exists(abs_path):
                # Thử tìm trong thư mục data
                alt_path = os.path.join('data', os.path.basename(image_path))
                alt_path = os.path.abspath(alt_path)
                logger.debug(f"Trying alternative path: {alt_path}")
                logger.debug(f"Alternative path exists: {os.path.exists(alt_path)}")
                if os.path.exists(alt_path):
                    image_path = alt_path

            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            draw_box(draw, [x, y, x+w, y+h], "red")

            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}", exc_info=True)
            return None

    try:
        return {
            'image1': load_and_draw_box(row['Image1'], row['X1'], row['Y1'], row['W1'], row['H1']),
            'image2': load_and_draw_box(row['Image2'], row['X2'], row['Y2'], row['W2'], row['H2']),
            'similarity': f"{row['Similarity']:.4f}",
            'processing_time': f"{row['TotalProcessingTime']:.4f}"
        }
    except Exception as e:
        logger.error(f"Error in process_image_pair: {str(e)}", exc_info=True)
        return {
            'image1': None,
            'image2': None,
            'similarity': 'N/A',
            'processing_time': 'N/A'
        }
