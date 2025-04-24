# Ứng dụng nhận diện khuôn mặt (Flask)

Ứng dụng web Flask để hiển thị và phân tích kết quả nhận diện khuôn mặt từ các thuật toán khác nhau.

## Tính năng

- Hiển thị kết quả nhận diện khuôn mặt từ các thuật toán khác nhau (MTCNN, Yolo, Haar Cascade, RetinaFace)
- Phân tích các chỉ số IoU, khoảng cách tâm, thời gian xử lý
- Hiển thị biểu đồ phân tích
- Hiển thị ảnh khách hàng với bounding box
- Phân trang cho danh sách khách hàng

## Cài đặt

1. Clone repository:
```
git clone <repository-url>
cd Report
```

2. Tạo môi trường ảo:
```
python -m venv venv
```

3. Kích hoạt môi trường ảo:
```
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Cài đặt các thư viện cần thiết:
```
pip install -r requirements.txt
```

## Chạy ứng dụng

1. Kích hoạt môi trường ảo (nếu chưa kích hoạt):
```
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. Chạy ứng dụng:
```
python run.py
```

3. Mở trình duyệt và truy cập địa chỉ:
```
http://127.0.0.1:5000/
```

## Cấu trúc thư mục

```
Report/
├── app.py                  # File chính của ứng dụng Flask
├── run.py                  # File để chạy ứng dụng Flask
├── requirements.txt        # Danh sách các thư viện cần thiết
├── static/                 # Thư mục chứa các file tĩnh
│   ├── css/                
│   │   └── style.css       
│   ├── js/                 
│   │   └── main.js         
├── templates/              # Thư mục chứa các template HTML
│   ├── base.html           
│   ├── index.html          
│   ├── face_detection.html 
│   └── customer_view.html  
├── data/                   # Thư mục dữ liệu
├── result/                 # Thư mục kết quả
└── utils/                  # Thư mục tiện ích
    ├── __init__.py         
    └── face_recognition/   
        ├── __init__.py     
        ├── data_processing.py  # Xử lý dữ liệu
        ├── visualization.py    # Tạo biểu đồ và hình ảnh
        └── face_detection.py   # Nhận diện khuôn mặt
```

## Các trang

1. **Trang chủ** (`/`): Trang chủ của ứng dụng.
2. **Nhận diện khuôn mặt** (`/face_detection`): Hiển thị kết quả nhận diện khuôn mặt từ các thuật toán khác nhau.
3. **Xem khách hàng** (`/customer_view`): Hiển thị danh sách khách hàng.

## Các thuật toán nhận diện khuôn mặt

1. **MTCNN**: Multi-task Cascaded Convolutional Networks
2. **Yolo**: You Only Look Once
3. **Haar Cascade**: Phương pháp phát hiện đối tượng Viola-Jones
4. **RetinaFace**: Phương pháp phát hiện khuôn mặt dựa trên RetinaNet

## Các chỉ số đánh giá

1. **IoU (Intersection over Union)**: Tỷ lệ giao trên hợp của bounding box dự đoán và ground truth.
2. **Khoảng cách tâm**: Khoảng cách Euclidean giữa tâm của bounding box dự đoán và ground truth.
3. **Thời gian xử lý**: Thời gian xử lý của thuật toán nhận diện khuôn mặt.

## So sánh với Streamlit

### Ưu điểm của Flask:

1. **Tùy biến cao**: Flask cho phép bạn tùy biến giao diện người dùng và cấu trúc ứng dụng theo ý muốn.
2. **Hiệu suất tốt hơn**: Flask thường có hiệu suất tốt hơn Streamlit, đặc biệt là với các ứng dụng lớn.
3. **Kiểm soát tốt hơn**: Bạn có thể kiểm soát chi tiết cách ứng dụng hoạt động, từ routing đến cách dữ liệu được xử lý.
4. **Dễ dàng mở rộng**: Flask dễ dàng mở rộng với các tính năng mới và tích hợp với các công nghệ khác.
5. **Phù hợp cho production**: Flask phù hợp hơn cho các ứng dụng production, với nhiều tùy chọn triển khai.

### Nhược điểm của Flask so với Streamlit:

1. **Phức tạp hơn**: Flask đòi hỏi nhiều code hơn và kiến thức về web development.
2. **Thời gian phát triển lâu hơn**: Phát triển ứng dụng với Flask thường mất nhiều thời gian hơn so với Streamlit.
3. **Cần viết HTML/CSS/JS**: Bạn cần viết code HTML, CSS, và JavaScript để tạo giao diện người dùng.
4. **Không có tính năng reactive**: Flask không có tính năng reactive như Streamlit, nên bạn cần xử lý các tương tác người dùng thủ công.
