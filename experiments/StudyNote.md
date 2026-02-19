1. Hướng xử lý khi Label không cân bằng
    1. Data - Level
        1. Over Sampling: Dịch qua tiếng khác xong dịch ngược lại
        2. Under Sampling: Giảm số mẫu (loại bớt các mẫu có label nhiều hơn)
        3. Data Augmentation: Dùng thêm LLM để sinh thêm mẫu cho nhãn thiếu
    2. Model - Level
        1. Dùng Weighted Cross Entropy: Trả giá đắt hơn nếu đoán sai nhãn dễ
        2. Focal Loss: Ép mô hình tập trung vào các nhãn khó

    * Bài này mình chọn Weighted Cross Entropy
        - Các nhãn quan trọng như nhau
        - Không cần ép học các case khó
        - Đang cần phân loại
