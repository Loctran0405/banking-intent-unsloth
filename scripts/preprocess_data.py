import os
import pandas as pd
from datasets import load_dataset

def main():
    print("Đang tải dataset BANKING77...")
    # Tạo thư mục nếu chưa có
    os.makedirs("sample_data", exist_ok=True)
    
    # Tải dataset gốc
    dataset = load_dataset("PolyAI/banking77")
    
    # Lấy hàm chuyển từ nhãn số (0, 1, 2...) sang nhãn chữ (ví dụ: card_lost)
    int2str = dataset['train'].features['label'].int2str
    
    # Lấy subset (1500 cho train, 300 cho test) để train nhanh trên T4
    train_ds = dataset['train'].shuffle(seed=42).select(range(1500))
    test_ds = dataset['test'].shuffle(seed=42).select(range(300))
    
    # Chuyển sang Pandas DataFrame
    df_train = train_ds.to_pandas()
    df_test = test_ds.to_pandas()
    
    # Đổi nhãn số thành nhãn chữ cho dễ học
    df_train['label'] = df_train['label'].apply(int2str)
    df_test['label'] = df_test['label'].apply(int2str)
    
    # Lưu file
    df_train.to_csv("sample_data/train.csv", index=False)
    df_test.to_csv("sample_data/test.csv", index=False)
    print("Đã lưu dữ liệu thành công vào thư mục sample_data/")

if __name__ == "__main__":
    main()