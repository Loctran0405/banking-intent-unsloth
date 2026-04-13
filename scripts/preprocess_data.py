import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def main():
    print("Đang tải dataset BANKING77...")
    os.makedirs("sample_data", exist_ok=True)
    
    # 1. Tải dataset gốc chuẩn từ Hugging Face
    # dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    dataset = load_dataset("banking77")
    
    # Lấy hàm map nhãn (từ số sang chữ, ví dụ: 0 -> card_arrival)
    int2str = dataset['train'].features['label'].int2str
    
    # Gộp toàn bộ train và test lại để chia lại từ đầu cho chuẩn
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    # 2. Chia tập Train/Test dùng Stratify (Đảm bảo đủ 77 intents)
    # Lấy đúng 1500 câu train và 300 câu test cho nhẹ máy
    train_df, test_df = train_test_split(
        df_all, 
        train_size=1500, 
        test_size=300, 
        stratify=df_all['label'], # Chia đều đặn các nhóm ý định
        random_state=42
    )
    
    # 3. Đổi nhãn số thành nhãn chữ cho LLM dễ học
    train_df['label'] = train_df['label'].apply(int2str)
    test_df['label'] = test_df['label'].apply(int2str)
    
    # 4. Chỉ giữ lại 2 cột cần thiết và lưu file
    train_df[['text', 'label']].to_csv("sample_data/train.csv", index=False)
    test_df[['text', 'label']].to_csv("sample_data/test.csv", index=False)
    
    print(f"Đã lưu thành công: Train ({len(train_df)} mẫu), Test ({len(test_df)} mẫu).")

if __name__ == "__main__":
    main()