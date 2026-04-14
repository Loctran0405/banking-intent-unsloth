import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def main():
    print("Đang tải dataset BANKING77...")
    os.makedirs("sample_data", exist_ok=True)
    
    # Tải dataset gốc chuẩn từ Hugging Face
    dataset = load_dataset("banking77")
    
    int2str = dataset['train'].features['label'].int2str
    
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    # Lấy đúng 3200 câu train và 500 câu test 
    train_df, test_df = train_test_split(
        df_all, 
        train_size=3200, 
        test_size=500, 
        stratify=df_all['label'], 
        random_state=42
    )
    
    #Đổi nhãn số thành nhãn chữ cho LLM dễ học
    train_df['label'] = train_df['label'].apply(int2str)
    test_df['label'] = test_df['label'].apply(int2str)
    
    train_df[['text', 'label']].to_csv("sample_data/train.csv", index=False)
    test_df[['text', 'label']].to_csv("sample_data/test.csv", index=False)
    
    print(f"Đã lưu thành công: Train ({len(train_df)} mẫu), Test ({len(test_df)} mẫu).")

if __name__ == "__main__":
    main()