import yaml
import pandas as pd
from unsloth import FastLanguageModel

prompt_template = """Dưới đây là tin nhắn của khách hàng. Hãy phân loại ý định (intent) của tin nhắn này.
### Tin nhắn:
{}
### Ý định:
"""

class IntentClassification:
    def __init__(self, model_path):
        # Đọc file config thông qua biến model_path
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Tải model đã train
        print("Đang tải model để suy luận...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.config["model_path"],
            max_seq_length = self.config.get("max_seq_length", 256),
            dtype = None,
            load_in_4bit = True,
        )
        # Bật chế độ suy luận nhanh gấp 2 lần của Unsloth
        FastLanguageModel.for_inference(self.model)

    def __call__(self, message):
        # Đưa câu nhập vào đúng format đã train
        inputs = self.tokenizer(
            [prompt_template.format(message)], return_tensors="pt"
        ).to("cuda")

        # Sinh kết quả
        outputs = self.model.generate(**inputs, max_new_tokens=64, use_cache=True)
        result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        predicted_label = result.split("### Ý định:\n")[-1].strip()
        return predicted_label

# ==========================================
# PHẦN TEST VỚI TOÀN BỘ FILE CSV & VÍ DỤ NHANH
# ==========================================
if __name__ == "__main__":
    # Khởi tạo mô hình (Truyền file config vào tham số model_path theo đúng đề bài)
    classifier = IntentClassification("configs/inference.yaml")
    
    # ---------------------------------------------------------
    # 1. SHORT USAGE EXAMPLE (Bắt buộc theo yêu cầu Rubric)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("VÍ DỤ SỬ DỤNG NHANH (SHORT USAGE EXAMPLE)")
    print("="*50)
    
    sample_msg = "I lost my card yesterday, what should I do?"
    expected_intent = "lost_or_stolen_card"
    predicted_intent = classifier(sample_msg)
    
    print(f"Message: {sample_msg}")
    print(f"Expected Intent : {expected_intent}")
    print(f"Predicted Intent: {predicted_intent}")
    
    if expected_intent == predicted_intent:
        print("=> Kết quả: ✅ ĐÚNG\n")
    else:
        print("=> Kết quả: ❌ SAI\n")
    
    # ---------------------------------------------------------
    # 2. ĐÁNH GIÁ TRÊN TOÀN BỘ TẬP TEST (Đo Accuracy)
    # ---------------------------------------------------------
    test_file = classifier.config.get("test_data_path", "sample_data/test.csv")
    print(f"Đang đọc dữ liệu kiểm thử từ: {test_file}")
    
    try:
        df = pd.read_csv(test_file)
        total_samples = len(df)
        correct_count = 0  
        
        print("\n" + "="*50)
        print(f"BẮT ĐẦU DỰ ĐOÁN TRÊN TOÀN BỘ {total_samples} CÂU")
        print("="*50)
        
        # Lặp qua toàn bộ file CSV
        for index, row in df.iterrows():
            msg = row['text']
            real_intent = row['label']
            
            # Cho AI dự đoán
            predicted_intent = classifier(msg)
            
            # In quá trình chạy
            print(f"\n[{index+1}/{total_samples}] Message: {msg}")
            print(f"    - Thực tế : {real_intent}")
            print(f"    - AI Đoán : {predicted_intent}")
            
            if real_intent == predicted_intent:
                print("    => Kết quả: ✅ ĐÚNG")
                correct_count += 1
            else:
                print("    => Kết quả: ❌ SAI")
        
        # BẢNG TỔNG KẾT SIÊU NGẦU CUỐI CÙNG
        accuracy = (correct_count / total_samples) * 100
        print("\n" + "★"*50)
        print(" BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (EVALUATION REPORT)")
        print("★"*50)
        print(f" Tổng số câu test : {total_samples}")
        print(f" Số câu đoán ĐÚNG : {correct_count}")
        print(f" Số câu đoán SAI  : {total_samples - correct_count}")
        print(f" ĐỘ CHÍNH XÁC (ACCURACY): {accuracy:.2f}%")
        print("★"*50 + "\n")
                
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {test_file}. Kiểm tra lại đường dẫn nhé!")