# import yaml
# from unsloth import FastLanguageModel

# prompt_template = """Dưới đây là tin nhắn của khách hàng. Hãy phân loại ý định (intent) của tin nhắn này.
# ### Tin nhắn:
# {}
# ### Ý định:
# """

# class IntentClassification:
#     def __init__(self, model_path):
#         # 1. Đọc file config (inference.yaml)
#         with open(model_path, "r") as f:
#             config = yaml.safe_load(f)
            
#         # 2. Tải model đã train
#         print("Đang tải model để suy luận...")
#         self.model, self.tokenizer = FastLanguageModel.from_pretrained(
#             model_name = config["model_path"],
#             max_seq_length = config.get("max_seq_length", 256),
#             dtype = None,
#             load_in_4bit = True,
#         )
#         # Bật chế độ suy luận nhanh gấp 2 lần của Unsloth
#         FastLanguageModel.for_inference(self.model)

#     def __call__(self, message):
#         # Đưa câu nhập vào đúng format đã train
#         inputs = self.tokenizer(
#             [prompt_template.format(message)], return_tensors="pt"
#         ).to("cuda")

#         # Sinh kết quả
#         outputs = self.model.generate(**inputs, max_new_tokens=64, use_cache=True)
#         result = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
#         # Cắt bỏ phần râu ria, chỉ lấy nhãn (label)
#         predicted_label = result.split("### Ý định:\n")[-1].strip()
#         return predicted_label

# # Đoạn test nhanh
# if __name__ == "__main__":
#     classifier = IntentClassification("configs/inference.yaml")
    
#     # Thử một vài câu
#     test_messages = [
#         "I lost my card yesterday, what should I do?",
#         "How can I top up my mobile phone?",
#         "Why is my transfer still pending?"
#     ]
    
#     for msg in test_messages:
#         print(f"\nMessage: {msg}")
#         print(f"Intent : {classifier(msg)}")



import yaml
import pandas as pd
from unsloth import FastLanguageModel

prompt_template = """Dưới đây là tin nhắn của khách hàng. Hãy phân loại ý định (intent) của tin nhắn này.
### Tin nhắn:
{}
### Ý định:
"""

class IntentClassification:
    def __init__(self, config_path):
        # 1. Đọc file config (inference.yaml)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # 2. Tải model đã train
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
        
        # Cắt bỏ phần râu ria, chỉ lấy nhãn (label)
        predicted_label = result.split("### Ý định:\n")[-1].strip()
        return predicted_label

# ==========================================
# PHẦN TEST VỚI FILE CSV
# ==========================================
if __name__ == "__main__":
    # Khởi tạo mô hình
    classifier = IntentClassification("configs/inference.yaml")
    
    # Lấy đường dẫn file test từ config
    test_file = classifier.config.get("test_data_path", "sample_data/test.csv")
    print(f"\nĐang đọc dữ liệu kiểm thử từ: {test_file}")
    
    try:
        df = pd.read_csv(test_file)
        
        print("\n" + "="*50)
        print("BẮT ĐẦU DỰ ĐOÁN (LẤY MẪU 10 CÂU ĐẦU TIÊN)")
        print("="*50)
        
        # Lấy 10 dòng đầu tiên để test nhanh
        for index, row in df.head(10).iterrows():
            msg = row['text']
            real_intent = row['label'] # Lấy nhãn thực tế từ file CSV
            
            # Cho AI dự đoán
            predicted_intent = classifier(msg)
            
            # In kết quả so sánh siêu ngầu
            print(f"\n[{index+1}] Message: {msg}")
            print(f"    - Thực tế : {real_intent}")
            print(f"    - AI Đoán : {predicted_intent}")
            
            if real_intent == predicted_intent:
                print("    => Kết quả: ✅ ĐÚNG")
            else:
                print("    => Kết quả: ❌ SAI")
                
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {test_file}. Lộc kiểm tra lại đường dẫn nhé!")