# Fine-Tuning Intent Detection Model using Unsloth

**Dự án fine-tuning mô hình LLM để phân loại ý định (Intent Detection) trong lĩnh vực ngân hàng**

Sử dụng thư viện **Unsloth** + **LoRA**
## Hướng Dẫn Cài Đặt

### A. Trên VS code
#### Bước 1: Cài Đặt Dependency
```bash
pip install -r requirements.txt
```
> - `unsloth`: Thư viện tối ưu huấn luyện LLM
> - `transformers==4.53.2`: HuggingFace transformers
> - `trl==0.20.0`: Supervised Fine-Tuning Trainer
> - `datasets`: Xử lý dataset
> - `pyyaml`: Đọc file config

---

### B. Trên Kaggle Notebook 

1. Đăng nhập vào Hugging Face
2. Tạo Access Token
3. Copy token này
4. Trên Kaggle:
   - Vào **Settings** → **Secrets**
   - Thêm secret mới với tên: `HF_TOKEN`
   - Paste token HuggingFace vào


# CLONE REPO TỪ GITHUB
!git clone https://github.com/Loctran0405/banking-intent-unsloth.git

# ============================================
# CÀI ĐẶT DEPENDENCIES
# ============================================
%cd banking-intent-unsloth
!pip install -r requirements.txt -q

---


## Huấn Luyện Mô Hình
Mở `configs/train.yaml`:

```yaml
model_name: "unsloth/llama-3-8b-bnb-4bit"    
max_seq_length: 256                        
learning_rate: 0.0002                    
batch_size: 4                              
gradient_accumulation_steps: 4              
epochs: 3                                    
train_data_path: "sample_data/train.csv"
output_dir: "outputs/banking_intent_model"
```

### Chạy Huấn Luyện
```bash
python scripts/preprocess_data.py
python scripts/train.py --config configs/train.yaml
```

### Cấu Hình Inference

Mở `configs/inference.yaml`:

```yaml
model_path: "outputs/banking_intent_model"
test_data_path: "sample_data/test.csv"
max_seq_length: 256
```
### Chạy Train
**Trên Kaggle**
!chmod +x train.sh
!./train.sh

### Chạy Inference
**Trên Kaggle**
!chmod +x inference.sh
!./inference.sh
