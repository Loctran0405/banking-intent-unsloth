# 🏦 Banking Intent Classification - Unsloth

Dự án phân loại ý định khách hàng (Intent Classification) sử dụng **Llama-3-8B** được tối ưu hóa với **Unsloth** để training nhanh gấp 2 lần. Model được huấn luyện trên dataset Banking77 để nhận diện 77 ý định khác nhau trong giao dịch ngân hàng.

---

##  Mục Lục
1. [Mô Tả Dự Án](#mô-tả-dự-án)
2. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
3. [Cài Đặt Môi Trường](#cài-đặt-môi-trường)
4. [Chuẩn Bị Dữ Liệu](#chuẩn-bị-dữ-liệu)
5. [Huấn Luyện Model](#huấn-luyện-model)
6. [Suy Luận & Kiểm Thử](#suy-luận--kiểm-thử)
7. [Cấu Trúc Dự Án](#cấu-trúc-dự-án)
8. [Cấu Hình Chi Tiết](#cấu-hình-chi-tiết)
9. [Chi Tiết Các Tham Số Huấn Luyện](#-chi-tiết-các-tham-số-huấn-luyện)
10. [Hướng Dẫn Kaggle](#hướng-dẫn-kaggle)
---

## Mô Tả Dự Án

**Mục đích:** Phân loại ý định của tin nhắn khách hàng ngân hàng thành 77 loại ý định khác nhau (ví dụ: chuyển tiền, khiếu nại, hỗ trợ thẻ, v.v.)

**Công nghệ:**
- **Model Base:** Llama-3-8B (lượng tử hóa 4-bit)
- **Optimization:** Unsloth (tăng tốc độ training 2x)
- **Fine-tuning:** LoRA (Low-Rank Adaptation)
- **Framework:** Hugging Face Transformers + TRL (Transformer Reinforcement Learning)

**Dataset:** BANKING77 - tham khảo từ Hugging Face Hub

---

## Yêu Cầu Hệ Thống

### Phần Cứng (Kaggle)
- **GPU:** T4 (14GB VRAM) hoặc cao hơn
- **CPU:** Intel/AMD 4+ cores
- **RAM:** 8GB+ (Kaggle cung cấp 30GB)
- **Lưu trữ:** 50GB+ cho outputs và models

### Phần Mềm
- Python 3.10+
- PyTorch 2.0+
- CUDA 12.0+ 
- Git (để clone repository)

---

## Cài Đặt Môi Trường

### Chạy Trên Kaggle (Khuyến Nghị) 

Kaggle cung cấp GPU miễn phí và môi trường được cấu hình sẵn, đây là cách nhanh nhất.

#### Bước 1: Nạp HF_TOKEN

```python
import os
from kaggle_secrets import UserSecretsClient

# Lấy chìa khóa HF_TOKEN và nạp vào hệ thống
try:
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
    print(" Đã nạp HF_TOKEN thành công!")
except Exception as e:
    print(" LỖI: Chưa nạp được Token (public models sẽ vẫn hoạt động)")

# Tắt W&B logging để tiết kiệm tài nguyên
os.environ["WANDB_DISABLED"] = "true"
```

#### Bước 2: Clone Repository

```bash
!git clone https://github.com/Loctran0405/banking-intent-unsloth.git
%cd banking-intent-unsloth
```
#### Bước 3: Cài Đặt Dependencies

```bash
!pip install -r requirements.txt
```

#### Bước 2: Cài Dependencies

```bash
pip install -r requirements.txt
```

## Huấn Luyện Model

### Dùng Kaggle

```bash
!chmod +x train.sh
!./train.sh
```
1. ✅ Chạy `preprocess_data.py` → Tải & chuẩn bị dữ liệu
   ### Tự Động Tải & Xử Lý (Kaggle)
   Chạy script xử lý dữ liệu để tải từ Hugging Face và tạo tập train/test:
```bash
   python scripts/preprocess_data.py
```

   **Hoạt động:**
   1. Tải dataset BANKING77 từ Hugging Face Hub
   2. Chọn 3,200 mẫu để train + 500 mẫu để test
   3. Chuyển nhãn từ số thành chữ (ví dụ: 0 → "lost_or_stolen_card")
   4. Lưu vào:
      - `sample_data/train.csv` (3,200 hàng)
      - `sample_data/test.csv` (500 hàng)

   **Định dạng dữ liệu:**
   ```csv
   text,label
   "I lost my card yesterday","lost_or_stolen_card"
   "How do I activate my new card?","activate_my_card"
   ...
   ```

2. ✅ Chạy `train.py` → Huấn luyện model với config `configs/train.yaml`
```bash
   python scripts/train.py --config configs/train.yaml
```

Model sẽ lưu vào: `outputs/banking_intent_model/`

---

##  Suy Luận & Kiểm Thử

### Chạy Inference Trên Toàn Bộ Test Set (Kaggle)

```bash
!chmod +x inference.sh
!./inference.sh
```

**Kết Quả Output:**

```
==================================================
VÍ DỤ SỬ DỤNG NHANH (SHORT USAGE EXAMPLE)
==================================================
Message: I lost my card yesterday, what should I do?
Expected Intent : lost_or_stolen_card
Predicted Intent: lost_or_stolen_card
=> Kết quả: ✅ ĐÚNG

==================================================
BẮT ĐẦU DỰ ĐOÁN TRÊN TOÀN BỘ 500 CÂU
==================================================
[1/500] Message: Your exchange rate is totally wrong for my card payment
    - Thực tế : card_payment_wrong_exchange_rate
    - AI Đoán : card_payment_wrong_exchange_rate
    => Kết quả: ✅ ĐÚNG

[2/500] Message: How do I go about transferring money using my credit card?
    - Thực tế : topping_up_by_card
    - AI Đoán : topping_up_by_card
    => Kết quả: ✅ ĐÚNG
...

[499/500] Message: How many days until the money will be in my account?
    - Thực tế : transfer_timing
    - AI Đoán : transfer_timing
    => Kết quả: ✅ ĐÚNG

[500/500] Message: I deposited a cheque, why isn't my balance showing that I did?
    - Thực tế : balance_not_updated_after_cheque_or_cash_deposit
    - AI Đoán : balance_not_updated_after_cheque_or_cash_deposit
    => Kết quả: ✅ ĐÚNG

★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (EVALUATION REPORT)
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
 Tổng số câu test : 500
 Số câu đoán ĐÚNG : 459
 Số câu đoán SAI  : 41
 ĐỘ CHÍNH XÁC (ACCURACY): 91.80%
★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
```

---

## Cấu Trúc Dự Án

```
banking-intent-unsloth/
├── README.md                  # Tài liệu 
├── requirements.txt           # Các thư viện cần thiết
├── train.sh                   # Script huấn luyện 
├── inference.sh               # Script suy luận 
│
├── configs/                   # Tệp cấu hình
│   ├── train.yaml            # Cấu hình huấn luyện
│   └── inference.yaml        # Cấu hình suy luận
│
├── scripts/                   # Code chính
│   ├── preprocess_data.py    # Tải & xử lý dữ liệu từ HF
│   ├── train.py              # Huấn luyện model
│   └── inference.py          # Suy luận & kiểm thử
│
├── sample_data/               # Dữ liệu mẫu (tạo bởi preprocess_data.py)
│   ├── train.csv             # 3,200 mẫu huấn luyện
│   └── test.csv              # 500 mẫu kiểm thử
│
├── outputs/                   # Đầu ra (tạo sau khi train)
│   └── banking_model_23127407/ # Model đã train
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── pytorch_model.bin
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
```

---

## Cấu Hình Chi Tiết

### `configs/train.yaml` - Cấu Hình Huấn Luyện

```yaml
# Model base
model_name: "unsloth/llama-3-8b-bnb-4bit"    # Llama 3 8B lượng tử 4-bit
max_seq_length: 256                          # Độ dài tối đa của input

# Hyper-parameters
learning_rate: 0.0002                        # Tốc độ học
batch_size: 4                                # Batch size cho mỗi step
gradient_accumulation_steps: 4               # Tích lũy gradient 4 bước
epochs: 3                                    # Số lần lặp toàn bộ dữ liệu

# Đường dẫn
train_data_path: "sample_data/train.csv"    # File train
output_dir: "outputs/banking_intent_model"  # Nơi lưu model
```

**Giải Thích:**
- **learning_rate = 0.0002:** Tốc độ học nhỏ để tránh overfitting
- **batch_size = 4:** Sử dụng 4 mẫu mỗi step (nhỏ để tiết kiệm VRAM)
- **gradient_accumulation_steps = 4:** Tích lũy gradient từ 4 step = tương đương batch_size 16
- **epochs = 3:** Lặp 3 lần toàn bộ 3,200 mẫu training

### `configs/inference.yaml` - Cấu Hình Suy Luận

```yaml
model_path: "outputs/banking_intent_model"  # Đường dẫn model đã train
test_data_path: "sample_data/test.csv"      # File để kiểm thử
max_seq_length: 256                          # Độ dài tối đa (phải giống train)
```

---

## 🔧 Chi Tiết Các Tham Số Huấn Luyện

### Tham Số Trong `configs/train.yaml`

| Tham Số | Giá Trị | Giải Thích |
|---------|--------|-----------|
| `model_name` | `unsloth/llama-3-8b-bnb-4bit` | Model base đã được lượng tử 4-bit để tiết kiệm VRAM |
| `max_seq_length` | 256 | Độ dài tối đa của input sequence. Tăng → tiêu tốn VRAM hơn, nhưng có thể capture tin nhắn dài |
| `learning_rate` | 0.0002 | Tốc độ cập nhật weights. Quá cao → overfitting, quá thấp → training chậm |
| `batch_size` | 4 | Số mẫu xử lý mỗi step. Cao → nhanh nhưng cần VRAM lớn, thấp → chậm nhưng dùng ít VRAM |
| `gradient_accumulation_steps` | 4 | Tích lũy gradient từ 4 step trước khi update. Giả lập batch_size = 4 × 4 = 16 |
| `epochs` | 3 | Số lần lặp toàn bộ training set (3,200 mẫu). Nhiều → model học tốt hơn nhưng dễ overfitting |
| `train_data_path` | `sample_data/train.csv` | Đường dẫn tới file training data |
| `output_dir` | `outputs/banking_intent_model` | Nơi lưu model sau khi train |

### Tham Số Cứng Trong `train.py` (TrainingArguments)

Các tham số này được set trực tiếp trong code, bạn có thể chỉnh sửa trong `train.py`:

```python
args = TrainingArguments(
    # Hyperparameters (từ config)
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    num_train_epochs = 3,
    learning_rate = 0.0002,
    
    # Optimizer & Scheduler
    optim = "adamw_8bit",                    # Sử dụng AdamW 8-bit để tiết kiệm VRAM
    weight_decay = 0.01,                     # L2 regularization để tránh overfitting
    lr_scheduler_type = "linear",            # Giảm learning rate tuyến tính theo steps
    
    # Logging & Saving
    logging_steps = 10,                      # Log loss mỗi 10 steps
    output_dir = config["output_dir"],       # Nơi lưu outputs
    
    # Precision
    fp16 = not torch.cuda.is_bf16_supported(),     # Dùng float16 nếu không support bfloat16
    bf16 = torch.cuda.is_bf16_supported(),         # Dùng bfloat16 nếu GPU support (A100+)
    
    # Random Seed
    seed = 3407,                             # Random seed để kết quả reproducible
)
```

**Chi Tiết Từng Tham Số:**

| Tham Số | Giá Trị | Tác Dụng |
|---------|--------|---------|
| `optim` | `adamw_8bit` | Optimizer 8-bit tiết kiệm memory so với AdamW thường |
| `weight_decay` | 0.01 | L2 regularization giúp tránh overfitting (giá trị 0.01 = 1% penalty) |
| `lr_scheduler_type` | `linear` | Giảm learning rate tuyến tính: LR = initial_lr × (1 - step / total_steps) |
| `logging_steps` | 10 | Hiển thị loss mỗi 10 steps để monitor training |
| `fp16` | Tự động | Float16 precision nếu GPU không support bfloat16 (tiết kiệm 50% memory) |
| `bf16` | Tự động | Bfloat16 precision cho GPU A100+ (duy trì độ chính xác hơn fp16) |
| `seed` | 3407 | Random seed cố định để training kết quả nhất quán |

### Tham Số Trong `SFTTrainer` (Supervised Fine-Tuning)

```python
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",             # Cột trong dataset để đưa vào model
    max_seq_length = 256,                    # Độ dài tối đa sequence
    packing = False,                         # False = không pack multiple samples vào 1 sequence
)
```

| Tham Số | Giá Trị | Giải Thích |
|---------|--------|-----------|
| `dataset_text_field` | `"text"` | Tên cột chứa dữ liệu text để train |
| `max_seq_length` | 256 | Cắt ngắn/pad sequence về độ dài 256 tokens |
| `packing` | False | False = mỗi sample train riêng; True = nhóm nhiều samples (nhanh hơn nhưng memory cần lớn) |

### Tham Số Trong LoRA (Low-Rank Adaptation)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                                  # Rank của LoRA adapters
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"],  # Layers để apply LoRA
    lora_alpha = 16,                         # Scaling factor cho LoRA
    lora_dropout = 0,                        # Dropout trong LoRA (0 = không dùng)
    bias = "none",                           # Không train bias
    use_gradient_checkpointing = "unsloth",  # Tối ưu memory (unsloth version)
    random_state = 3407,
)
```

| Tham Số | Giá Trị | Tác Dụng |
|---------|--------|---------|
| `r` (rank) | 16 | Số chiều của LoRA matrices. Cao → nhiều parameters (~1.3M), thấp → ít hơn nhưng học kém hơn |
| `lora_alpha` | 16 | Scaling factor: output = original + (alpha / r) × LoRA_output. Thường = r |
| `target_modules` | Query, Key, Value, Gate, etc. | Những layers nào sẽ được fine-tune với LoRA |
| `lora_dropout` | 0 | Dropout rate trong LoRA layer (0 = vô hiệu) |
| `use_gradient_checkpointing` | `"unsloth"` | Tối ưu memory bằng cách không lưu intermediate activations |

---

## 📊 Ảnh Hưởng Của Tham Số Lên Training

### 1. **Learning Rate** ↔ **Training Stability**
```
Learning Rate = 0.0002 (hiện tại)
├── ✅ Ổn định, ít overfitting
├── ❌ Có thể train chậm
└── Thử: 0.0001 (chậm hơn), 0.0005 (nhanh hơn)
```

### 2. **Batch Size & Gradient Accumulation**
```
Hiệu quả batch_size = 4 × 4 = 16
├── ✅ Cân bằng VRAM & learning
├── ❌ Nếu VRAM quá ít: giảm batch_size xuống 2
└── Nếu VRAM đủ: tăng batch_size lên 8 (nhanh hơn ~30%)
```

### 3. **Epochs** ↔ **Overfitting Risk**
```
Epochs = 3 (hiện tại)
├── 1 epoch: Model quá nhanh, chưa học hết
├── 3 epochs: ✅ Cân bằng tốt
├── 5+ epochs: ⚠️ Có thể overfitting
└── Hướng: Nếu Accuracy < 80%, tăng epochs lên 5
```

### 4. **Weight Decay** ↔ **Regularization**
```
Weight Decay = 0.01 (hiệu quả)
├── 0: Không regularize → overfitting
├── 0.01: ✅ Tiêu chuẩn (khuyên dùng)
└── 0.1: Quá mạnh → model học chậm
```

---

## 🎯 Cách Điều Chỉnh Tham Số Khi Gặp Vấn Đề

### Vấn Đề: Accuracy Thấp (< 85%)
```yaml
# Giải pháp: Tăng epochs & giảm learning rate
epochs: 5              # Từ 3 → 5
learning_rate: 0.00015  # Từ 0.0002 → 0.00015
batch_size: 4          # Giữ nguyên
```

### Vấn Đề: CUDA Out of Memory
```yaml
# Giải pháp: Giảm batch size (sẽ chậm ~2x)
batch_size: 2          # Từ 4 → 2
gradient_accumulation_steps: 8  # Từ 4 → 8 (để giữ effective batch_size = 16)
max_seq_length: 128    # Từ 256 → 128 (cắt ngắn input)
```

### Vấn Đề: Training Quá Chậm
```yaml
# Giải pháp: Tăng batch size & giảm epochs
batch_size: 8          # Từ 4 → 8
epochs: 2              # Từ 3 → 2
learning_rate: 0.0003  # Tăng slightly để compensate
```

### Vấn Đề: Loss Dao động Lắc Lư
```yaml
# Giải pháp: Giảm learning rate (quá nhanh)
learning_rate: 0.0001  # Từ 0.0002 → 0.0001
```
---
## Troubleshooting

### Lỗi: `CUDA out of memory`

**Nguyên nhân:** Batch size quá lớn

**Giải pháp:** Giảm `batch_size` trong `configs/train.yaml`
```yaml
batch_size: 2  # Từ 4 xuống 2
```

### Lỗi: `Module not found: unsloth`

**Giải pháp:**
```bash
!pip install unsloth[colab-new] --force-reinstall
```

### Model không học (Loss không giảm)

**Kiểm tra:**
1. Learning rate có quá cao? → Giảm xuống 0.0001
2. Dữ liệu có đúng format? → Kiểm tra `sample_data/train.csv`
3. Epoch có đủ? → Tăng lên 5-10 epochs

### Inference quá chậm

**Lý do:** Model Llama 3 8B lớn

**Giải pháp:**
- Dùng Kaggle GPU T4 (faster than CPU)
- Giảm `max_seq_length` trong config
- Nếu cần realtime: fine-tune model nhỏ hơn (DistilBERT)

---

## Tips & Tricks

### 1. **Tối ưu hóa VRAM**
```yaml
batch_size: 2
gradient_accumulation_steps: 8  # 2 * 8 = 16 effective batch
```

### 2. **Lưu Checkpoint Định Kỳ**
Thêm vào `TrainingArguments` trong `train.py`:
```python
save_strategy = "steps",
save_steps = 50,  # Lưu mỗi 50 steps
```

### 3. **Sử Dụng Dataset Cân Bằng**
```python
# Trong preprocess_data.py
stratify=df_all['label']  # Đảm bảo mỗi label có 10% train + 10% test
```

---

## Tài Liệu Tham Khảo

- **Unsloth:** https://github.com/unslothai/unsloth
- **Llama 3:** https://huggingface.co/meta-llama/Llama-3-8b
- **BANKING77 Dataset:** https://huggingface.co/datasets/banking77
- **Hugging Face Transformers:** https://huggingface.co/docs/transformers/
- **LoRA Paper:** https://arxiv.org/abs/2106.09685

---

## Đường dẫn

- GitHub: [Loctran0405/banking-intent-unsloth](https://github.com/Loctran0405/banking-intent-unsloth)
- Video: [Video thực hiện](https://drive.google.com/file/d/1eyKjeqqJB69zZaCa6mgMkOUCHz_10KwX/view?usp=sharing)
---

