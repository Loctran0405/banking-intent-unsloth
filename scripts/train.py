import yaml
import argparse
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

prompt_template = """Dưới đây là tin nhắn của khách hàng. Hãy phân loại ý định (intent) của tin nhắn này.
### Tin nhắn:
{}
### Ý định:
{}"""

def format_data(examples):
    texts = [prompt_template.format(text, label) + eos_token for text, label in zip(examples["text"], examples["label"])]
    return {"text": texts}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model_name"],
        max_seq_length = config["max_seq_length"],
        dtype = None,
        load_in_4bit = True,
    )

    model = FastLanguageModel.get_peft_model(
        model, r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16, lora_dropout = 0, bias = "none", use_gradient_checkpointing = "unsloth", random_state = 3407,
    )
    eos_token = tokenizer.eos_token

    df = pd.read_csv(config["train_data_path"])
    dataset = Dataset.from_pandas(df).map(format_data, batched=True)

    trainer = SFTTrainer(
        model = model, tokenizer = tokenizer, train_dataset = dataset, dataset_text_field = "text",
        max_seq_length = config["max_seq_length"], packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = config["batch_size"],
            gradient_accumulation_steps = config["gradient_accumulation_steps"],
            num_train_epochs = config["epochs"], learning_rate = config["learning_rate"],
            fp16 = not torch.cuda.is_bf16_supported(), bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 10, optim = "adamw_8bit", weight_decay = 0.01,
            lr_scheduler_type = "linear", seed = 3407, output_dir = config["output_dir"],
        ),
    )

    trainer.train()
    model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print("Train thành công và đã lưu model!")