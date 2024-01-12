import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

model_name = "taide_model/b.11.0.0" # 原始模型
new_model = "taide-11.0_v2" # 訓練好的模型
output_dir = "./results" # tensorboard結果

# lora parameters
lora_r = 64 
lora_alpha = 16
lora_dropout = 0.1

# 4-bit quantization configuration
use_4bit = True
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_quant_type = "nf4"
use_double_quant = False

fp16 = False
bf16 = True


# training configuration
num_train_epochs = 1
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
gradient_accumulation_steps = 1
gradient_checkpointing = True

max_grad_norm = 0.3
learning_rate = 5e-5
weight_decay = 0.001

optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"

max_steps = -1
warmup_ratio = 0.03

group_by_length = True

save_steps = 0
logging_steps = 25
max_seq_length = 4096
packing = False

device_map = "auto"

# load dataset
dataset = load_dataset('json',data_files={
                            "train":"drcd_llama2_format_hardnegative/hardnegative_DRCD_training_512.json",
                            "eval":"drcd_llama2_format_hardnegative/hardnegative_DRCD_dev_512.json"
                        }, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_double_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.bfloat16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters

# DataCollatorForCompletionOnlyLM會把instruction_template到response_template之間的label設成-100，所以就不會計算loss
instruction_template = "[INST]"
response_template = "[/INST]"
collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    data_collator=collator,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()
trainer.model.save_pretrained(new_model) # 儲存lora參數

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload() # 將lora與原始模型融合

model.save_pretrained(new_model+"-full") # 儲存完整模型
tokenizer.save_pretrained(new_model+"-full")