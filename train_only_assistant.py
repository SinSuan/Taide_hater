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

PATH_2_MODEL_ORIGINAL = "taide_model/b.11.0.0" # 原始模型
PATH_2_MODEL_TRAINED = "taide-11.0_v2" # 訓練好的模型
output_dir = "./results" # tensorboard結果

# lora parameters
LORA_R = 64 
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# 4-bit quantization configuration
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4"
USE_DOUBLE_QUANT = False

FP16 = False
BF16 = True


# training configuration
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRANDIENT_ACCUMLATION_STEPS = 1
GRANDIENT_CHECKPOINTING = True

MAX_GRAD_NORM = 0.3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.001

OPTIM = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"

MAX_STEPS = -1
WARMUP_RATIO = 0.03

GROUP_BY_LENGTH = True

SAVE_STEPS = 0
LOGGING_STEPS = 25
MAX_SEQ_LENGTH = 4096
PACKING = False

DEVICE_MAP = "auto"

# load dataset
dataset = load_dataset('json',data_files={
                            "train":"drcd_llama2_format_hardnegative/hardnegative_DRCD_training_512.json",
                            "eval":"drcd_llama2_format_hardnegative/hardnegative_DRCD_dev_512.json"
                        }, split="train")

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_4BIT,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.bfloat16 and USE_4BIT:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    PATH_2_MODEL_ORIGINAL,
    quantization_config=bnb_config,
    device_map=DEVICE_MAP
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(PATH_2_MODEL_ORIGINAL, trust_remote_code=True)
tokenizer.padding_side = "right"

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRANDIENT_ACCUMLATION_STEPS,
    optim=OPTIM,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fp16=FP16,
    bf16=BF16,
    max_grad_norm=MAX_GRAD_NORM,
    max_steps=MAX_STEPS,
    warmup_ratio=WARMUP_RATIO,
    group_by_length=GROUP_BY_LENGTH,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
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
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=PACKING,
)

trainer.train()
trainer.model.save_pretrained(PATH_2_MODEL_TRAINED) # 儲存lora參數

base_model = AutoModelForCausalLM.from_pretrained(
    PATH_2_MODEL_ORIGINAL,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=DEVICE_MAP,
)
model = PeftModel.from_pretrained(base_model, PATH_2_MODEL_TRAINED)
model = model.merge_and_unload() # 將lora與原始模型融合

model.save_pretrained(PATH_2_MODEL_TRAINED+"-full") # 儲存完整模型
tokenizer.save_pretrained(PATH_2_MODEL_TRAINED+"-full")