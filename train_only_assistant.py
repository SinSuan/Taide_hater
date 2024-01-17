"""
找 parameter 定義的方法：
ex: type(model) == 'transformers.models.llama.modeling_llama.LlamaForCausalLM'
    transformers, models, llama, modeling_llama, LlamaForCausalLM 每層的文件都找找看
"""

# import os
import torch
from datasets import (
    load_dataset,
    Dataset             # dataset = 'datasets.arrow_dataset.Dataset'
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    # HfArgumentParser,
    LlamaForCausalLM,   # model == 'transformers.models.llama.modeling_llama.LlamaForCausalLM'
    LlamaTokenizerFast, # tokenizer == 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'
    TrainingArguments,  # training_arguments == 'transformers.training_args.TrainingArguments'
    # pipeline,
    # logging,
)
from peft import (
    LoraConfig,         # peft_config == 'transformers.training_args.TrainingArguments'
    PeftModel
)
from trl import (
    SFTTrainer,         # trainer == 'trl.trainer.sft_trainer.SFTTrainer'
    DataCollatorForCompletionOnlyLM
                        # collator == 'trl.trainer.utils.DataCollatorForCompletionOnlyLM'
)

PATH_2_DATA_TRAIN = "/user_data/my/project/Taide_hater/data/test.json"
PATH_2_DATA_EVAL = "/user_data/my/project/Taide_hater/data/test.json"

# 路徑不對會出現
# If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`
PATH_2_MODEL_ORIGINAL = "model/b.11.0.0" # 原始模型
PATH_2_MODEL_TRAINED = "model/test/20240117_1327" # 訓練好的模型
OUTPUT_DIR = "./results" # tensorboard結果

# lora parameters
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# 4-bit quantization configuration
USE_4BIT = True
# USE_4BIT = False
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
BNB_4BIT_QUANT_TYPE = "nf4" # normalized float 4
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

# DataCollatorForCompletionOnlyLM會把instruction_template到response_template之間的label設成-100，所以就不會計算loss
INSTRUCTION_TEMPLATE = "[INST]"
RESPONSE_TEMPLATE = "[/INST]"

DEVICE_MAP = "auto"

# load dataset
def create_dataset(datapath: str) -> Dataset:
    """
    123
    """
    
    dataset = load_dataset(
        'json',
        data_files=datapath,
        split="train"   # 不能省
    )

    return dataset

def print_gpu_compatibility(dtype1, dtype2) -> None:
    """Check GPU compatibility with bfloat16
    """
    
    if dtype1 == dtype2 and USE_4BIT:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

def load_model(path_2_model: str, config) -> LlamaForCausalLM:
    """Load model
    """

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        path_2_model,
        quantization_config=config,
        device_map=DEVICE_MAP
    )

    return model

def load_tokenizer(path_2_model) -> LlamaTokenizerFast:
    """Load tokenizer
    """

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path_2_model, trust_remote_code=True)
    tokenizer.padding_side = "right"

    return tokenizer

def param_4_trainer(tokenizer: AutoTokenizer) -> ['Config', TrainingArguments, 'Collator']:
    """
    123
    """

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
        output_dir=OUTPUT_DIR,
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

    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=INSTRUCTION_TEMPLATE,
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer, mlm=False
    )

    return peft_config, training_arguments, collator

def main():
    """
    123
    """

    # Load tokenizer and model with QLoRA configuration

    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)

    # Check GPU compatibility with bfloat16
    print_gpu_compatibility(compute_dtype, torch.bfloat16)

    # https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        # enable 4-bit quantization (replace the Linear layers with FP4/NF4 layers from bitsandbytes)
        # defaults to False.
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        # quantization data type in the bnb.nn.Linear4Bit layers
        # "fp4" or "nf4". defaults to "fp4".
        bnb_4bit_compute_dtype=compute_dtype,
        # computational type which might be different than the input time
        # defaults to torch.float32.
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
        # enable nested quantization (get the quantization constants from the 1st quantization.)
        # defaults to False.
    )

    model = load_model(PATH_2_MODEL_ORIGINAL, bnb_config)
    tokenizer = load_tokenizer(PATH_2_MODEL_ORIGINAL)

    # Set supervised fine-tuning parameters
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    peft_config, training_arguments, collator = param_4_trainer(tokenizer)
    dataset = create_dataset(PATH_2_DATA_TRAIN)

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

    # 'transformers.models.llama.modeling_llama.LlamaForCausalLM'
    base_model = AutoModelForCausalLM.from_pretrained(
        PATH_2_MODEL_ORIGINAL,      # pretrained_model_name_or_path
        low_cpu_mem_usage=True,
        #       ***transformers.models.from_pretrained***
        # Tries to not use more than 1x model size in CPU memory while loading the model
        return_dict=True,
        # https://huggingface.co/docs/transformers/v4.33.3/en/main_classes/configuration
        torch_dtype=torch.float16,
        #       ***transformers.models.from_pretrained***
        # default to torch.float (fp32).
        device_map=DEVICE_MAP,
        #       ***transformers.models.from_pretrained***
        # specifies the devices on which each submodule (ex: layers in a model) should execute on
        # "cpu", "cuda:1", "mps" -> entire model to this device
        # 0 -> entire model to GPU 0
        # "auto" -> Accelerate will compute the most optimized 'device_map' automatically
    )
    
    
    model = PeftModel.from_pretrained(base_model, PATH_2_MODEL_TRAINED) # infernece 時，這樣可能就可以用了
    model = model.merge_and_unload() # 將lora與原始模型融合

    model.save_pretrained(PATH_2_MODEL_TRAINED+"-full") # 儲存完整模型
    tokenizer.save_pretrained(PATH_2_MODEL_TRAINED+"-full")
    
    print("\twell done")

if __name__ == '__main__':
    main()
    