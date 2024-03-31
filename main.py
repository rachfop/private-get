from data_generation import generate_data
from train import train_model

prompt = "A model that takes in a puzzle-like reasoning-heavy question in English, and responds with a well-reasoned, step-by-step thought out response in Spanish."
temperature = .4
number_of_examples = 1

model_name = "NousResearch/llama-2-7b-chat-hf"
dataset_name = "content/train.jsonl"
new_model = "llama-2-7b-custom"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
device_map = {"": 0}

system_message = generate_data(prompt, temperature, number_of_examples)
train_model(model_name, dataset_name, new_model, lora_r, lora_alpha, lora_dropout, use_4bit,
            bnb_4bit_compute_dtype, bnb_4bit_quant_type, use_nested_quant, output_dir,
            num_train_epochs, fp16, bf16, per_device_train_batch_size, per_device_eval_batch_size,
            gradient_accumulation_steps, gradient_checkpointing, max_grad_norm, learning_rate,
            weight_decay, optim, lr_scheduler_type, max_steps, warmup_ratio, group_by_length,
            save_steps, logging_steps, max_seq_length, packing, device_map, system_message)