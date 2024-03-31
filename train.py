import pandas as pd
import torch
from peft import LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments, logging,
                          pipeline)
from trl import SFTTrainer


def train_model(
    model_name,
    dataset_name,
    new_model,
    lora_r,
    lora_alpha,
    lora_dropout,
    use_4bit,
    bnb_4bit_compute_dtype,
    bnb_4bit_quant_type,
    use_nested_quant,
    output_dir,
    num_train_epochs,
    fp16,
    bf16,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    gradient_accumulation_steps,
    gradient_checkpointing,
    max_grad_norm,
    learning_rate,
    weight_decay,
    optim,
    lr_scheduler_type,
    max_steps,
    warmup_ratio,
    group_by_length,
    save_steps,
    logging_steps,
    max_seq_length,
    packing,
    device_map,
    system_message,
):
    train_dataset = pd.read_json("content/train.jsonl", lines=True)
    valid_dataset = pd.read_json("content/test.jsonl", lines=True)
    train_dataset_mapped = pd.DataFrame(
        {
            "text": train_dataset.apply(
                lambda x: f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n"
                + x["prompt"]
                + " [/INST] "
                + x["response"]
            )
        }
    )
    valid_dataset_mapped = pd.DataFrame(
        {
            "text": valid_dataset.apply(
                lambda x: f"[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n"
                + x["prompt"]
                + " [/INST] "
                + x["response"]
            )
        }
    )
    # train_dataset_mapped = train_dataset.apply(lambda x: {'text': f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + x['prompt'] + ' [/INST] ' + x['response']}, axis=1)
    # valid_dataset_mapped = valid_dataset.apply(lambda x: {'text': f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + x['prompt'] + ' [/INST] ' + x['response']}, axis=1)
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

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
        report_to="all",
        evaluation_strategy="steps",
        eval_steps=5,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset_mapped,
        eval_dataset=valid_dataset_mapped,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    trainer.train()
    trainer.model.save_pretrained(new_model)

    logging.set_verbosity(logging.CRITICAL)
    prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\nWrite a function that reverses a string. [/INST]"
    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, max_length=200
    )
    result = pipe(prompt)
    print(result[0]["generated_text"])
