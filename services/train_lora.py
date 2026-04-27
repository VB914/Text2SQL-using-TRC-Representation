from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional LoRA fine-tuning entry point for exported JSONL data.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--model-name", default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--output-dir", default="models/text2sql-lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Optional LoRA dependencies are missing. Install them with "
            "`pip install -r requirements-lora.txt` before running this script."
        ) from exc

    train_file = Path(args.train_file)
    if not train_file.exists():
        raise SystemExit(f"Training file not found: {train_file}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, config)

    dataset = load_dataset("json", data_files=str(train_file), split="train")

    def tokenize(row: dict) -> dict:
        if "messages" in row:
            text = "\n".join(f"{message['role']}: {message['content']}" for message in row["messages"])
        else:
            text = f"{row.get('prompt', '')}\n{row.get('completion', '')}"
        return tokenizer(text, truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        logging_steps=25,
        save_steps=250,
        save_total_limit=2,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
