"""
Fine-tuning GPT-2 (causal LM) — réglages orientés affinage sur CPU.
Validation + sauvegarde du meilleur checkpoint selon eval_loss.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    TrainerCallback,
    Trainer,
    TrainingArguments,
)

from data_loader import DATASET_ID, MODEL_NAME, get_train_dataset
from model import load_for_finetune, save_model_and_tokenizer


class StopAfterTimeCallback(TrainerCallback):
    """
    Arrête l'entraînement après un nombre de secondes donné (utile pour CPU).
    Le stop peut tomber entre deux eval ; on s'appuie sur load_best_model_at_end
    + l'évaluation périodique pour conserver un checkpoint "meilleur".
    """

    def __init__(self, max_seconds: int) -> None:
        super().__init__()
        self.max_seconds = int(max_seconds)
        self.start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        import time

        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # Si pas de limite, ne fait rien.
        if not self.max_seconds or self.max_seconds <= 0:
            return

        import time

        if self.start_time is None:
            return

        elapsed_s = time.time() - self.start_time
        if elapsed_s >= self.max_seconds:
            print(f"\n[TimeLimit] Arrêt demandé après ~{int(elapsed_s)}s (max={self.max_seconds}s).")
            control.should_training_stop = True
            # Tentative de sauvegarde du checkpoint au moment du stop.
            control.should_save = True


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Périphérique : {device}")
    print(f"Dataset : {DATASET_ID}")

    out_dir = os.path.join(os.path.dirname(__file__), "model")
    resume = os.environ.get("HF_RESUME", "").lower() in ("1", "true", "yes")

    model, tokenizer = load_for_finetune(MODEL_NAME, out_dir, resume=resume)
    ckpt = Path(out_dir) / "config.json"
    if resume and ckpt.is_file():
        print(f"Reprise du fine-tuning depuis : {out_dir} (HF_RESUME=1)")
    elif resume:
        print("HF_RESUME activé mais aucun checkpoint — départ depuis GPT-2 de base.")
    else:
        print(f"Départ : {MODEL_NAME} (poids Hugging Face)")

    model.resize_token_embeddings(len(tokenizer))

    full_ds = get_train_dataset(tokenizer)
    if len(full_ds) < 10:
        raise RuntimeError("Pas assez d'exemples après nettoyage ; vérifie le dataset.")

    split = full_ds.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"Exemples train : {len(train_ds)} | eval : {len(eval_ds)}")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=None,
    )

    epochs = int(os.environ.get("HF_TRAIN_EPOCHS", "4"))
    lr = float(os.environ.get("HF_LEARNING_RATE", "3e-5"))

    # Steps d’optimisation par epoch ≈ len(train) / (batch × accumulation)
    optim_steps_per_epoch = max(1, len(train_ds) // 8)
    eval_steps = max(30, min(500, optim_steps_per_epoch))

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        warmup_ratio=0.08,
        logging_steps=max(10, eval_steps // 4),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        prediction_loss_only=True,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # Optionnel : arrêter après X secondes (par défaut désactivé)
    max_seconds = int(os.environ.get("HF_TRAIN_MAX_SECONDS", "0"))
    callbacks = []
    if max_seconds > 0:
        callbacks.append(StopAfterTimeCallback(max_seconds=max_seconds))

    if callbacks:
        trainer.add_callback(callbacks[0])

    trainer.train()
    save_model_and_tokenizer(model, tokenizer, out_dir)
    print(f"Meilleur modèle (selon eval_loss) sauvegardé dans : {out_dir}")


if __name__ == "__main__":
    main()
