"""
Charge un dataset Hugging Face, nettoie le texte, tokenise avec GPT-2.
Par défaut : angeluriot/french_instruct (instructions / réponses en français).

Variable d'environnement optionnelle :
  HF_DATASET=nohurry/Opus-4.6-Reasoning-3000x-filtered  → ancien dataset « raisonnement »

Limité en taille pour CPU + RAM modeste.
"""

from __future__ import annotations

import os
from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# Dataset Hugging Face (défaut : French Instruct)
DATASET_ID = os.environ.get("HF_DATASET", "angeluriot/french_instruct")

# Limite d'exemples (corpus énorme — défaut relevé pour un meilleur affinage ; réduire si peu de RAM)
_MAX = int(os.environ.get("HF_MAX_SAMPLES", "3000"))
MAX_SAMPLES = max(200, min(_MAX, 20_000))
# True = ne garder que les documents avec author == "human" (moins d'exemples, souvent plus propres)
ONLY_HUMAN_AUTHOR = os.environ.get("FRENCH_INSTRUCT_HUMAN_ONLY", "").lower() in ("1", "true", "yes")

# Longueur cible en tokens (plus haut = mieux pour les dialogues longs ; plus de RAM)
_SL = int(os.environ.get("HF_MAX_SEQ_LENGTH", "384"))
MAX_SEQ_LENGTH = max(128, min(_SL, 512))
# GPT-2 base (léger, adapté au CPU)
MODEL_NAME = "gpt2"


def format_chat_prompt_fr(user_line: str) -> str:
    """Aligné sur le texte d'entraînement produit par _french_instruct_to_text."""
    return f"Utilisateur:\n{user_line}\n\nAssistant:\n"


def format_chat_prompt_opus(user_line: str) -> str:
    """Ancien dataset Opus (problem / thinking)."""
    return f"PROBLEM:\n{user_line}\n\nTHINKING:\n"


def _is_french_instruct_dataset() -> bool:
    return "french_instruct" in DATASET_ID.lower()


def _french_instruct_to_text(example: dict[str, Any]) -> str:
    """
    Aplatit context + conversation (rôles user/assistant) en un bloc texte pour le LM causal.
    """
    parts: list[str] = []
    ctx = (example.get("context") or "").strip()
    if ctx:
        parts.append(f"Contexte:\n{ctx}")

    for turn in example.get("conversation") or []:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("role") or "").strip().lower()
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            parts.append(f"Utilisateur:\n{text}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{text}")
        else:
            parts.append(f"{role}:\n{text}")

    return "\n\n".join(parts)


def _opus_example_to_text(example: dict[str, Any]) -> str:
    """Dataset nohurry/Opus-4.6-Reasoning-3000x-filtered (problem / thinking / solution)."""
    if "text" in example and example["text"]:
        return str(example["text"]).strip()

    parts: list[str] = []
    for key in ("problem", "thinking", "solution"):
        val = example.get(key)
        if val is not None and str(val).strip():
            parts.append(f"{key.upper()}:\n{str(val).strip()}")

    if parts:
        return "\n\n".join(parts)

    chunks: list[str] = []
    for _k, v in example.items():
        if isinstance(v, str) and v.strip():
            chunks.append(v.strip())
    return "\n\n".join(chunks) if chunks else ""


def _record_to_text(example: dict[str, Any]) -> str:
    if _is_french_instruct_dataset():
        return _french_instruct_to_text(example)
    return _opus_example_to_text(example)


def _clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()


def load_raw_hf_dataset():
    """
    Charge le dataset depuis Hugging Face (cache : %USERPROFILE%\\.cache\\huggingface\\hub\\...).

    Sous-échantillonne après mélange (seed fixe) pour ne pas toujours prendre les mêmes débuts de corpus.
    """
    ds = load_dataset(DATASET_ID)
    split_names = list(ds.keys())
    if not split_names:
        raise RuntimeError("Dataset vide : aucun split trouvé.")
    main = ds[split_names[0]]

    if _is_french_instruct_dataset() and ONLY_HUMAN_AUTHOR:
        main = main.filter(lambda x: x.get("author") == "human")

    n = min(MAX_SAMPLES, len(main))
    if n == 0:
        raise RuntimeError(
            "Aucun exemple après filtrage. Essayez ONLY_HUMAN_AUTHOR=0 ou augmentez le corpus."
        )
    main = main.shuffle(seed=42).select(range(n))
    return main


def build_text_dataset(hf_split) -> Dataset:
    """Nettoyage + colonne unique 'text'."""

    def _row_to_text(example: dict[str, Any]) -> dict[str, str]:
        return {"text": _clean_text(_record_to_text(example))}

    cols = hf_split.column_names
    ds = hf_split.map(_row_to_text, remove_columns=cols)
    ds = ds.filter(lambda x: len(x["text"]) > 0)
    return ds


def tokenize_for_causal_lm(
    text_dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Dataset:
    """Tokenisation avec truncation ; padding au batch (collator)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize_batch(examples: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
            add_special_tokens=True,
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    return text_dataset.map(
        _tokenize_batch,
        batched=True,
        remove_columns=text_dataset.column_names,
        desc="Tokenisation",
    )


def get_train_dataset(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Dataset:
    """Pipeline : chargement HF → texte → tokenisation pour Trainer + collator."""
    raw = load_raw_hf_dataset()
    text_ds = build_text_dataset(raw)
    return tokenize_for_causal_lm(text_ds, tokenizer)
