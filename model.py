"""
GPT-2 pour fine-tuning causal LM : chargement, sauvegarde, rechargement.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from data_loader import MODEL_NAME


def get_tokenizer(model_name: str = MODEL_NAME) -> GPT2TokenizerFast:
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(model_name: str = MODEL_NAME) -> GPT2LMHeadModel:
    """
    Charge GPT-2 avec tête de langage pour la génération (fine-tuning standard).
    """
    return GPT2LMHeadModel.from_pretrained(model_name)


def load_model_and_tokenizer_for_training(
    resume_dir: str | Path,
) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    """Reprend un checkpoint local (mêmes poids + tokenizer) pour continuer le fine-tuning."""
    resume_dir = Path(resume_dir)
    if not (resume_dir / "config.json").is_file():
        raise FileNotFoundError(f"Pas de checkpoint dans {resume_dir} (config.json manquant).")

    tok = GPT2TokenizerFast.from_pretrained(resume_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(resume_dir)
    return model, tok


def load_for_finetune(
    base_model_name: str,
    resume_dir: str | Path,
    resume: bool,
) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    """Ouverture : GPT-2 de base, ou poursuite depuis resume_dir si resume et dossier valide."""
    resume_dir = Path(resume_dir)
    if resume and (resume_dir / "config.json").is_file():
        return load_model_and_tokenizer_for_training(resume_dir)
    model = build_model(base_model_name)
    tok = get_tokenizer(base_model_name)
    return model, tok


def save_model_and_tokenizer(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_model_and_tokenizer(
    load_dir: str | Path,
    map_location: str | None = None,
) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    """
    Charge un modèle fine-tuné depuis ./model (ou autre dossier).

    map_location : "cpu" pour forcer l'inférence sans GPU.
    """
    load_dir = Path(load_dir)
    if not load_dir.is_dir():
        raise FileNotFoundError(f"Dossier modèle introuvable : {load_dir}")

    tok = GPT2TokenizerFast.from_pretrained(load_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT2LMHeadModel.from_pretrained(load_dir)
    model.to(map_location)
    model.eval()
    return model, tok
