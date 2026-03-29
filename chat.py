"""
Chat / génération interactive dans le terminal à partir du modèle fine-tuné dans ./model
Utilise température et truncation du prompt.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import set_seed

from data_loader import format_chat_prompt_fr, format_chat_prompt_opus
from model import load_model_and_tokenizer

# Si la génération invente un 2e tour de dialogue, on coupe avant.
_TURN_MARKERS_FR = (
    "\n\nUtilisateur:",
    "\nUtilisateur:",
    "\n\nAssistant:",
    "\nAssistant:",
    "\n\nContexte:",
    "\nContexte:",
)


def print_session_help(args: argparse.Namespace, device: str, temperature_courante: float) -> None:
    """Aide affichée au démarrage et sur /help."""
    print()
    print("  Mode « chat » (complétion de texte)")
    print("  " + "—" * 44)
    print("  • Écrivez une phrase ou une question, puis Entrée.")
    print(
        "    En format « fr », votre message est envoyé comme un tour « Utilisateur » ; "
        "le modèle enchaîne ce qu’il imagine être la réponse « Assistant »."
    )
    print()
    print("  • Ce n’est pas une source de vérité : faits, noms et dates peuvent être faux.")
    print(
        "    GPT-2 petit + peu d’entraînement ≈ français souvent maladroit ; "
        "les réponses courtes du type « salut » sont en général peu convaincantes."
    )
    print()
    print("  Commandes")
    print("    /help ou /aide     — afficher ce texte")
    print("    /quit              — quitter")
    print("    /temp 0.5          — changer la température (ex. plus bas = plus régulier)")
    print()
    print(
        "  Réglages actuels : "
        f"périphérique={device}, format={args.format}, "
        f"température={temperature_courante}, max_new_tokens={args.max_new_tokens}"
    )
    print()


def _clip_one_assistant_turn(text: str) -> str:
    """Garde une seule réponse ; enlève les bouts où le modèle enchaîne un autre rôle."""
    out = text
    low = out.lower()
    for m in _TURN_MARKERS_FR:
        j = low.find(m.lower())
        if j != -1:
            out = out[:j].strip()
            low = out.lower()
    return out.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Génération GPT-2 (modèle local ./model)")
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dossier du modèle (défaut : ./model à côté de ce script)",
    )
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.65,
        help="Plus bas = plus prévisible (souvent un peu plus lisible avec un petit modèle).",
    )
    p.add_argument("--top_p", type=float, default=0.92)
    p.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Utilisé seulement en échantillonnage (temperature > 0).",
    )
    p.add_argument("--repetition_penalty", type=float, default=1.25)
    p.add_argument(
        "--no_repeat_ngram",
        type=int,
        default=4,
        help="Taille des n-grammes à ne pas répéter (0 = désactive).",
    )
    p.add_argument(
        "--format",
        choices=("fr", "opus", "raw"),
        default="fr",
        help=(
            "fr = Utilisateur/Assistant (French Instruct) ; "
            "opus = PROBLEM/THINKING (ancien dataset) ; "
            "raw = votre texte tel quel."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = args.model_dir or os.path.join(base, "model")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Chargement du modèle (peut prendre quelques secondes)…")
    model, tokenizer = load_model_and_tokenizer(model_dir, map_location=device)

    temperature = float(args.temperature)
    print_session_help(args, device, temperature)

    while True:
        try:
            line = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            sys.exit(0)

        if not line:
            continue
        if line in ("/quit", "/exit", ":q"):
            print("Au revoir.")
            break

        if line in ("/help", "/aide", "/?"):
            print_session_help(args, device, temperature)
            continue

        if line.startswith("/temp"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    temperature = max(0.05, float(parts[1]))
                    print(f"Température = {temperature}")
                except ValueError:
                    print("Usage : /temp 0.8")
            else:
                print("Usage : /temp 0.8")
            continue

        if args.format == "fr":
            prompt = format_chat_prompt_fr(line)
        elif args.format == "opus":
            prompt = format_chat_prompt_opus(line)
        else:
            prompt = line

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        gen_kw: dict = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": float(args.repetition_penalty),
        }
        if args.no_repeat_ngram and args.no_repeat_ngram > 0:
            gen_kw["no_repeat_ngram_size"] = int(args.no_repeat_ngram)

        if temperature and temperature > 0:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = float(temperature)
            gen_kw["top_p"] = float(args.top_p)
            gen_kw["top_k"] = int(args.top_k)
        else:
            gen_kw["do_sample"] = False

        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kw)

        # Uniquement les nouveaux tokens (évite les bugs si le décodage ne recolle pas au caractère près)
        new_ids = out_ids[0, prompt_len:]
        reply = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if args.format == "fr":
            reply = _clip_one_assistant_turn(reply)
        print(f"Bot : {reply}\n")


if __name__ == "__main__":
    main()
