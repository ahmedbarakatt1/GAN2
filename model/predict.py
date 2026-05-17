"""
predict.py — required inference entry point
Usage: python predict.py -i <input_file> -o <output_file>
"""
from __future__ import annotations
import argparse, torch
from pathlib import Path

from dataset import DateTokenizer
from models.transformer_model import TransformerDateGenerator

MODEL_NAME   = "transformer"
WEIGHTS_PATH = f"weights/{MODEL_NAME}_best.pt"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 64


def load_model(tokenizer: DateTokenizer) -> TransformerDateGenerator:
    model = TransformerDateGenerator(
        cond_vocab_size=tokenizer.cond_vocab_size,
        date_vocab_size=tokenizer.date_vocab_size,
    ).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def predict_batch(model, tokenizer, lines: list[str]) -> list[str]:
    encoded   = [tokenizer.encode_conditions(l) for l in lines]
    cond      = torch.tensor(encoded, dtype=torch.long, device=DEVICE)
    token_ids = model.generate(cond, tokenizer.bos_id, tokenizer.eos_id)
    return [tokenizer.decode_date(ids) for ids in token_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, help="Path to input conditions file")
    parser.add_argument("-o", required=True, help="Path to output predictions file")
    args = parser.parse_args()

    tokenizer   = DateTokenizer()
    model       = load_model(tokenizer)
    input_lines = Path(args.i).read_text().strip().splitlines()
    output_lines = []

    for i in range(0, len(input_lines), BATCH_SIZE):
        batch = input_lines[i:i + BATCH_SIZE]
        dates = predict_batch(model, tokenizer, batch)
        for line, date in zip(batch, dates):
            output_lines.append(f"{line.strip()} {date}")

    Path(args.o).write_text("\n".join(output_lines) + "\n")
    print(f"Wrote {len(output_lines)} predictions to {args.o}")


if __name__ == "__main__":
    main()
