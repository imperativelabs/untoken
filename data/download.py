import argparse
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    # Core — confirmed parquet-based, always download
    "meetingbank": {"path": "microsoft/MeetingBank-LLMCompressed"},
    "cnn_dailymail": {"path": "abisee/cnn_dailymail", "name": "3.0.0"},
    "gsm8k": {"path": "openai/gsm8k", "name": "main"},
    # v2 additions — parquet-based, no dataset scripts required
    "booksum": {"path": "kmfoda/booksum"},
    "xsum": {"path": "EdinburghNLP/xsum"},
    "dialogsum": {"path": "knkarthick/dialogsum"},
    "billsum": {"path": "FiscalNote/billsum"},
    # multi_news / samsum / scientific_papers / big_patent / wikihow / reddit_tifu
    # are either script-based or have no reliable parquet mirror on the Hub.
    # Their loaders in loaders.py return [] gracefully when the directory is absent.
}


def download_all(names: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}, skipping")
            continue
        kwargs = DATASETS[name].copy()
        print(f"Downloading {name} from {kwargs['path']}...")
        try:
            ds = load_dataset(**kwargs)
        except Exception as e:
            print(f"  [warn] {name} failed: {e}. Skipping.")
            continue
        save_path = output_dir / name
        ds.save_to_disk(str(save_path))
        print(f"  Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--output-dir", default="/data/untoken")
    args = parser.parse_args()
    download_all(args.datasets, Path(args.output_dir))
