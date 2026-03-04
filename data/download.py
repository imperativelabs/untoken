import argparse
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "meetingbank": {"path": "microsoft/MeetingBank-LLMCompressed"},
    "cnn_dailymail": {"path": "abisee/cnn_dailymail", "name": "3.0.0"},
    "gsm8k": {"path": "gsm8k", "name": "main"},
}


def download_all(names: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}, skipping")
            continue
        kwargs = DATASETS[name].copy()
        print(f"Downloading {name} from {kwargs['path']}...")
        ds = load_dataset(**kwargs)
        save_path = output_dir / name
        ds.save_to_disk(str(save_path))
        print(f"  Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--output-dir", default="/data/untoken")
    args = parser.parse_args()
    download_all(args.datasets, Path(args.output_dir))
