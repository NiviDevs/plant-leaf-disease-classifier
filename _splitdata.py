import shutil
import random
from pathlib import Path


def split_dataset(
    src_dir,
    out_dir="data",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    img_extensions=(".jpg", ".jpeg", ".png"),
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )
    random.seed(42)

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    class_dirs = [d for d in src_dir.iterdir() if d.is_dir()]

    for cls_path in class_dirs:
        cls_name = cls_path.name
        images = [f for f in cls_path.glob("*") if f.suffix.lower() in img_extensions]

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split, files in splits.items():
            split_dir = out_dir / split / cls_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for file in files:
                shutil.copy(file, split_dir / file.name)

        print(
            f"{cls_name}: {n} images train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )


if __name__ == "__main__":
    raw_data_path = "data/raw"
    final_output_path = "data"

    split_dataset(
        src_dir=raw_data_path,
        out_dir=final_output_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
