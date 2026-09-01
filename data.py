import shutil
from pathlib import Path

import kagglehub


def main():
    print("Downloading dataset from Kaggle...")
    raw = Path(
        kagglehub.dataset_download("prodigyanalysis/improper-scooter-parking-detection")
    )
    yolo_dir = next(iter(raw.rglob("labels"))).parent  # Find the YOLO dataset directory
    proper_stems = {p.stem for p in (raw / "Images" / "Proper").glob("*")}

    out = Path("data")
    for split in ["train", "val"]:
        counts = {"proper": 0, "improper": 0}
        for img in (yolo_dir / "images" / split).glob("*"):
            lbl_file = yolo_dir / "labels" / split / f"{img.stem}.txt"
            is_proper = (img.stem in proper_stems) or (
                lbl_file.exists() and lbl_file.read_text().startswith("0")
            )
            cls = "proper" if is_proper else "improper"

            dest_dir = out / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest_dir / img.name)
            counts[cls] += 1

        print(f"[{split}] {counts['proper']} proper, {counts['improper']} improper")

    print("Dataset ready at data/")


if __name__ == "__main__":
    main()
