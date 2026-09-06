from pathlib import Path

from PIL import Image

GALLERY_DIR = Path(__file__).resolve().parent
REPO_ROOT = GALLERY_DIR.parent


def generate_gallery_thumbnails(
    data_dir: Path = REPO_ROOT / "data",
    gallery_images_dir: Path = GALLERY_DIR / "images",
    max_size: tuple[int, int] = (400, 400),
    jpeg_quality: int = 80,
):
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for split in ["val", "train"]:
        src_dir = data_dir / split
        if not src_dir.exists():
            print(f"Skipping [{split}]: {src_dir} does not exist.")
            continue

        dest_dir = gallery_images_dir / split
        dest_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in src_dir.glob("*/*.*") if p.suffix.lower() in valid_exts]
        print(f"Processing {len(images)} images for [{split}]...")

        count = 0
        for img_path in images:
            target_path = dest_dir / img_path.name
            try:
                with Image.open(img_path) as img:
                    rgb_img = img.convert("RGB")
                    rgb_img.thumbnail(max_size)
                    rgb_img.save(target_path, "JPEG", quality=jpeg_quality)
                    count += 1
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")

        total_mb = sum(f.stat().st_size for f in dest_dir.glob("*.*")) / (1024 * 1024)
        print(
            f"Generated {count} thumbnails for [{split}] ({total_mb:.2f} MB total in {dest_dir})\n"
        )


if __name__ == "__main__":
    generate_gallery_thumbnails()
