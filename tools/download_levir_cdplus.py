from pathlib import Path
from datasets import load_dataset, Image as HFImage
from PIL import Image

LEVIR_CDPlus = load_dataset("blanchon/LEVIR_CDPlus")

out_root = Path("/home/sasha/MasksComp/LEVIR_CDPlus_export")
out_root.mkdir(parents=True, exist_ok=True)

for split_name, ds in LEVIR_CDPlus.items():
    img_cols = [name for name, feat in ds.features.items() if isinstance(feat, HFImage)]
    print(f"\n[{split_name}] image columns: {img_cols}, rows: {len(ds)}")

    # Создаём папки под каждую image-колонку
    for col in img_cols:
        (out_root / split_name / col).mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(ds):
        # Базовое имя файла
        base_name = f"{i:06d}"

        # Если есть какой-то id/имя, можно использовать его
        for id_key in ("id", "name", "filename", "image_id"):
            if id_key in row and row[id_key] is not None:
                base_name = str(row[id_key]).replace("/", "_")
                break

        for col in img_cols:
            img = row[col]  # PIL.Image при decode=True
            if not isinstance(img, Image.Image):
                # На случай если вдруг вернулась dict/bytes (редко)
                continue

            save_path = out_root / split_name / col / f"{base_name}.png"
            img.save(save_path)

        if (i + 1) % 50 == 0:
            print(f"  saved {i + 1}/{len(ds)}")

print(f"\nDone. Exported to: {out_root}")