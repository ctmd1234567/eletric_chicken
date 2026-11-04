"""Build an isolated B3 YOLO data config by appending generated MLS images."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_image_list(value, base_root: Path, config_dir: Path):
    if isinstance(value, list):
        return [_resolve(base_root, item) for item in value]
    path = _resolve(base_root, str(value))
    if not path.is_file():
        path = _resolve(config_dir, str(value))
    if not path.is_file():
        raise FileNotFoundError(f"Training/validation list does not exist: {value}")
    return [_resolve(path.parent, line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build(base_data: Path, generated_root: Path, output: Path, *, repeat=1):
    base_data = base_data.resolve()
    generated_root = generated_root.resolve()
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    if repeat < 1:
        raise ValueError("repeat must be positive")
    config = yaml.safe_load(base_data.read_text(encoding="utf-8"))
    config_dir = base_data.parent
    base_root = Path(config.get("path", config_dir))
    if not base_root.is_absolute():
        base_root = (config_dir / base_root).resolve()
    else:
        base_root = base_root.resolve()

    if "train" not in config or "val" not in config:
        raise ValueError("Base data.yaml must define train and val")
    base_train = _read_image_list(config["train"], base_root, config_dir)
    _read_image_list(config["val"], base_root, config_dir)
    if config.get("test"):
        _read_image_list(config["test"], base_root, config_dir)
    base_val = _read_image_list(config["val"], base_root, config_dir)
    base_test = _read_image_list(config["test"], base_root, config_dir) if config.get("test") else []

    manifest_path = generated_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MLS generation manifest not found: {manifest_path}")
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("MLS manifest contains no generated samples")
    augmented = []
    checked_banks = set()
    for record in records:
        if record.get("algorithm") == "det_mscea":
            review = record.get("human_review", {})
            if (review.get("status") != "approved" or not review.get("reviewer", "").strip()
                    or not review.get("notes", "").strip()):
                raise ValueError("Det-MSCEA sample requires approved human_review with reviewer and notes")
            if (record.get("constraint_version") != 2
                    or record.get("feature_pooling") != "fractional_instance_mask"):
                raise ValueError("Det-MSCEA requires mask pooling and version 2 fire-aware constraints")
            source = _resolve(generated_root, record["source_image"])
            if source not in base_train or source in base_val or source in base_test:
                raise ValueError("Det-MSCEA source must belong exclusively to the base training split")
            bank = _resolve(generated_root, record["reference_bank"])
            bank_key = (bank, record["reference_bank_sha256"])
            if bank_key not in checked_banks:
                if _sha256(bank) != bank_key[1]:
                    raise ValueError("Det-MSCEA reference bank hash mismatch")
                for reference in json.loads(bank.read_text(encoding="utf-8")):
                    reference_image = _resolve(generated_root, reference["image"])
                    if (reference_image not in base_train or reference_image in base_val
                            or reference_image in base_test):
                        raise ValueError("Det-MSCEA reference bank must use training images exclusively")
                    if _sha256(reference_image) != reference["image_sha256"]:
                        raise ValueError("Det-MSCEA reference image hash mismatch")
                checked_banks.add(bank_key)
        image = _resolve(generated_root, record["image"])
        label = _resolve(generated_root, record["label"])
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(f"Generated image/label pair is incomplete: {image}")
        if record.get("algorithm") == "det_mscea":
            for key in ("image", "mask", "label"):
                artifact = _resolve(generated_root, record[key])
                if _sha256(artifact) != record.get("output_sha256", {}).get(key):
                    raise ValueError(f"Det-MSCEA reviewed {key} hash mismatch")
                source = _resolve(generated_root, record[f"source_{key}"])
                if _sha256(source) != record.get("source_sha256", {}).get(key):
                    raise ValueError(f"Det-MSCEA source {key} hash mismatch")
        image_parts = list(image.parts)
        try:
            index = len(image_parts) - 1 - image_parts[::-1].index("images")
        except ValueError as exc:
            raise ValueError(f"Generated image must be under an images directory: {image}") from exc
        expected = Path(*image_parts[:index], "labels", *image_parts[index + 1:]).with_suffix(".txt")
        if expected.resolve() != label.resolve():
            raise ValueError(f"Generated YOLO label path mismatch: {expected} != {label}")
        augmented.append(image)

    output.mkdir(parents=True)
    train_list = output / "train_b3.txt"
    all_train = base_train + [image for image in augmented for _ in range(repeat)]
    train_list.write_text("\n".join(path.as_posix() for path in all_train) + "\n", encoding="utf-8")
    data_b3 = dict(
        path=base_root.as_posix(),
        train=train_list.as_posix(),
        val=config["val"],
        names=config["names"],
    )
    if config.get("test"):
        data_b3["test"] = config["test"]
    data_path = output / "data_b3.yaml"
    data_path.write_text(yaml.safe_dump(data_b3, sort_keys=False, allow_unicode=True), encoding="utf-8")
    summary = dict(
        status="ready",
        base_data=str(base_data),
        base_data_sha256=_sha256(base_data),
        base_train_images=len(base_train),
        generated_images=len(augmented),
        generated_repeat=repeat,
        final_train_entries=len(all_train),
        validation_images=len(base_val),
        test_images=len(base_test),
        data_yaml=str(data_path),
        limitations=[
            "Only generated MLS images are appended; validation and test sets remain unchanged.",
            "The generated samples must be visually audited before training.",
            "This is B3 random local-MLS augmentation, not the full evolutionary MSCEA method.",
        ],
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)
    build(args.base_data, args.generated_root, args.out, repeat=args.repeat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
