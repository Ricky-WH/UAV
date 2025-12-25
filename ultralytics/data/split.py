# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections.abc import Sequence
import random
import shutil
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import DATASETS_DIR, LOGGER, TQDM


def split_classify_dataset(source_dir: str | Path, train_ratio: float = 0.8) -> Path:
    """
    Split classification dataset into train and val directories in a new directory.

    Creates a new directory '{source_dir}_split' with train/val subdirectories, preserving the original class
    structure with an 80/20 split by default.

    Directory structure:
        Before:
            caltech/
            ├── class1/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            ├── class2/
            │   ├── img1.jpg
            │   └── ...
            └── ...

        After:
            caltech_split/
            ├── train/
            │   ├── class1/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   ├── class2/
            │   │   ├── img1.jpg
            │   │   └── ...
            │   └── ...
            └── val/
                ├── class1/
                │   ├── img2.jpg
                │   └── ...
                ├── class2/
                │   └── ...
                └── ...

    Args:
        source_dir (str | Path): Path to classification dataset root directory.
        train_ratio (float): Ratio for train split, between 0 and 1.

    Returns:
        (Path): Path to the created split directory.

    Examples:
        Split dataset with default 80/20 ratio
        >>> split_classify_dataset("path/to/caltech")

        Split with custom ratio
        >>> split_classify_dataset("path/to/caltech", 0.75)
    """
    source_path = Path(source_dir)
    split_path = Path(f"{source_path}_split")
    train_path, val_path = split_path / "train", split_path / "val"

    # Create directory structure
    split_path.mkdir(exist_ok=True)
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    # Process class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob("*.*"))) for d in class_dirs)
    stats = f"{len(class_dirs)} classes, {total_images} images"
    LOGGER.info(f"Splitting {source_path} ({stats}) into {train_ratio:.0%} train, {1 - train_ratio:.0%} val...")

    for class_dir in class_dirs:
        # Create class directories
        (train_path / class_dir.name).mkdir(exist_ok=True)
        (val_path / class_dir.name).mkdir(exist_ok=True)

        # Split and copy files
        image_files = list(class_dir.glob("*.*"))
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)

        for img in image_files[:split_idx]:
            shutil.copy2(img, train_path / class_dir.name / img.name)

        for img in image_files[split_idx:]:
            shutil.copy2(img, val_path / class_dir.name / img.name)

    LOGGER.info(f"Split complete in {split_path} ✅")
    return split_path


def split_semisupervised_yolo(
    dataset_dir: str | Path,
    split: str = "train",
    ratio: Sequence[float] | float = (1, 3),
    seed: int = 0,
    output_dir: str | Path | None = None,
    absolute_paths: bool = True,
    annotated_only: bool = True,
) -> tuple[Path, Path]:
    """Split a YOLO-format dataset into labeled/unlabeled lists for semi-supervised workflows.

    Args:
        dataset_dir (str | Path): Root directory containing ``images/`` and ``labels/`` subfolders.
        split (str): Subdirectory under ``images/`` and ``labels/`` to process (e.g. ``train``).
        ratio (Sequence[float] | float): Labeled-to-unlabeled ratio or labeled fraction if float.
        seed (int): RNG seed used before shuffling to make the split reproducible.
        output_dir (str | Path | None): Optional directory to save ``*_labeled.txt`` and ``*_unlabeled.txt``.
        absolute_paths (bool): Write absolute image paths. If ``False`` paths are relative to ``dataset_dir``.
        annotated_only (bool): Skip images without a matching YOLO label file.

    Returns:
        tuple[Path, Path]: Paths to the generated labeled and unlabeled text files.
    """

    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images" / split
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory '{images_dir}' does not exist")

    files = sorted(x for x in images_dir.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)
    if annotated_only:
        labeled_files = []
        for img in files:
            label_path = Path(img2label_paths([str(img)])[0])
            if label_path.exists():
                labeled_files.append(img)
        files = labeled_files

    if not files:
        raise FileNotFoundError(f"No images found for split '{split}' in {dataset_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)

    if isinstance(ratio, Sequence) and not isinstance(ratio, (str, bytes)):
        if len(ratio) != 2:
            raise ValueError("ratio sequence must contain two elements: labeled and unlabeled parts")
        total = float(ratio[0] + ratio[1])
        if total <= 0:
            raise ValueError("ratio parts must sum to a positive value")
        labeled_fraction = ratio[0] / total
    else:
        labeled_fraction = float(ratio)

    if not 0 < labeled_fraction < 1:
        raise ValueError("ratio must produce a labeled fraction between 0 and 1")

    labeled_count = max(1, int(round(len(files) * labeled_fraction)))
    if labeled_count >= len(files):
        labeled_count = len(files) - 1 if len(files) > 1 else len(files)

    labeled_paths = files[:labeled_count]
    unlabeled_paths = files[labeled_count:]

    if not unlabeled_paths:
        LOGGER.warning("Unlabeled split is empty; consider adjusting the ratio or ensuring more data is available")

    output_dir = Path(output_dir) if output_dir else dataset_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_txt = output_dir / f"{split}_labeled.txt"
    unlabeled_txt = output_dir / f"{split}_unlabeled.txt"

    def _format_path(p: Path) -> str:
        if absolute_paths:
            return str(p.resolve())
        try:
            return p.relative_to(dataset_dir).as_posix()
        except ValueError:
            return p.as_posix()

    labeled_txt.write_text("\n".join(_format_path(p) for p in labeled_paths) + "\n", encoding="utf-8")
    unlabeled_txt.write_text("\n".join(_format_path(p) for p in unlabeled_paths) + "\n", encoding="utf-8")

    LOGGER.info(
        f"Created semi-supervised split at '{output_dir}': "
        f"{len(labeled_paths)} labeled / {len(unlabeled_paths)} unlabeled images (target ratio {ratio})."
    )

    return labeled_txt, unlabeled_txt


def autosplit(
    path: Path = DATASETS_DIR / "coco8/images",
    weights: tuple[float, float, float] = (0.9, 0.1, 0.0),
    annotated_only: bool = False,
) -> None:
    """
    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files.

    Args:
        path (Path): Path to images directory.
        weights (tuple): Train, validation, and test split fractions.
        annotated_only (bool): If True, only images with an associated txt file are used.

    Examples:
        Split images with default weights
        >>> from ultralytics.data.split import autosplit
        >>> autosplit()

        Split with custom weights and annotated images only
        >>> autosplit(path="path/to/images", weights=(0.8, 0.15, 0.05), annotated_only=True)
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # add image to txt file


if __name__ == "__main__":
    split_classify_dataset("caltech101")
