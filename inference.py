import os
import sys
import shutil
import subprocess
from pathlib import Path
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Input and output folders are beside inference.py
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# Your trained Pix2Pix model
MODEL_NAME = "CCTV_FINAL_V2"

# Checkpoint to use
EPOCH = "70"

# Temporary dataset
TEMP_DIR = BASE_DIR / "_inference_temp"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# CREATE INPUT / OUTPUT FOLDERS
# ============================================================

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# FIND INPUT IMAGES
# ============================================================

input_images = [
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]


if not input_images:

    print()
    print("=" * 60)
    print("ERROR: No IR images found.")
    print("=" * 60)
    print()
    print("Put your IR image inside:")
    print(INPUT_DIR)
    print()
    print("Example:")
    print("input/test.jpg")
    print()

    exit(1)


print()
print("=" * 60)
print("       CCTV IR → RGB INFERENCE")
print("=" * 60)
print()
print("Model :", MODEL_NAME)
print("Epoch :", EPOCH)
print("Input :", INPUT_DIR)
print("Output:", OUTPUT_DIR)
print("Images:", len(input_images))
print()


# ============================================================
# CREATE TEMPORARY PIX2PIX DATASET
# ============================================================

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

TEMP_TEST_DIR = TEMP_DIR / "test"

TEMP_TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CREATE ALIGNED IMAGES
# IR | DUMMY RGB
# ============================================================

for image_path in input_images:

    try:

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        # Dummy B image
        dummy_b = Image.new(
            "RGB",
            (width, height),
            (0, 0, 0)
        )

        # Create:
        #
        # LEFT  = IR
        # RIGHT = dummy B
        #
        aligned = Image.new(
            "RGB",
            (width * 2, height)
        )

        aligned.paste(image, (0, 0))
        aligned.paste(dummy_b, (width, 0))

        save_path = TEMP_TEST_DIR / image_path.name

        aligned.save(
            save_path,
            quality=95
        )

        print("Prepared:", image_path.name)

    except Exception as e:

        print(
            f"ERROR preparing {image_path.name}: {e}"
        )


# ============================================================
# RUN PIX2PIX
# ============================================================

print()
print("=" * 60)
print("Running Pix2Pix...")
print("=" * 60)
print()


command = [
    sys.executable,
    "test.py",

    "--dataroot",
    str(TEMP_DIR),

    "--name",
    MODEL_NAME,

    "--model",
    "pix2pix",

    "--dataset_mode",
    "aligned",

    "--direction",
    "AtoB",

    "--gpu_ids",
    "0",

    "--preprocess",
    "resize_and_crop",

    "--load_size",
    "286",

    "--crop_size",
    "256",

    "--num_test",
    str(len(input_images)),

    "--epoch",
    EPOCH
]


result = subprocess.run(command)


if result.returncode != 0:

    print()
    print("Pix2Pix failed.")
    print()

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    exit(result.returncode)


# ============================================================
# FIND RESULTS
# ============================================================

possible_result_dirs = [

    BASE_DIR
    / "results"
    / MODEL_NAME
    / f"test_{EPOCH}"
    / "images",

    BASE_DIR
    / "results"
    / MODEL_NAME
    / "test_latest"
    / "images"
]


RESULT_DIR = None


for directory in possible_result_dirs:

    if directory.exists():

        RESULT_DIR = directory
        break


if RESULT_DIR is None:

    print()
    print("ERROR: Generated result folder not found.")
    print()

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    exit(1)


# ============================================================
# COPY GENERATED RGB IMAGES
# ============================================================

generated_count = 0


for fake_image in RESULT_DIR.glob("*_fake_B.png"):

    destination = OUTPUT_DIR / fake_image.name

    shutil.copy2(
        fake_image,
        destination
    )

    generated_count += 1


# ============================================================
# CLEAN TEMPORARY DATA
# ============================================================

if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("              COMPLETE")
print("=" * 60)
print()
print("Generated:", generated_count)
print()
print("RGB images saved to:")
print(OUTPUT_DIR)
print()
print("=" * 60)
print()