"""
DermAI - Skin Image Preprocessing (local VS Code version)

Usage:
    python dermai.py
    python dermai.py "Skin_Diseases.jpg"

Place images in the same folder as this script, or pass an image/folder path.
Outputs are saved in ./processed_images/

Note: This script preprocesses images only; it does not diagnose skin disease.
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = 224
OUTPUT_DIR = Path("processed_images")
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def center_crop(image):
    """Crop an image to a centered square."""
    height, width = image.shape[:2]
    side = min(height, width)
    y = (height - side) // 2
    x = (width - side) // 2
    return image[y:y + side, x:x + side]


def inspect_image(image, path):
    """Return basic image-quality information."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    height, width = image.shape[:2]

    status = "OK"
    if min(height, width) < IMAGE_SIZE:
        status = "Low resolution"
    elif brightness < 60:
        status = "Too dark"
    elif brightness > 200:
        status = "Overexposed"

    return {
        "filename": str(path),
        "status": status,
        "width": width,
        "height": height,
        "size_kb": round(Path(path).stat().st_size / 1024, 1),
        "blur": round(blur_score, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
    }


def preprocess_image(path, size=IMAGE_SIZE):
    """
    Process one image:
    validate -> quality info -> center crop -> resize -> hair removal
    -> CLAHE contrast enhancement -> sharpening -> normalize.

    Returns a dictionary with processed images and metadata, or None on failure.
    """
    path = Path(path)

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        print(f"Skipped {path}: unsupported file format.")
        return None

    image = cv2.imread(str(path))
    if image is None:
        print(f"Could not read image: {path}")
        return None

    info = inspect_image(image, path)
    print(
        f"\n{path.name}: {info['width']}x{info['height']}, "
        f"{info['size_kb']} KB, brightness={info['brightness']}, "
        f"contrast={info['contrast']}, blur={info['blur']}, "
        f"quality status={info['status']}"
    )
    if info["status"] != "OK":
        print("Warning: image quality check flagged this image; processing will continue.")

    cropped = center_crop(image)
    resized = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

    # Detect dark hair-like structures using a black-hat morphology operation.
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    hair_pixels = int(np.count_nonzero(hair_mask))

    if hair_pixels > 200:
        hair_free = cv2.inpaint(resized, hair_mask, 2, cv2.INPAINT_TELEA)
    else:
        hair_free = resized.copy()

    # CLAHE enhancement on the lightness channel in LAB color space.
    lab = cv2.cvtColor(hair_free, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = CLAHE.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpen using an unsharp mask.
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)

    # Model-ready pixel values in RGB order, range 0.0 to 1.0.
    final_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
    normalized = final_rgb.astype(np.float32) / 255.0

    return {
        "original": image,
        "resized": resized,
        "hair_mask": hair_mask,
        "hair_removed": hair_free,
        "enhanced": enhanced,
        "final_bgr": sharpened,
        "final_rgb": final_rgb,
        "normalized": normalized,
        "info": info,
        "hair_pixels": hair_pixels,
        "blur_original": info["blur"],
    }


def show_steps(result, title="DermAI preprocessing"):
    """Display the main preprocessing stages."""
    steps = [
        (cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB), "1. Original"),
        (cv2.cvtColor(result["resized"], cv2.COLOR_BGR2RGB), "2. Crop + resize"),
        (result["hair_mask"], f"3. Hair mask ({result['hair_pixels']} px)"),
        (cv2.cvtColor(result["hair_removed"], cv2.COLOR_BGR2RGB), "4. Hair removal"),
        (cv2.cvtColor(result["enhanced"], cv2.COLOR_BGR2RGB), "5. CLAHE"),
        (result["final_rgb"], "6. Sharpened final"),
    ]

    plt.figure(figsize=(18, 4.5))
    plt.suptitle(title)
    for index, (image, label) in enumerate(steps, start=1):
        plt.subplot(1, 6, index)
        plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
        plt.title(label, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_image_paths(input_path):
    """Accept a single image path or a directory containing images."""
    path = Path(input_path)
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    if path.is_dir():
        return sorted(
            item for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []




def main():
    import tkinter as tk
    from tkinter import filedialog

    # Open a dialog to select one image
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected_file = filedialog.askopenfilename(
        title="Select a skin image",
        initialdir=str(Path(__file__).resolve().parent),
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ]
    )

    root.destroy()

    if not selected_file:
        print("No image selected. Exiting.")
        return

    path = Path(selected_file)
    print(f"\nSelected image: {path.name}")

    result = preprocess_image(path)

    if result is None:
        print("Image preprocessing failed.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{path.stem}_processed.jpg"

    if not cv2.imwrite(str(output_path), result["final_bgr"]):
        print(f"Could not save image: {output_path}")
        return

    normalized = result["normalized"]

    print(f"Saved processed image: {output_path.resolve()}")
    print(f"Normalized shape: {normalized.shape}")
    print(
        f"Normalized range: {normalized.min():.3f} "
        f"to {normalized.max():.3f}"
    )

    # Display only the selected image and its preprocessing stages
    show_steps(result, title=path.name)

    print("\nImage processing completed successfully.")








if __name__ == "__main__":
    main()
