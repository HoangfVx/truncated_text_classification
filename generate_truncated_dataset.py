import os
import cv2
import numpy as np
import random
import string

# Define parameters
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]  # Different fonts
OUTPUT_DIR = "truncated_text_dataset"
TRUNCATED_IMAGE_SIZE = (64, 64)  # Fixed size for truncated images

# Create directories
normal_dir = os.path.join(OUTPUT_DIR, "normal")
truncated_dir = os.path.join(OUTPUT_DIR, "truncated")
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(truncated_dir, exist_ok=True)

# Generate dataset for A-Z, a-z, 0-9
characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9

def crop_to_character(img):
    """Remove white spaces around the character by finding its bounding box."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # Invert colors for better detection
    coords = cv2.findNonZero(thresh)  # Get all non-zero points
    x, y, w, h = cv2.boundingRect(coords)  # Get bounding box
    return img[y:y+h, x:x+w]  # Crop image to character region

def create_character_image(character, font, crop=False):
    """Generate an image of a character, with dynamic size for normal and fixed size for truncated."""
    font_scale = 2
    thickness = 3

    # Get text size
    text_size, baseline = cv2.getTextSize(character, font, font_scale, thickness)
    text_width, text_height = text_size

    # Create an image slightly larger than text to ensure we capture the full character
    img_width, img_height = text_width + 20, text_height + baseline + 20  # Add padding
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # White background
    text_x, text_y = 10, text_height + 10  # Offset text position slightly

    # Put text
    cv2.putText(img, character, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    # Crop the normal image tightly to remove excess white space
    img = crop_to_character(img)

    if crop:
        # Apply random cropping
        h, w = img.shape[:2]
        crop_x1 = random.randint(0, w // 4)  # Crop up to 1/4 from left
        crop_x2 = random.randint(3 * w // 4, w)  # Crop up to 1/4 from right
        crop_y1 = random.randint(0, h // 4)  # Crop up to 1/4 from top
        crop_y2 = random.randint(3 * h // 4, h)  # Crop up to 1/4 from bottom
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize back to fixed truncated size
        img = cv2.resize(img, TRUNCATED_IMAGE_SIZE)

        # Crop whitespace again to ensure the character fills the space
        img = crop_to_character(img)

    return img

# Generate images
for character in characters:
    for i in range(3):  # Create multiple variations
        font = random.choice(FONTS)

        # Normal image (dynamically sized, no extra space)
        normal_img = create_character_image(character, font, crop=False)
        normal_path = os.path.join(normal_dir, f"{character}_{i}.png")
        cv2.imwrite(normal_path, normal_img)

        # Truncated image (fixed size, but no extra white space)
        truncated_img = create_character_image(character, font, crop=True)
        truncated_path = os.path.join(truncated_dir, f"{character}_{i}.png")
        cv2.imwrite(truncated_path, truncated_img)

print("Dataset generation complete. Images saved in:", OUTPUT_DIR)
