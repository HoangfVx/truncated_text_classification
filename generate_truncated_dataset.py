import os 
import cv2 
import numpy as np
import random 
import string 

IMAGE_SIZE = (64, 64)
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]  # Different fonts
OUTPUT_DIR = "truncated_text_dataset"

# Create directories
normal_dir = os.path.join(OUTPUT_DIR, "normal")
truncated_dir = os.path.join(OUTPUT_DIR, "truncated")
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(truncated_dir, exist_ok=True)

# Generate dataset for A-Z, a-z, 0-9
characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9

def create_character_image(character, font, crop=False):
    """Generate an image of a character, optionally truncated."""
    img = np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8) * 255  # White background
    font_scale = 2
    thickness = 3

    # Get text size
    text_size = cv2.getTextSize(character, font, font_scale, thickness)[0]
    text_x = (IMAGE_SIZE[0] - text_size[0]) // 2
    text_y = (IMAGE_SIZE[1] + text_size[1]) // 2

    # Put text
    cv2.putText(img, character, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

    # Apply random cropping if needed
    if crop:
        crop_x1 = random.randint(0, IMAGE_SIZE[0] // 3)
        crop_x2 = random.randint(2 * IMAGE_SIZE[0] // 3, IMAGE_SIZE[0])
        crop_y1 = random.randint(0, IMAGE_SIZE[1] // 3)
        crop_y2 = random.randint(2 * IMAGE_SIZE[1] // 3, IMAGE_SIZE[1])
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize back to fixed size
        img = cv2.resize(img, IMAGE_SIZE)

    return img

# Generate images
for character in characters:
    for i in range(3):  # Create multiple variations
        font = random.choice(FONTS)

        # Normal image
        normal_img = create_character_image(character, font, crop=False)
        normal_path = os.path.join(normal_dir, f"{character}_{i}.png")
        cv2.imwrite(normal_path, normal_img)

        # Truncated image
        truncated_img = create_character_image(character, font, crop=True)
        truncated_path = os.path.join(truncated_dir, f"{character}_{i}.png")
        cv2.imwrite(truncated_path, truncated_img)

print("Dataset generation complete. Images saved in:", OUTPUT_DIR)