import cv2
import numpy as np
import albumentations as A
import glob
import os

# Load images
container_images_path = 'containers-testing/*.jpg'
food_images_path = 'food-testing/*.png'  # Food images in .png format
lid_image_path = '../takeaway-top.png'  # Assuming a single lid image

container_images = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in glob.glob(container_images_path)]
food_images = [cv2.imread(img, cv2.IMREAD_UNCHANGED) for img in glob.glob(food_images_path)]  # Load food images with alpha channel
lid_img = cv2.imread(lid_image_path, cv2.IMREAD_UNCHANGED)  # Load the lid image with alpha channel

# Albumentations pipeline for additional augmentations
augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.2),
    A.RGBShift(p=0.0)
])

def find_container_center_and_size(container_img):
    # Convert image to grayscale and apply threshold
    if container_img.shape[2] == 4:  # RGBA
        gray = cv2.cvtColor(container_img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:  # RGB
        gray = cv2.cvtColor(container_img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the container
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    center = (x + w // 2, y + h // 2)
    size = (w, h)

    return center, size

def place_food_in_container(container_img, food_img, lid_img):
    center, size = find_container_center_and_size(container_img)

    # Resize food image to fit within the container
    food_h, food_w = food_img.shape[:2]
    aspect_ratio = food_w / food_h

    new_food_h = size[1] // 2
    new_food_w = int(new_food_h * aspect_ratio)

    if new_food_w > size[0]:
        new_food_w = size[0]
        new_food_h = int(new_food_w / aspect_ratio)

    food_img_resized = cv2.resize(food_img, (new_food_w, new_food_h), interpolation=cv2.INTER_AREA)

    # Calculate position to center the food image horizontally and place it at the detected y position
    x_offset = container_img.shape[1] // 2 - (new_food_w // 2)
    y_offset = center[1] - (new_food_h // 2)

    # Ensure the overlay region does not go out of bounds
    if y_offset < 0:
        y_offset = 0
    if x_offset < 0:
        x_offset = 0
    if y_offset + new_food_h > container_img.shape[0]:
        y_offset = container_img.shape[0] - new_food_h
    if x_offset + new_food_w > container_img.shape[1]:
        x_offset = container_img.shape[1] - new_food_w

    # Ensure the food image has an alpha channel
    if food_img_resized.shape[2] == 3:
        food_img_resized = cv2.cvtColor(food_img_resized, cv2.COLOR_BGR2BGRA)

    # Create ROI for blending food
    roi = container_img[y_offset:y_offset + new_food_h, x_offset:x_offset + new_food_w]
    
    # Check if the ROI is in bounds
    if roi.shape[0] != food_img_resized.shape[0] or roi.shape[1] != food_img_resized.shape[1]:
        return container_img  # Skip if dimensions don't match

    alpha_channel = food_img_resized[:, :, 3] / 255.0
    for c in range(3):  # Blend only the BGR channels
        container_img[y_offset:y_offset + new_food_h, x_offset:x_offset + new_food_w, c] = \
            food_img_resized[:, :, c] * alpha_channel + \
            container_img[y_offset:y_offset + new_food_h, x_offset:x_offset + new_food_w, c] * (1.0 - alpha_channel)

    # Resize the lid to match the container size
    lid_resized = cv2.resize(lid_img, (container_img.shape[1], container_img.shape[0]), interpolation=cv2.INTER_AREA)

    # Overlay the lid on top of the food and container
    lid_alpha = lid_resized[:, :, 3] / 255.0
    for c in range(3):  # Blend only the BGR channels
        container_img[:, :, c] = lid_resized[:, :, c] * lid_alpha + container_img[:, :, c] * (1.0 - lid_alpha)

    return container_img

# Create synthetic dataset
synthetic_dataset_path = 'overlayed_images'
os.makedirs(synthetic_dataset_path, exist_ok=True)

# Ensure that there are the same number of food and container images
num_images = min(len(container_images), len(food_images))

for idx in range(num_images):
    container_img = container_images[idx]
    food_img = food_images[idx]
    synthetic_img = place_food_in_container(container_img, food_img, lid_img)

    # Apply additional augmentations
    augmented = augmentations(image=synthetic_img)
    synthetic_img = augmented['image']

    # Save synthetic image
    output_path = os.path.join(synthetic_dataset_path, f'synthetic_{idx}.png')
    cv2.imwrite(output_path, synthetic_img)

print(f"Synthetic images saved to {synthetic_dataset_path}")
