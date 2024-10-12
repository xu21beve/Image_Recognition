import cv2
import numpy as np
import glob
import os

def resize_to_fit(image, diameter):
    h, w = image.shape[:2]
    scale = diameter / min(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

def crop_to_circle(image, diameter):
    # Resize the image to fit the circle
    image = resize_to_fit(image, diameter)
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = diameter // 2

    # Create a circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)

    # Apply the mask to create an RGBA image
    if image.shape[2] == 3:  # If the image is RGB
        b, g, r = cv2.split(image)
        alpha = np.zeros_like(b)
        alpha[mask == 255] = 255
        result_with_alpha = cv2.merge((b, g, r, alpha))
    else:  # If the image already has an alpha channel
        b, g, r, a = cv2.split(image)
        alpha = np.zeros_like(b)
        alpha[mask == 255] = 255
        result_with_alpha = cv2.merge((b, g, r, alpha))

    # Crop the image to the bounding box of the circle
    x1, y1 = center[0] - radius, center[1] - radius
    x2, y2 = center[0] + radius, center[1] + radius
    cropped = result_with_alpha[y1:y2, x1:x2]

    # Resize the cropped image to the desired diameter
    resized_cropped = cv2.resize(cropped, (diameter, diameter))

    return resized_cropped

# Paths
food_images_path = '../food_photos/original/*'
cropped_images_path = '../food_photos/cropped_png'
os.makedirs(cropped_images_path, exist_ok=True)

# Diameter of the circle
circle_diameter = 200  # Adjust this value as needed

# Process each image
for img_path in glob.glob(food_images_path):
    if img_path.endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        cropped_image = crop_to_circle(image, circle_diameter)

        # Ensure the output file is PNG to preserve transparency
        filename = os.path.basename(img_path)
        output_path = os.path.join(cropped_images_path, f'cropped_{os.path.splitext(filename)[0]}.png')
        cv2.imwrite(output_path, cropped_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

print(f"Cropped images saved to {cropped_images_path}")
