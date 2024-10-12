import cv2
import glob
import os

def convert_to_png(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Process each JPEG/JPG image
    for img_path in glob.glob(os.path.join(input_dir, '*')):
        if img_path.lower().endswith(('.jpeg', '.jpg')):
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if image is None:
                print(f"Failed to read image: {img_path}")
                continue

            filename = os.path.basename(img_path)
            filename_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f'{filename_without_ext}.png')

            cv2.imwrite(output_path, image)
            print(f"Converted {img_path} to {output_path}")

# Paths
input_directory = 'food_photos/cropped'
output_directory = 'food_photos/cropped_png'

convert_to_png(input_directory, output_directory)
