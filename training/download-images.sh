#!/bin/bash

# File containing the list of image URLs
URL_FILE="./food-links"

# Directory to save the downloaded images
DOWNLOAD_DIR="../food_photos"

# Create the download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Read the file line by line
while IFS= read -r url; do
    if [[ ! -z "$url" ]]; then
        echo "Downloading $url"
        wget -P "$DOWNLOAD_DIR" "$url"
    fi
done < "$URL_FILE"
