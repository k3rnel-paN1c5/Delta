#!/bin/bash

# Directory containing the images
image_dir=$1

counter=1

for file in "$image_dir"/*; do
    # Check if the file is a regular file (not a directory)
    if [[ -f "$file" ]]; then
        # Extract the file extension
        extension="${file##*.}"
        
        # new filename
        new_name="image$counter.$extension"
        
        mv "$file" "$image_dir/$new_name"
        
        counter=$((counter + 1))
    fi
done

echo "All files have been renamed sequentially."