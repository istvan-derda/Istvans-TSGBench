import os
import shutil

# Define source base directory relative to the script's location
source_base = "../../data/ori"

# Target directory
target_dir = "gen"

# Ensure target directory exists
os.makedirs(target_dir, exist_ok=True)

# Walk through the source directory and copy _train.npy files, preserving relative structure
for root, dirs, files in os.walk(source_base):
    for file in files:
        if file.endswith("_train.npy"):
            # Get relative path from source_base
            rel_path = os.path.relpath(root, source_base)
            # Create corresponding directory in target
            dst_dir = os.path.join(target_dir, rel_path)
            os.makedirs(dst_dir, exist_ok=True)
            # Copy the file
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy(src_file, dst_file)
            print(f"Copied {src_file} to {dst_file}")