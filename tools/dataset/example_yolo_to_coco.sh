#!/bin/bash

# Example script for converting YOLO format to COCO format
# This is a template - modify the paths according to your dataset

# Set your paths here
YOLO_ROOT="/path/to/your/yolo/dataset"
OUTPUT_ROOT="/path/to/output/coco/dataset"
CLASS_NAMES_FILE="/path/to/your/classes.txt"

# Optional: Create a sample classes.txt file if you don't have one
# Uncomment and modify the following lines:
# cat > classes.txt << EOF
# person
# car
# bicycle
# dog
# cat
# EOF
# CLASS_NAMES_FILE="classes.txt"

# Run the conversion
python tools/dataset/yolo_to_coco.py \
    --yolo_root "$YOLO_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --class_names_file "$CLASS_NAMES_FILE" \
    --splits train val

echo ""
echo "Conversion complete!"
echo "Your COCO dataset is ready at: $OUTPUT_ROOT"
echo ""
echo "Next steps:"
echo "1. Update your config file (e.g., configs/dataset/custom_detection.yml)"
echo "2. Set the correct paths for img_folder and ann_file"
echo "3. Set num_classes to match your dataset"
echo "4. Set remap_mscoco_category: False"

