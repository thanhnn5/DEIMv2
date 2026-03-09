"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

COCO Dataset Validator

This script validates a COCO format dataset and provides statistics.

Usage:
    python tools/dataset/validate_coco.py \
        --coco_root /path/to/coco/dataset \
        --splits train val
"""

import json
import os
import argparse
from collections import defaultdict


def validate_coco_split(images_dir, ann_file, split_name):
    """
    Validate a single COCO split and print statistics.
    
    Args:
        images_dir: Directory containing images
        ann_file: COCO annotation JSON file
        split_name: Name of the split (train/val/test)
    """
    print(f"\n{'='*60}")
    print(f"Validating {split_name} split")
    print(f"{'='*60}")
    
    # Check if annotation file exists
    if not os.path.exists(ann_file):
        print(f"❌ Error: Annotation file not found: {ann_file}")
        return False
    
    # Check if images directory exists
    if not os.path.exists(images_dir):
        print(f"❌ Error: Images directory not found: {images_dir}")
        return False
    
    # Load annotations
    print(f"Loading annotations from {ann_file}...")
    try:
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False
    
    # Validate structure
    required_keys = ['images', 'annotations', 'categories']
    for key in required_keys:
        if key not in coco_data:
            print(f"❌ Error: Missing required key '{key}' in annotation file")
            return False
    
    print("✓ Annotation file structure is valid")
    
    # Statistics
    num_images = len(coco_data['images'])
    num_annotations = len(coco_data['annotations'])
    num_categories = len(coco_data['categories'])
    
    print(f"\nDataset Statistics:")
    print(f"  Images: {num_images}")
    print(f"  Annotations: {num_annotations}")
    print(f"  Categories: {num_categories}")
    
    if num_images > 0:
        print(f"  Avg annotations per image: {num_annotations / num_images:.2f}")
    
    # Category statistics
    print(f"\nCategories:")
    category_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        category_counts[ann['category_id']] += 1
    
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    for cat_id in sorted(category_counts.keys()):
        cat_name = category_id_to_name.get(cat_id, f"Unknown (ID: {cat_id})")
        count = category_counts[cat_id]
        print(f"  {cat_id}: {cat_name} - {count} instances")
    
    # Validate image files
    print(f"\nValidating image files...")
    missing_images = []
    for img_info in coco_data['images']:
        img_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            missing_images.append(img_info['file_name'])
    
    if missing_images:
        print(f"❌ Warning: {len(missing_images)} image files not found")
        if len(missing_images) <= 10:
            for img_name in missing_images:
                print(f"  - {img_name}")
        else:
            print(f"  First 10 missing images:")
            for img_name in missing_images[:10]:
                print(f"  - {img_name}")
    else:
        print(f"✓ All {num_images} image files found")
    
    # Validate annotations
    print(f"\nValidating annotations...")
    invalid_annotations = []
    for ann in coco_data['annotations']:
        # Check required fields
        required_ann_fields = ['id', 'image_id', 'category_id', 'bbox', 'area']
        for field in required_ann_fields:
            if field not in ann:
                invalid_annotations.append(f"Annotation {ann.get('id', 'unknown')} missing field '{field}'")
                break
        
        # Validate bbox format
        if 'bbox' in ann:
            bbox = ann['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                invalid_annotations.append(f"Annotation {ann['id']} has invalid bbox format")
            elif any(x < 0 for x in bbox):
                invalid_annotations.append(f"Annotation {ann['id']} has negative bbox values")
    
    if invalid_annotations:
        print(f"❌ Warning: {len(invalid_annotations)} invalid annotations found")
        if len(invalid_annotations) <= 10:
            for msg in invalid_annotations:
                print(f"  - {msg}")
        else:
            print(f"  First 10 issues:")
            for msg in invalid_annotations[:10]:
                print(f"  - {msg}")
    else:
        print(f"✓ All {num_annotations} annotations are valid")
    
    print(f"\n{'='*60}")
    if not missing_images and not invalid_annotations:
        print(f"✓ {split_name} split validation PASSED")
    else:
        print(f"⚠ {split_name} split validation completed with warnings")
    print(f"{'='*60}")
    
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Validate COCO format dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--coco_root',
        type=str,
        required=True,
        help='Root directory of COCO format dataset'
    )
    
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val'],
        help='Dataset splits to validate (default: train val)'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("COCO Dataset Validator")
    print(f"Dataset root: {args.coco_root}")
    
    all_valid = True
    for split in args.splits:
        images_dir = os.path.join(args.coco_root, 'images', split)
        ann_file = os.path.join(args.coco_root, 'annotations', f'instances_{split}.json')
        
        valid = validate_coco_split(images_dir, ann_file, split)
        all_valid = all_valid and valid
    
    print("\n" + "="*60)
    if all_valid:
        print("✓ All splits validated successfully!")
    else:
        print("⚠ Some splits had validation errors")
    print("="*60)


if __name__ == '__main__':
    main()

