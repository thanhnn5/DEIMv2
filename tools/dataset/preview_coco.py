"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

COCO Dataset Preview

This script displays sample images with annotations from a COCO dataset.
Useful for visually verifying the conversion was successful.

Usage:
    python tools/dataset/preview_coco.py \
        --coco_root /path/to/coco/dataset \
        --split train \
        --num_samples 5
"""

import json
import os
import argparse
import random
from PIL import Image, ImageDraw, ImageFont


def draw_bbox(draw, bbox, label, color='red', width=3):
    """Draw a bounding box on the image."""
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h
    
    # Draw rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)
    
    # Draw label background
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    # Get text size using textbbox
    bbox_text = draw.textbbox((x_min, y_min), label, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    
    # Draw label background
    draw.rectangle(
        [x_min, y_min - text_height - 4, x_min + text_width + 4, y_min],
        fill=color
    )
    
    # Draw label text
    draw.text((x_min + 2, y_min - text_height - 2), label, fill='white', font=font)


def preview_samples(images_dir, ann_file, num_samples=5, output_dir=None):
    """
    Preview random samples from the dataset.
    
    Args:
        images_dir: Directory containing images
        ann_file: COCO annotation JSON file
        num_samples: Number of samples to preview
        output_dir: Directory to save preview images (if None, display only)
    """
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create category mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create image_id to annotations mapping
    image_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Select random images
    images_with_annotations = [
        img for img in coco_data['images'] 
        if img['id'] in image_annotations
    ]
    
    if len(images_with_annotations) == 0:
        print("No images with annotations found!")
        return
    
    num_samples = min(num_samples, len(images_with_annotations))
    sample_images = random.sample(images_with_annotations, num_samples)
    
    print(f"\nPreviewing {num_samples} random samples...")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    for idx, img_info in enumerate(sample_images, 1):
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        print(f"\n[{idx}/{num_samples}] {img_info['file_name']}")
        print(f"  Size: {img_info['width']}x{img_info['height']}")
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Get annotations for this image
        annotations = image_annotations.get(img_info['id'], [])
        print(f"  Annotations: {len(annotations)}")
        
        # Draw each annotation
        category_counts = {}
        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = category_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            
            bbox = ann['bbox']
            color = colors[cat_id % len(colors)]
            label = f"{cat_name}"
            
            draw_bbox(draw, bbox, label, color=color)
        
        # Print category distribution for this image
        for cat_name, count in sorted(category_counts.items()):
            print(f"    - {cat_name}: {count}")
        
        # Save or display
        if output_dir:
            output_path = os.path.join(output_dir, f"preview_{idx}_{img_info['file_name']}")
            img.save(output_path)
            print(f"  Saved to: {output_path}")
        else:
            # Display image (requires display capability)
            try:
                img.show()
            except:
                print("  (Cannot display image - no display available)")
    
    print(f"\nPreview complete!")
    if output_dir:
        print(f"Preview images saved to: {output_dir}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Preview samples from COCO dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--coco_root',
        type=str,
        required=True,
        help='Root directory of COCO format dataset'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split to preview (default: train)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of samples to preview (default: 5)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save preview images (default: None, display only)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sample selection (default: 42)'
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    images_dir = os.path.join(args.coco_root, 'images', args.split)
    ann_file = os.path.join(args.coco_root, 'annotations', f'instances_{args.split}.json')
    
    # Check if files exist
    if not os.path.exists(ann_file):
        print(f"Error: Annotation file not found: {ann_file}")
        return
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    preview_samples(
        images_dir=images_dir,
        ann_file=ann_file,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

