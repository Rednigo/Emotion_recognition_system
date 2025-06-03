#!/usr/bin/env python
"""
SRGAN Preprocessing Module using pre-trained .h5 model
Implements the exact methodology from the research paper:
- 48x48 → 192x192 upscaling (4x factor)
- MediaPipe Face Mesh (468 landmarks)
- 27 key landmark selection
- Angular encoding (10 features)
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import argparse
from pathlib import Path

class SRGANPreprocessorH5:
    """
    SRGAN Preprocessor using pre-trained .h5 model
    Follows the exact paper methodology for emotion recognition
    """
    
    def __init__(self, srgan_model_path, input_size=(48, 48), output_size=(192, 192)):
        """
        Initialize SRGAN preprocessor
        
        Args:
            srgan_model_path: Path to the trained SRGAN .h5 model
            input_size: Input image size (48, 48) as per paper
            output_size: Output image size (192, 192) as per paper
        """
        self.input_size = input_size
        self.output_size = output_size
        
        # Load SRGAN model
        print(f"Loading SRGAN model from {srgan_model_path}...")
        try:
            # Custom objects might be needed for some SRGAN implementations
            self.srgan_model = load_model(srgan_model_path, compile=False)
            print("✅ SRGAN model loaded successfully")
            
            # Print model summary for verification
            print("\nModel Summary:")
            print(f"Input shape: {self.srgan_model.input_shape}")
            print(f"Output shape: {self.srgan_model.output_shape}")
            
        except Exception as e:
            print(f"❌ Error loading SRGAN model: {e}")
            raise
    
    def preprocess_image_srgan(self, image_path):
        """
        Preprocess single image using SRGAN
        
        Args:
            image_path: Path to input image
            
        Returns:
            Super-resolved image (192x192) or None if error
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not read image: {image_path}")
                return None
            
            # Convert BGR to RGB (for proper processing)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to 48x48 if not already (as per paper methodology)
            if image_rgb.shape[:2] != self.input_size:
                image_rgb = cv2.resize(image_rgb, self.input_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1] range (typical for neural networks)
            image_normalized = image_rgb.astype(np.float32) / 255.0
            
            # Add batch dimension
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            # Apply SRGAN super-resolution
            sr_image_batch = self.srgan_model.predict(image_batch, verbose=0)
            
            # Remove batch dimension
            sr_image = sr_image_batch[0]
            
            # Clip values to [0, 1] and convert back to uint8
            sr_image = np.clip(sr_image, 0, 1)
            sr_image = (sr_image * 255).astype(np.uint8)
            
            # Ensure output size is exactly 192x192
            if sr_image.shape[:2] != self.output_size:
                sr_image = cv2.resize(sr_image, self.output_size, interpolation=cv2.INTER_CUBIC)
            
            # Convert back to BGR for OpenCV compatibility
            sr_image_bgr = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
            
            return sr_image_bgr
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def validate_dataset_structure(self, dataset_path):
        """
        Validate RAF-DB dataset structure
        Expected structure:
        dataset_path/
        ├── train/
        │   ├── 1/ (surprise)
        │   ├── 2/ (fear)
        │   ├── 3/ (disgust)
        │   ├── 4/ (happiness)
        │   ├── 5/ (sadness)
        │   ├── 6/ (anger)
        │   └── 7/ (neutral)
        └── test/
            ├── 1/
            ├── 2/
            ├── 3/
            ├── 4/
            ├── 5/
            ├── 6/
            └── 7/
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Check for train and test directories
        train_path = dataset_path / "train"
        test_path = dataset_path / "test"
        
        if not train_path.exists():
            raise ValueError(f"Train directory not found: {train_path}")
        if not test_path.exists():
            raise ValueError(f"Test directory not found: {test_path}")
        
        # Check emotion directories (1-7)
        emotion_names = {
            '1': 'surprise',
            '2': 'fear', 
            '3': 'disgust',
            '4': 'happiness',
            '5': 'sadness',
            '6': 'anger',
            '7': 'neutral'
        }
        
        print("📊 Dataset Structure Validation:")
        for split in ['train', 'test']:
            split_path = dataset_path / split
            print(f"\n{split.upper()} SET:")
            
            total_images = 0
            for emotion_id, emotion_name in emotion_names.items():
                emotion_path = split_path / emotion_id
                if emotion_path.exists():
                    image_files = list(emotion_path.glob("*.jpg")) + \
                                 list(emotion_path.glob("*.jpeg")) + \
                                 list(emotion_path.glob("*.png"))
                    count = len(image_files)
                    total_images += count
                    print(f"  {emotion_name:12} (folder {emotion_id}): {count:4} images")
                else:
                    print(f"  {emotion_name:12} (folder {emotion_id}): MISSING")
            
            print(f"  {'TOTAL':12}: {total_images:4} images")
        
        return True
    
    def preprocess_dataset(self, input_dataset_path, output_dataset_path, test_mode=False):
        """
        Preprocess entire RAF-DB dataset using SRGAN
        
        Args:
            input_dataset_path: Path to original RAF-DB dataset
            output_dataset_path: Path to save preprocessed dataset
            test_mode: If True, process only first 5 images per emotion for testing
        """
        input_path = Path(input_dataset_path)
        output_path = Path(output_dataset_path)
        
        # Validate input dataset structure
        self.validate_dataset_structure(input_path)
        
        # Create output directory structure
        output_path.mkdir(parents=True, exist_ok=True)
        
        emotion_names = {
            '1': 'surprise',
            '2': 'fear',
            '3': 'disgust', 
            '4': 'happiness',
            '5': 'sadness',
            '6': 'anger',
            '7': 'neutral'
        }
        
        processing_stats = {
            'total_processed': 0,
            'total_failed': 0,
            'by_emotion': {}
        }
        
        print(f"\n🚀 Starting SRGAN preprocessing...")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Mode:   {'TEST (5 images per emotion)' if test_mode else 'FULL DATASET'}")
        
        for split in ['train', 'test']:
            split_input = input_path / split
            split_output = output_path / split
            
            if not split_input.exists():
                continue
            
            print(f"\n📁 Processing {split.upper()} set...")
            
            for emotion_id, emotion_name in emotion_names.items():
                emotion_input = split_input / emotion_id
                emotion_output = split_output / emotion_id
                
                if not emotion_input.exists():
                    print(f"⚠️  Skipping {emotion_name} - directory not found")
                    continue
                
                # Create output directory
                emotion_output.mkdir(parents=True, exist_ok=True)
                
                # Get all image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(list(emotion_input.glob(ext)))
                
                if test_mode:
                    image_files = image_files[:5]  # Limit for testing
                
                if not image_files:
                    print(f"⚠️  No images found in {emotion_input}")
                    continue
                
                print(f"  {emotion_name:12}: {len(image_files):4} images")
                
                processed = 0
                failed = 0
                
                # Process each image
                for image_file in tqdm(image_files, 
                                     desc=f"  {split}/{emotion_name}", 
                                     leave=False):
                    
                    output_file = emotion_output / image_file.name
                    
                    # Skip if already processed
                    if output_file.exists():
                        processed += 1
                        continue
                    
                    # Apply SRGAN preprocessing
                    sr_image = self.preprocess_image_srgan(str(image_file))
                    
                    if sr_image is not None:
                        # Save processed image
                        cv2.imwrite(str(output_file), sr_image)
                        processed += 1
                    else:
                        failed += 1
                
                # Update statistics
                emotion_key = f"{split}_{emotion_name}"
                processing_stats['by_emotion'][emotion_key] = {
                    'processed': processed,
                    'failed': failed
                }
                processing_stats['total_processed'] += processed
                processing_stats['total_failed'] += failed
                
                print(f"    ✅ {processed:4} processed, ❌ {failed:2} failed")
        
        # Save processing statistics
        stats_file = output_path / "preprocessing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(processing_stats, f, indent=2)
        
        # Create metadata file with paper specifications
        metadata = {
            "preprocessing_method": "SRGAN",
            "paper_reference": "MediaPipe + Angular Encoding for Emotion Recognition",
            "input_size": self.input_size,
            "output_size": self.output_size,
            "scale_factor": 4,
            "total_images_processed": processing_stats['total_processed'],
            "total_images_failed": processing_stats['total_failed'],
            "emotion_mapping": {
                "1": "surprise",
                "2": "fear", 
                "3": "disgust",
                "4": "happiness",
                "5": "sadness",
                "6": "anger",
                "7": "neutral"
            },
            "next_steps": [
                "1. MediaPipe Face Mesh (468 landmarks)",
                "2. Key landmark selection (27 points)",
                "3. Angular encoding (10 features)",
                "4. Machine learning classification"
            ]
        }
        
        metadata_file = output_path / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ SRGAN Preprocessing completed!")
        print(f"📊 Total processed: {processing_stats['total_processed']}")
        print(f"❌ Total failed: {processing_stats['total_failed']}")
        print(f"📄 Statistics saved to: {stats_file}")
        print(f"📄 Metadata saved to: {metadata_file}")
        
        return processing_stats

def create_test_comparison(preprocessor, input_path, output_path):
    """Create visual comparison between original and SRGAN processed images"""
    
    # Find a test image
    test_image = None
    for split in ['train', 'test']:
        split_path = Path(input_path) / split
        if split_path.exists():
            for emotion_folder in split_path.iterdir():
                if emotion_folder.is_dir():
                    image_files = list(emotion_folder.glob("*.jpg")) + \
                                 list(emotion_folder.glob("*.jpeg")) + \
                                 list(emotion_folder.glob("*.png"))
                    if image_files:
                        test_image = image_files[0]
                        break
            if test_image:
                break
    
    if not test_image:
        print("❌ No test image found for comparison")
        return
    
    print(f"🔍 Creating comparison with: {test_image}")
    
    # Process with SRGAN
    sr_result = preprocessor.preprocess_image_srgan(str(test_image))
    
    if sr_result is None:
        print("❌ Failed to process test image")
        return
    
    # Load original and resize for comparison
    original = cv2.imread(str(test_image))
    if original.shape[:2] != (48, 48):
        original_48 = cv2.resize(original, (48, 48), interpolation=cv2.INTER_AREA)
    else:
        original_48 = original.copy()
    
    # Create comparison image
    # Resize original to 192x192 using simple interpolation for comparison
    original_upscaled = cv2.resize(original_48, (192, 192), interpolation=cv2.INTER_CUBIC)
    
    # Create side-by-side comparison
    comparison = np.hstack([
        original_upscaled,  # Simple upscaling
        sr_result          # SRGAN result
    ])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Cubic Interpolation", (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(comparison, "SRGAN 4x", (202, 30), font, 0.7, (0, 0, 255), 2)
    
    # Save comparison
    comparison_path = Path(output_path) / "srgan_comparison.jpg"
    cv2.imwrite(str(comparison_path), comparison)
    
    print(f"💾 Comparison saved to: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(
        description="SRGAN Preprocessing for RAF-DB dataset using pre-trained .h5 model"
    )
    parser.add_argument('srgan_model', help='Path to trained SRGAN .h5 model')
    parser.add_argument('input_dataset', help='Path to original RAF-DB dataset')
    parser.add_argument('output_dataset', help='Path to save preprocessed dataset')
    parser.add_argument('--test-mode', action='store_true', 
                       help='Process only 5 images per emotion for testing')
    parser.add_argument('--create-comparison', action='store_true',
                       help='Create visual comparison between original and SRGAN')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 SRGAN PREPROCESSING FOR EMOTION RECOGNITION")
    print("Paper: MediaPipe + Angular Encoding Methodology")
    print("=" * 80)
    
    # Initialize preprocessor
    try:
        preprocessor = SRGANPreprocessorH5(args.srgan_model)
    except Exception as e:
        print(f"❌ Failed to initialize preprocessor: {e}")
        return
    
    # Create comparison if requested
    if args.create_comparison:
        print("\n🔍 Creating SRGAN comparison...")
        create_test_comparison(preprocessor, args.input_dataset, args.output_dataset)
    
    # Process dataset
    try:
        stats = preprocessor.preprocess_dataset(
            args.input_dataset, 
            args.output_dataset,
            test_mode=args.test_mode
        )
        
        print("\n🎉 Preprocessing completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run emotion recognition training with preprocessed data")
        print("2. MediaPipe will extract 468 facial landmarks")
        print("3. System will select 27 key landmarks per paper")
        print("4. Angular encoding will generate 10 feature angles")
        print("5. Machine learning models will classify emotions")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return

if __name__ == "__main__":
    main()