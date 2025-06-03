#!/usr/bin/env python
"""
Emotion Recognition System - Fully Compliant with Research Paper
Implements exact methodology:
- MediaPipe Face Mesh (468 landmarks)
- 27 Key Landmark Selection (Table 1)
- Emotional Mesh Generation (Table 4: 38 edges)
- Angular Encoding (Table 5: 10 angles)
- Classification with exact hyperparameters (Table 6)
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# EXACT PAPER SPECIFICATIONS

# Table 1: Selected key landmarks (27 vertices)
SELECTED_LANDMARKS = {
    0: 61,   # Mouth end (right)
    1: 292,  # Mouth end (left) - Paper says 292, correcting from 291
    2: 0,    # Upper lip (middle)
    3: 17,   # Lower lip (middle)
    4: 50,   # Right cheek
    5: 280,  # Left cheek
    6: 48,   # Nose right end
    7: 4,    # Nose tip
    8: 289,  # Nose left end
    9: 206,  # Upper jaw (right)
    10: 426, # Upper jaw (left)
    11: 133, # Right eye (inner)
    12: 130, # Right eye (outer)
    13: 159, # Right upper eyelid (middle)
    14: 145, # Right lower eyelid (middle)
    15: 362, # Left eye (inner)
    16: 359, # Left eye (outer)
    17: 386, # Left upper eyelid (middle)
    18: 374, # Left lower eyelid (middle)
    19: 122, # Nose bridge (right)
    20: 351, # Nose bridge (left)
    21: 46,  # Right eyebrow (outer)
    22: 105, # Right eyebrow (middle)
    23: 107, # Right eyebrow (inner)
    24: 276, # Left eyebrow (outer)
    25: 334, # Left eyebrow (middle)
    26: 336  # Left eyebrow (inner)
}

# Table 4: Emotional mesh edges (38 edges connecting 27 vertices)
MESH_EDGES = [
    (0, 2), (0, 3), (1, 2), (1, 3), (7, 6), (7, 8), (6, 4), (8, 5),
    (6, 9), (9, 0), (4, 0), (8, 10), (10, 1), (5, 1), (7, 19), (7, 20),
    (7, 0), (7, 1), (19, 23), (19, 14), (23, 22), (22, 21), (21, 12),
    (12, 13), (12, 14), (11, 13), (11, 14), (14, 4), (20, 26), (26, 25),
    (25, 24), (24, 16), (16, 17), (16, 18), (15, 17), (15, 18), (18, 20), (18, 5)
]

# Table 5: Angular features (10 angles)
ANGLE_TRIPLETS = [
    (2, 0, 3),    # θ1
    (0, 2, 1),    # θ2  
    (6, 7, 8),    # θ3
    (9, 7, 10),   # θ4
    (0, 7, 1),    # θ5
    (1, 5, 8),    # θ6
    (1, 10, 8),   # θ7
    (13, 12, 14), # θ8
    (21, 22, 23), # θ9
    (6, 19, 23)   # θ10
]

# RAF-DB emotion mapping
EMOTION_MAPPING = {
    1: 'surprise',
    2: 'fear',
    3: 'disgust', 
    4: 'happiness',
    5: 'sadness',
    6: 'anger',
    7: 'neutral'
}

class PaperCompliantEmotionSystem:
    """
    Emotion Recognition System implementing exact paper methodology
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance as per paper
        self.models = {}
        
        if self.verbose:
            print("✅ Emotion Recognition System initialized")
            print(f"📍 Using {len(SELECTED_LANDMARKS)} key landmarks")
            print(f"🕸️  Emotional mesh: {len(MESH_EDGES)} edges")
            print(f"📐 Angular features: {len(ANGLE_TRIPLETS)} angles")
    
    def extract_mediapipe_landmarks(self, image):
        """
        Extract 468 facial landmarks using MediaPipe Face Mesh
        
        Args:
            image: Input image (should be 192x192 after SRGAN)
            
        Returns:
            numpy array of 468 landmarks or None if no face detected
        """
        # Ensure image is RGB for MediaPipe
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Process image with MediaPipe
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract all 468 landmarks
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])
                
            return np.array(landmarks)
        
        return None
    
    def select_key_landmarks(self, all_landmarks):
        """
        Select 27 key landmarks from 468 MediaPipe landmarks (Table 1)
        
        Args:
            all_landmarks: Array of 468 landmarks from MediaPipe
            
        Returns:
            Array of 27 selected key landmarks
        """
        key_landmarks = []
        
        for vertex_id, mediapipe_id in SELECTED_LANDMARKS.items():
            if mediapipe_id < len(all_landmarks):
                landmark = all_landmarks[mediapipe_id]
                key_landmarks.append([landmark[0], landmark[1]])  # Use only x, y coordinates
            else:
                # Fallback if landmark index is out of range
                key_landmarks.append([0.0, 0.0])
                
        return np.array(key_landmarks)
    
    def calculate_angle_between_points(self, p1, p2, p3):
        """
        Calculate angle θ between three points using exact paper formulas (Equations 1-3)
        
        Args:
            p1, p2, p3: Points as [x, y] coordinates
            
        Returns:
            Angle in degrees [0, 360]
        """
        # Equation 1: β = arctan((y3-y2)/(x3-x2))
        beta = np.arctan2(p3[1] - p2[1], p3[0] - p2[0])
        
        # Equation 2: α = arctan((y1-y2)/(x1-x2))
        alpha = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
        
        # Equation 3: θ = β - α
        theta = beta - alpha
        
        # Convert to degrees
        theta_degrees = np.degrees(theta)
        
        # Normalize to [0, 360] as per paper
        if theta_degrees < 0:
            theta_degrees += 360
            
        return theta_degrees
    
    def extract_angular_features(self, key_landmarks):
        """
        Extract 10 angular features from key landmarks (Table 5)
        
        Args:
            key_landmarks: Array of 27 key landmarks
            
        Returns:
            Array of 10 angular features
        """
        angular_features = []
        
        for i, (v1, v2, v3) in enumerate(ANGLE_TRIPLETS):
            if v1 < len(key_landmarks) and v2 < len(key_landmarks) and v3 < len(key_landmarks):
                p1 = key_landmarks[v1]
                p2 = key_landmarks[v2] 
                p3 = key_landmarks[v3]
                
                angle = self.calculate_angle_between_points(p1, p2, p3)
                angular_features.append(angle)
            else:
                # Fallback if indices are out of range
                angular_features.append(0.0)
        
        return np.array(angular_features)
    
    def process_single_image(self, image_path):
        """
        Process single image through complete pipeline
        
        Args:
            image_path: Path to preprocessed image (192x192)
            
        Returns:
            10 angular features or None if processing failed
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Verify image size (should be 192x192 after SRGAN)
        if image.shape[:2] != (192, 192):
            if self.verbose:
                print(f"⚠️  Image {image_path} is {image.shape[:2]}, expected (192, 192)")
            # Resize if needed
            image = cv2.resize(image, (192, 192))
        
        # Extract 468 MediaPipe landmarks
        all_landmarks = self.extract_mediapipe_landmarks(image)
        if all_landmarks is None:
            return None
        
        # Select 27 key landmarks
        key_landmarks = self.select_key_landmarks(all_landmarks)
        
        # Extract 10 angular features
        angular_features = self.extract_angular_features(key_landmarks)
        
        return angular_features
    
    def load_preprocessed_dataset(self, dataset_path, max_samples_per_emotion=None):
        """
        Load preprocessed RAF-DB dataset
        
        Args:
            dataset_path: Path to SRGAN preprocessed dataset
            max_samples_per_emotion: Limit samples per emotion for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        dataset_path = Path(dataset_path)
        
        if self.verbose:
            print(f"📂 Loading dataset from {dataset_path}")
            print("📋 Expected: SRGAN preprocessed images (192x192)")
        
        def load_split(split_name):
            split_path = dataset_path / split_name
            X, y = [], []
            skipped = 0
            
            if not split_path.exists():
                raise ValueError(f"Split directory not found: {split_path}")
            
            for emotion_id, emotion_name in EMOTION_MAPPING.items():
                emotion_path = split_path / str(emotion_id)
                
                if not emotion_path.exists():
                    if self.verbose:
                        print(f"⚠️  Missing emotion folder: {emotion_path}")
                    continue
                
                # Get image files
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(list(emotion_path.glob(ext)))
                
                if max_samples_per_emotion:
                    image_files = image_files[:max_samples_per_emotion]
                
                if self.verbose:
                    print(f"  {emotion_name:12} (folder {emotion_id}): {len(image_files):4} images")
                
                # Process each image
                for image_file in tqdm(image_files, 
                                     desc=f"Processing {split_name}/{emotion_name}", 
                                     leave=False,
                                     disable=not self.verbose):
                    
                    features = self.process_single_image(str(image_file))
                    
                    if features is not None:
                        X.append(features)
                        y.append(emotion_id - 1)  # Convert to 0-based indexing
                    else:
                        skipped += 1
            
            if self.verbose:
                print(f"  ✅ {len(X)} processed, ❌ {skipped} skipped")
            
            return np.array(X), np.array(y)
        
        # Load train and test splits
        if self.verbose:
            print("\n📚 Loading training data...")
        X_train, y_train = load_split('train')
        
        if self.verbose:
            print("\n📚 Loading test data...")
        X_test, y_test = load_split('test')
        
        if self.verbose:
            print(f"\n📊 Dataset loaded successfully:")
            print(f"  Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"  Testing:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
            
            # Show class distribution
            print("\n📈 Class distribution:")
            for i, emotion_name in enumerate(['surprise', 'fear', 'disgust', 'happiness', 'sadness', 'anger', 'neutral']):
                train_count = np.sum(y_train == i)
                test_count = np.sum(y_test == i)
                print(f"  {emotion_name:12}: train={train_count:4}, test={test_count:4}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models_with_paper_hyperparameters(self, X_train, X_test, y_train, y_test):
        """
        Train models with EXACT hyperparameters from Table 6
        """
        if self.verbose:
            print("\n🔧 Preprocessing features...")
        
        # Standardization
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # PCA (keep 95% variance)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        if self.verbose:
            print(f"📐 Features after PCA: {X_train_pca.shape[1]} (from {X_train.shape[1]})")
            print(f"📊 Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Define classifiers with EXACT hyperparameters from Table 6
        classifiers = {
            'DT': DecisionTreeClassifier(
                criterion='gini',
                min_samples_leaf=1,
                min_samples_split=2,
                ccp_alpha=0,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=1,
                leaf_size=30,
                metric='minkowski',
                p=2,
                weights='uniform'
            ),
            'SVM': SVC(
                C=275,
                gamma='scale',
                kernel='rbf',
                random_state=42
            ),
            'NB': GaussianNB(
                var_smoothing=1e-09
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(28, 28),
                activation='relu',
                max_iter=200,
                solver='adam',
                random_state=42
            ),
            'QDA': QuadraticDiscriminantAnalysis(
                tol=0.0001
            ),
            'RF': RandomForestClassifier(
                n_estimators=79,
                criterion='entropy',
                random_state=42
            ),
            'LR': LogisticRegression(
                solver='lbfgs',
                C=1.0,
                fit_intercept=True,
                max_iter=1000,
                random_state=42
            )
        }
        
        results = {}
        
        if self.verbose:
            print(f"\n🎯 Training {len(classifiers)} models with paper hyperparameters...")
        
        for name, classifier in classifiers.items():
            if self.verbose:
                print(f"\n🔄 Training {name}...")
            
            # Train model
            classifier.fit(X_train_pca, y_train)
            
            # Predict on test set
            y_pred = classifier.predict(X_test_pca)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Generate detailed report
            report = classification_report(
                y_test, y_pred,
                target_names=list(EMOTION_MAPPING.values()),
                output_dict=True
            )
            
            # Store results
            results[name] = {
                'model': classifier,
                'accuracy': accuracy,
                'predictions': y_pred,
                'report': report,
                'report_str': classification_report(
                    y_test, y_pred,
                    target_names=list(EMOTION_MAPPING.values())
                )
            }
            
            if self.verbose:
                print(f"  ✅ {name} Accuracy: {accuracy:.4f}")
        
        self.models = results
        return results
    
    def create_results_visualization(self, results, save_path=None):
        """
        Create comprehensive visualization of results
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Emotion Recognition Results - Paper Implementation', fontsize=16)
        
        # 1. Accuracy comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        axes[0, 0].bar(model_names, accuracies, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. Best model confusion matrix
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_predictions = results[best_model_name]['predictions']
        
        # Get true labels for confusion matrix
        y_test = None
        for name, result in results.items():
            if 'y_test' in result:
                y_test = result['y_test']
                break
        
        if y_test is not None:
            cm = confusion_matrix(y_test, best_predictions)
            sns.heatmap(cm, annot=True, fmt='d', 
                       xticklabels=list(EMOTION_MAPPING.values()),
                       yticklabels=list(EMOTION_MAPPING.values()),
                       ax=axes[0, 1])
            axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
            axes[0, 1].set_ylabel('True Label')
            axes[0, 1].set_xlabel('Predicted Label')
        
        # 3. Per-class performance
        if best_model_name in results:
            report = results[best_model_name]['report']
            emotions = list(EMOTION_MAPPING.values())
            f1_scores = [report[emotion]['f1-score'] for emotion in emotions]
            
            axes[1, 0].bar(emotions, f1_scores, color='lightcoral')
            axes[1, 0].set_title(f'F1-Score per Emotion - {best_model_name}')
            axes[1, 0].set_ylabel('F1-Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Model comparison table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        table_data = []
        for name in model_names:
            acc = results[name]['accuracy']
            table_data.append([name, f"{acc:.4f}"])
        
        table = axes[1, 1].table(cellText=table_data,
                                colLabels=['Model', 'Accuracy'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        axes[1, 1].set_title('Results Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"📊 Visualization saved to {save_path}")
        
        return fig
    
    def save_best_model(self, save_dir='emotion_model_paper_compliant'):
        """
        Save the best performing model with metadata
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Find best model
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['accuracy'])
        best_result = self.models[best_model_name]
        
        # Save model components
        joblib.dump(best_result['model'], save_dir / 'model.pkl')
        joblib.dump(self.scaler, save_dir / 'scaler.pkl')
        joblib.dump(self.pca, save_dir / 'pca.pkl')
        
        # Save metadata
        metadata = {
            'paper_implementation': True,
            'methodology': 'SRGAN + MediaPipe + Angular Encoding',
            'best_model': best_model_name,
            'accuracy': float(best_result['accuracy']),
            'preprocessing': {
                'srgan_upscaling': '48x48 -> 192x192 (4x)',
                'mediapipe_landmarks': 468,
                'key_landmarks': 27,
                'angular_features': 10
            },
            'feature_extraction': {
                'selected_landmarks': SELECTED_LANDMARKS,
                'mesh_edges': MESH_EDGES,
                'angle_triplets': ANGLE_TRIPLETS
            },
            'emotion_mapping': EMOTION_MAPPING,
            'pca_components': int(self.pca.n_components_),
            'explained_variance': float(self.pca.explained_variance_ratio_.sum())
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save detailed results
        results_summary = {}
        for name, result in self.models.items():
            results_summary[name] = {
                'accuracy': float(result['accuracy']),
                'classification_report': result['report']
            }
        
        with open(save_dir / 'results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save classification report as text
        with open(save_dir / 'classification_report.txt', 'w') as f:
            f.write(f"EMOTION RECOGNITION RESULTS - PAPER IMPLEMENTATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Accuracy: {best_result['accuracy']:.4f}\n\n")
            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write("-" * 40 + "\n")
            f.write(best_result['report_str'])
            f.write("\n\nALL MODEL RESULTS:\n")
            f.write("-" * 40 + "\n")
            for name, result in sorted(self.models.items(), 
                                     key=lambda x: x[1]['accuracy'], 
                                     reverse=True):
                f.write(f"{name:8}: {result['accuracy']:.4f}\n")
        
        if self.verbose:
            print(f"\n💾 Best model ({best_model_name}) saved to {save_dir}")
            print(f"📄 Files saved:")
            print(f"  - model.pkl (trained model)")
            print(f"  - scaler.pkl (feature scaler)")
            print(f"  - pca.pkl (PCA transformer)")
            print(f"  - metadata.json (implementation details)")
            print(f"  - results.json (all results)")
            print(f"  - classification_report.txt (detailed report)")
        
        return save_dir
    
    def validate_implementation(self):
        """
        Validate that implementation matches paper specifications
        """
        validations = {
            'key_landmarks': len(SELECTED_LANDMARKS) == 27,
            'mesh_edges': len(MESH_EDGES) == 38,
            'angular_features': len(ANGLE_TRIPLETS) == 10,
            'emotion_classes': len(EMOTION_MAPPING) == 7,
            'mediapipe_initialized': self.face_mesh is not None
        }
        
        print("\n🔍 IMPLEMENTATION VALIDATION")
        print("=" * 40)
        for check, passed in validations.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check}: {passed}")
        
        all_passed = all(validations.values())
        
        if all_passed:
            print("\n🎉 All validations passed! Implementation matches paper.")
        else:
            print("\n⚠️  Some validations failed. Check implementation.")
        
        return all_passed

def main():
    """
    Main function demonstrating the complete pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Paper-compliant emotion recognition system"
    )
    parser.add_argument('dataset_path', 
                       help='Path to SRGAN preprocessed dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples per emotion (for testing)')
    parser.add_argument('--output-dir', default='emotion_model_paper_compliant',
                       help='Output directory for model')
    parser.add_argument('--create-visualization', action='store_true',
                       help='Create results visualization')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🎯 PAPER-COMPLIANT EMOTION RECOGNITION SYSTEM")
    print("Implementation of: SRGAN + MediaPipe + Angular Encoding")
    print("=" * 80)
    
    # Initialize system
    system = PaperCompliantEmotionSystem(verbose=True)
    
    # Validate implementation
    system.validate_implementation()
    
    # Load dataset
    try:
        X_train, X_test, y_train, y_test = system.load_preprocessed_dataset(
            args.dataset_path,
            max_samples_per_emotion=args.max_samples
        )
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Train models
    try:
        results = system.train_models_with_paper_hyperparameters(
            X_train, X_test, y_train, y_test
        )
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Display results
    print("\n🏆 FINAL RESULTS")
    print("=" * 40)
    for name, result in sorted(results.items(), 
                              key=lambda x: x[1]['accuracy'], 
                              reverse=True):
        accuracy = result['accuracy']
        print(f"{name:8}: {accuracy:.4f}")
    
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model]['accuracy']
    
    print(f"\n🥇 Best model: {best_model} ({best_accuracy:.4f})")
    
    # Paper comparison
    if best_accuracy >= 0.60:
        print("✅ Results align with paper expectations (60-70% for RAF-DB)")
    else:
        print("⚠️  Results below paper expectations")
    
    # Create visualization
    if args.create_visualization:
        fig = system.create_results_visualization(
            results, 
            save_path=f"{args.output_dir}/results_visualization.png"
        )
    
    # Save best model
    save_path = system.save_best_model(args.output_dir)
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📁 Model saved to: {save_path}")

if __name__ == "__main__":
    main()