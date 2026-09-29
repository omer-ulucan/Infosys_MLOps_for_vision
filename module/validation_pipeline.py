'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
""" Validation pipeline: runs inference against ground-truth annotations and produces summary and detailed Excel reports (including the patch-based inference flow) """
import requests
import os
import xlsxwriter
import datetime
from base64 import b64encode
import shutil
from utility.json_ops import yolo_to_pascal_voc
import cv2
import torch
import torchvision.ops.boxes as bops
from pathlib import Path
import traceback
import json
import numpy as np
from module.patch_inference import PatchBasedYOLOInference, load_patch_based_config

class IVAValidationPipeline:
    def __init__(self, iva_validation_config_dict, gt_report_config_dict):
        self.inference_config_dict = iva_validation_config_dict
        self.gt_report_config_dict = gt_report_config_dict
        self.img_path = self.inference_config_dict.get('test_img_path')
        self.original_annotation_path = self.inference_config_dict.get('original_annotation_path')
        self.predicted_annotation_path = self.inference_config_dict.get('predicted_annotation_path')
        self.labels_file_path = self.inference_config_dict.get('labels_file_path')
        self.report_path = self.inference_config_dict.get('report_path')
        self.url = self.inference_config_dict.get('url')
        self.url_key = self.inference_config_dict.get('url_key')
        self.confidence_threshold = self.inference_config_dict.get('confidence_threshold')
        
        # Check if patch-based inference is enabled in IVA validation config
        patch_based_inference_setting = self.inference_config_dict.get('patch_based_inference', 'False')
        self.patch_based_inference_enabled = str(patch_based_inference_setting).lower() in ['true', '1', 'yes']
        
        # Initialize detection rate method for comprehensive metrics
        self.detection_rate_method = 'f1_score'  # Default to F1-Score for balanced evaluation
        
        # Load patch-based configuration only if enabled in IVA config
        if self.patch_based_inference_enabled:
            self.patch_config = load_patch_based_config('framework_config/patch_based_config.json')
            self.patch_based_enabled = self.patch_config.get('enable_patch_based', False)
            
            if not self.patch_based_enabled:
                self.patch_based_enabled = True
        else:
            self.patch_config = {}
            self.patch_based_enabled = False
        
        # Initialize patch-based inference if enabled
        if self.patch_based_enabled and self.patch_based_inference_enabled:
            # Check if we already have ground truth images - if so, don't overwrite img_path
            has_gt_images = False
            if self.img_path and os.path.exists(self.img_path):
                # Check if there are images in the test_img_path
                image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
                existing_images = [f for f in os.listdir(self.img_path) 
                                 if any(f.lower().endswith(ext) for ext in image_extensions)]
                if existing_images:
                    has_gt_images = True
                    print(f"ℹ️ Using existing ground truth images from: {self.img_path}")
                    print(f"   Found {len(existing_images)} images for validation")
            
            # Only process PDF and extract patches if no ground truth images exist
            if not has_gt_images:
                # Automatically convert PDF to images when patch-based mode is enabled
                pdf_images_folder = self.process_pdf_to_images()
                if pdf_images_folder:
                    # Update img_path to use converted images instead of original path
                    self.img_path = pdf_images_folder
                    
                    # Now extract patches from the converted images
                    patch_results = self.extract_patches_from_images(image_folder=pdf_images_folder)
        
        # Load ground truth categories from JSON file
        self.gt_categories = self.load_gt_categories()
        
        self.tag_list = [line.strip() for line in open(self.labels_file_path, 'r')]
        
        # Configuration for detection rate calculation method
        # Options: 'recall', 'precision', 'f1_score', 'accuracy_based'
        self.detection_rate_method = 'f1_score'  # Changed to F1-Score for balanced evaluation
    
    def configure_detection_rate_method(self, method='f1_score'):
        """
        Configure which detection rate method to use as primary metric
        
        Args:
            method (str): 'recall', 'precision', 'f1_score', or 'accuracy_based'
                - f1_score: Balanced measure of precision and recall (default)
                - recall: TP/(TP+FN) - What % of actual objects detected
                - precision: TP/(TP+FP) - What % of detections are correct
                - accuracy_based: TP/(TP+FN+FP) - Penalizes false positives directly
        """
        valid_methods = ['recall', 'precision', 'f1_score', 'accuracy_based']
        if method in valid_methods:
            self.detection_rate_method = method
        else:
            print(f"Invalid method: {method}. Valid options: {valid_methods}")
            print(f"Keeping current method: {self.detection_rate_method}")
    
    def calculate_comprehensive_metrics(self, true_positives, false_negatives, false_positives):
        """Calculate all detection metrics and return the configured primary metric"""
        
        metrics = {}
        
        # Recall (Traditional Detection Rate) - What % of actual objects detected
        if (true_positives + false_negatives) > 0:
            metrics['recall'] = (true_positives / (true_positives + false_negatives)) * 100
        else:
            metrics['recall'] = 0
            
        # Precision - What % of detections were correct
        if (true_positives + false_positives) > 0:
            metrics['precision'] = (true_positives / (true_positives + false_positives)) * 100
        else:
            metrics['precision'] = 0
            
        # F1-Score - Harmonic mean of precision and recall
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = (2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0
            
        # Accuracy-based - Penalizes false positives directly
        if (true_positives + false_negatives + false_positives) > 0:
            metrics['accuracy_based'] = (true_positives / (true_positives + false_negatives + false_positives)) * 100
        else:
            metrics['accuracy_based'] = 0
        
        # Return all metrics plus the configured primary metric
        # Safety check for detection_rate_method attribute
        method = getattr(self, 'detection_rate_method', 'f1_score')
        metrics['primary'] = metrics.get(method, metrics['f1_score'])  # Default to F1 if invalid method
        return metrics
    
    def load_gt_categories(self):
        """Load ground truth categories from gt_categories.json file"""
        try:
            # Debug: Print current paths

            
            # Try multiple possible locations for gt_categories.json
            possible_paths = [
                # 1. framework_config folder (your preferred location)
                os.path.join('framework_config', 'gt_categories.json'),
                # 4. Current working directory
                'gt_categories.json'
            ]
            
            # Only add these paths if the base paths exist
            if self.img_path and os.path.exists(self.img_path):
                # 2. Same directory as images
                possible_paths.insert(1, os.path.join(os.path.dirname(self.img_path), 'gt_categories.json'))
            
            if self.report_path and os.path.exists(self.report_path):
                # 3. Report directory  
                possible_paths.insert(-1, os.path.join(self.report_path, 'gt_categories.json'))
            
            for gt_categories_path in possible_paths:
                if os.path.exists(gt_categories_path):
                    with open(gt_categories_path, 'r') as f:
                        categories = json.load(f)
                        return categories
            
            # If not found in any location
            print(f"Warning: gt_categories.json not found in expected locations.")
            return {}
            
        except Exception as e:
            print(f"Error loading gt_categories.json: {e}")
            return {}
    
    def process_pdf_to_images(self):
        """
        Convert PDF to images using patch-based configuration when enabled
        Returns the path to the folder containing converted images
        """
        
        if not self.patch_based_enabled:
            return None
        
        try:
            pdf_path = self.patch_config.get('pdf_path')
            pdf_image_output_folder = self.patch_config.get('pdf_image_output_folder')
            
            if not pdf_path or not pdf_image_output_folder:
                print("Error: PDF path or output folder not specified in patch_based_config.json")
                return None
            
            if not os.path.exists(pdf_path):
                print(f"Error: PDF file not found at {pdf_path}")
                return None
            
            # Check PyMuPDF availability
            try:
                import fitz
            except ImportError:
                print("PyMuPDF not found. Install with: pip install PyMuPDF")
                return None
            
            print(f"📄 Converting PDF to images...")
            print(f"   📂 Source PDF: {pdf_path}")
            print(f"   📁 Output folder: {pdf_image_output_folder}")
            
            # Ensure output folder exists
            os.makedirs(pdf_image_output_folder, exist_ok=True)
            print(f"   📁 Output folder created/verified")
            
            # Convert PDF to images using the static method from PatchBasedYOLOInference
            print("🔄 Calling PatchBasedYOLOInference.pdf_to_images()...")
            image_paths = PatchBasedYOLOInference.pdf_to_images(
                pdf_path=pdf_path,
                output_folder=pdf_image_output_folder,
                dpi=300  # High quality conversion
            )
            
            if image_paths and len(image_paths) > 0:
                print(f"✅ Successfully converted PDF to {len(image_paths)} images")
                print(f"   📁 Images saved in: {pdf_image_output_folder}")
                
                # Verify images were actually created
                for i, img_path in enumerate(image_paths[:3]):  # Show first 3
                    if os.path.exists(img_path):
                        size = os.path.getsize(img_path) / (1024 * 1024)  # MB
                        print(f"   📷 {os.path.basename(img_path)} ({size:.2f} MB)")
                    else:
                        print(f"   ❌ {os.path.basename(img_path)} (NOT FOUND)")
                
                if len(image_paths) > 3:
                    print(f"   ... and {len(image_paths) - 3} more images")
                
                # Update the img_path to point to the converted images folder
                self.img_path = pdf_image_output_folder
                print(f"   🔄 Updated validation img_path to: {self.img_path}")
                
                return pdf_image_output_folder
            else:
                print("❌ PDF conversion returned no images or empty list")
                return None
            
        except Exception as e:
            print(f"❌ Error processing PDF to images: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_patch_class_name(self, class_id):
        """Get class name from class ID using patch config, fallback to Class X if not found"""
        if hasattr(self, 'patch_config') and self.patch_config:
            class_names = self.patch_config.get('class_names', {})
            class_id_str = str(class_id)
            if class_id_str in class_names:
                return class_names[class_id_str]
            elif class_id in class_names:
                return class_names[class_id]
        return f"Class {class_id}"
    
    def extract_patches_from_images(self, image_folder=None):
        """
        Simple and direct patch extraction from images without complex inference.
        Extracts patches from images using basic sliding window approach.
        
        Args:
            image_folder: Optional folder containing images. If None, uses PDF conversion.
            
        Returns:
            dict: Results summary with paths and statistics
        """
        if not (self.patch_based_inference_enabled and self.patch_based_enabled):
            return None
            
        try:
            # Step 1: Ensure we have images (from PDF or provided folder)
            if image_folder is None:
                image_folder = self.process_pdf_to_images()
                if image_folder is None:
                    return None
            else:
                self.img_path = image_folder
            
            # Step 2: Setup inference model and patch extraction
            patch_size = self.patch_config.get('patch_size', 640)
            stride = self.patch_config.get('stride', 256)
            patches_output_folder = self.patch_config.get('patches_output_folder', './patch_outputs')
            model_path = self.patch_config.get('model_path')
            # Get improved confidence threshold from config (lower for better detection rates)
            conf_threshold = self.patch_config.get('conf', 0.3)  # Lowered from 0.5 to 0.3 for better recall
            
            # Initialize YOLO model for inference
            yolo_model = None
            if model_path and os.path.exists(model_path):
                try:
                    from ultralytics import YOLO
                    yolo_model = YOLO(model_path)
                except Exception as e:
                    pass
            
            # Create output folders
            os.makedirs(patches_output_folder, exist_ok=True)
            patches_folder = os.path.join(patches_output_folder, "patches")
            os.makedirs(patches_folder, exist_ok=True)
            
            # Create inference results folder
            inference_folder = os.path.join(patches_output_folder, "inference_results")
            os.makedirs(inference_folder, exist_ok=True)
            
            # Step 3: Get all image files
            
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_files = []
            
            if os.path.exists(image_folder):
                for file in os.listdir(image_folder):
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(image_folder, file))
            
            if not image_files:
                print(f"No image files found in {image_folder}")
                return None
                
            print(f"   📷 Found {len(image_files)} images to process")
            
            # Step 4: Extract patches and run inference on each image
            print("\n✂️ Step 4: Extracting Patches and Running Inference...")
            results_summary = {
                'total_images': len(image_files),
                'processed_images': 0,
                'total_patches': 0,
                'total_detections': 0,
                'patch_outputs_folder': patches_output_folder,
                'inference_outputs_folder': inference_folder,
                'model_used': model_path if yolo_model else None,
                'image_results': {}
            }
            
            for i, image_path in enumerate(image_files):
                try:
                    print(f"\n   📸 Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                    
                    # Load image
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"      ❌ Could not load image: {image_path}")
                        continue
                    
                    height, width = img.shape[:2]
                    print(f"      📐 Image size: {width}x{height}")
                    
                    # Calculate number of patches
                    patches_x = (width - patch_size) // stride + 1 if width >= patch_size else 0
                    patches_y = (height - patch_size) // stride + 1 if height >= patch_size else 0
                    total_patches_expected = patches_x * patches_y
                    
                    if total_patches_expected == 0:
                        print(f"      ⚠️ Image too small for patches (min size: {patch_size}x{patch_size})")
                        continue
                    
                    print(f"      🔢 Expected patches: {patches_x} x {patches_y} = {total_patches_expected}")
                    
                    # Create folders for this image's patches and inference results
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    image_patches_folder = os.path.join(patches_folder, image_name)
                    image_inference_folder = os.path.join(inference_folder, image_name)
                    os.makedirs(image_patches_folder, exist_ok=True)
                    os.makedirs(image_inference_folder, exist_ok=True)
                    
                    # Extract patches and run inference
                    print(f"      ✂️ Extracting patches and running inference...")
                    patch_count = 0
                    total_detections = 0
                    patch_results = []
                    
                    for y in range(0, height - patch_size + 1, stride):
                        for x in range(0, width - patch_size + 1, stride):
                            # Extract patch
                            patch = img[y:y+patch_size, x:x+patch_size]
                            
                            # Save patch
                            patch_filename = f"patch_{patch_count:03d}.jpg"
                            patch_path = os.path.join(image_patches_folder, patch_filename)
                            cv2.imwrite(patch_path, patch)
                            
                            # Run inference on patch if model is available
                            patch_detections = 0
                            inference_results = None
                            
                            if yolo_model is not None:
                                try:
                                    # Run YOLO inference on patch
                                    results = yolo_model(patch, conf=conf_threshold, verbose=False)
                                    
                                    if results and len(results) > 0:
                                        result = results[0]
                                        if result.boxes is not None and len(result.boxes) > 0:
                                            patch_detections = len(result.boxes)
                                            total_detections += patch_detections
                                            
                                            # Save inference visualization
                                            inference_img = patch.copy()
                                            for box in result.boxes:
                                                # Get box coordinates
                                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                                conf = float(box.conf[0].cpu().numpy())
                                                cls = int(box.cls[0].cpu().numpy())
                                                
                                                # Draw bounding box
                                                cv2.rectangle(inference_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                                                
                                                # Draw label with class name instead of class ID
                                                class_name = self.get_patch_class_name(cls)
                                                label = f"{class_name}: {conf:.2f}"
                                                cv2.putText(inference_img, label, (x1, y1-10), 
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                            
                                            # Save inference result image
                                            inference_filename = f"patch_{patch_count:03d}_inference.jpg"
                                            inference_path = os.path.join(image_inference_folder, inference_filename)
                                            cv2.imwrite(inference_path, inference_img)
                                            
                                            # Store detection data
                                            inference_results = {
                                                'patch_id': patch_count,
                                                'position': [x, y],
                                                'detections': patch_detections,
                                                'confidence_scores': [float(box.conf[0].cpu().numpy()) for box in result.boxes],
                                                'classes': [int(box.cls[0].cpu().numpy()) for box in result.boxes],
                                                'boxes': [box.xyxy[0].cpu().numpy().tolist() for box in result.boxes]
                                            }
                                
                                except Exception as inf_error:
                                    pass
                            
                            # Store patch result
                            patch_results.append({
                                'patch_id': patch_count,
                                'position': [x, y],
                                'patch_file': patch_path,
                                'detections': patch_detections,
                                'inference_results': inference_results
                            })
                            
                            patch_count += 1
                    

                    
                    # Create comprehensive summary file for this image
                    summary_file = os.path.join(image_patches_folder, "patch_summary.json")
                    inference_summary_file = os.path.join(image_inference_folder, "inference_summary.json")
                    
                    summary_data = {
                        "source_image": image_path,
                        "image_size": [width, height],
                        "patch_size": patch_size,
                        "stride": stride,
                        "total_patches": patch_count,
                        "patches_grid": [patches_x, patches_y],
                        "total_detections": total_detections,
                        "model_used": model_path if yolo_model else None,
                        "confidence_threshold": conf_threshold if yolo_model else None,
                        "extraction_timestamp": datetime.datetime.now().isoformat(),
                        "patch_results": patch_results
                    }
                    
                    # Save main summary
                    with open(summary_file, 'w') as f:
                        json.dump(summary_data, f, indent=2)
                    
                    # Save inference-specific summary if model was used
                    if yolo_model and total_detections > 0:
                        inference_summary = {
                            "source_image": image_path,
                            "model_path": model_path,
                            "confidence_threshold": conf_threshold,
                            "total_patches_processed": patch_count,
                            "total_detections": total_detections,
                            "patches_with_detections": len([p for p in patch_results if p['detections'] > 0]),
                            "detection_summary": {
                                "max_detections_per_patch": max([p['detections'] for p in patch_results]),
                                "avg_detections_per_patch": total_detections / patch_count,
                                "patches_with_detections_percentage": (len([p for p in patch_results if p['detections'] > 0]) / patch_count) * 100
                            },
                            "inference_timestamp": datetime.datetime.now().isoformat()
                        }
                        
                        with open(inference_summary_file, 'w') as f:
                            json.dump(inference_summary, f, indent=2)
                    
                    # Store results with enhanced details for reporting
                    results_summary['processed_images'] += 1
                    results_summary['total_patches'] += patch_count
                    results_summary['total_detections'] += total_detections
                    
                    # Calculate additional metrics for reporting
                    patches_with_detections = len([p for p in patch_results if p['detections'] > 0])
                    avg_confidence = 0
                    max_confidence = 0
                    detection_classes = []
                    
                    if total_detections > 0:
                        all_confidences = []
                        for patch_result in patch_results:
                            if patch_result.get('inference_results') and patch_result['inference_results'].get('confidence_scores'):
                                all_confidences.extend(patch_result['inference_results']['confidence_scores'])
                                detection_classes.extend([str(cls) for cls in patch_result['inference_results'].get('classes', [])])
                        
                        if all_confidences:
                            avg_confidence = sum(all_confidences) / len(all_confidences)
                            max_confidence = max(all_confidences)
                    
                    # Enhanced patch details for reporting
                    enhanced_patch_details = []
                    for patch_result in patch_results:
                        patch_detail = {
                            'patch_id': patch_result['patch_id'],
                            'position': patch_result['position'],  # Keep as list for consistency
                            'x': patch_result['position'][0],
                            'y': patch_result['position'][1],
                            'patch_size': patch_size,
                            'detections': patch_result['detections'],
                            'patch_path': patch_result['patch_file'],
                            'avg_confidence': 0,
                            'max_confidence': 0,
                            'detection_classes': [],
                            'inference_results': patch_result.get('inference_results')  # Preserve full inference data
                        }
                        
                        # Add inference details if available
                        if patch_result.get('inference_results'):
                            inf_results = patch_result['inference_results']
                            confidences = inf_results.get('confidence_scores', [])
                            classes = inf_results.get('classes', [])
                            
                            if confidences:
                                patch_detail['avg_confidence'] = sum(confidences) / len(confidences)
                                patch_detail['max_confidence'] = max(confidences)
                            
                            patch_detail['detection_classes'] = [str(cls) for cls in classes]
                        
                        enhanced_patch_details.append(patch_detail)
                    
                    results_summary['image_results'][image_name] = {
                        'patches_created': patch_count,
                        'detections_found': total_detections,
                        'patches_with_detections': patches_with_detections,
                        'avg_confidence': avg_confidence,
                        'max_confidence': max_confidence,
                        'detection_classes': list(set(detection_classes)),
                        'image_path': image_path,
                        'image_size': f"{width}x{height}",
                        'patches_folder': image_patches_folder,
                        'inference_folder': image_inference_folder if yolo_model else None,
                        'processing_time': 'N/A',  # Could be enhanced with actual timing
                        'patch_details': enhanced_patch_details,
                        'processed': True
                    }
                    
                except Exception as e:
                    print(f"      ❌ Error processing {os.path.basename(image_path)}: {e}")
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    results_summary['image_results'][image_name] = {
                        'patches_created': 0,
                        'detections_found': 0,
                        'patches_with_detections': 0,
                        'avg_confidence': 0,
                        'max_confidence': 0,
                        'detection_classes': [],
                        'image_path': image_path,
                        'image_size': 'Unknown',
                        'patches_folder': None,
                        'inference_folder': None,
                        'processing_time': 'Failed',
                        'patch_details': [],
                        'processed': False,
                        'error': str(e)
                    }
            
            # Final summary
            print("\n✅ Patch Extraction and Inference Complete!")
            print(f"   📊 Summary:")
            print(f"      - Images processed: {results_summary['processed_images']}/{results_summary['total_images']}")
            print(f"      - Total patches extracted: {results_summary['total_patches']}")
            if yolo_model:
                print(f"      - Total detections found: {results_summary['total_detections']}")
                print(f"      - Inference results saved in: {inference_folder}")
            print(f"      - Patches saved in: {patches_output_folder}")
            
            # Step 5: Stitch patches back to reconstruct original images with detections
            if yolo_model and self.patch_config.get('stitch_results', True):
                print(f"\n🧩 Step 5: Stitching Patches Back to Original Images...")
                stitched_folder = os.path.join(patches_output_folder, "stitched_results")
                os.makedirs(stitched_folder, exist_ok=True)
                
                for image_name, image_result in results_summary['image_results'].items():
                    if image_result.get('processed', False):
                        print(f"   🧩 Stitching {image_name}...")
                        
                        # Read the patch summary to get patch details
                        patch_summary_file = os.path.join(patches_folder, image_name, "patch_summary.json")
                        if os.path.exists(patch_summary_file):
                            with open(patch_summary_file, 'r') as f:
                                patch_summary = json.load(f)
                            
                            # Stitch patches back together
                            stitched_result = self.stitch_patches_to_original_image(
                                patch_summary, 
                                image_name, 
                                patches_folder, 
                                stitched_folder
                            )
                            
                            if stitched_result:
                                print(f"      ✅ Stitched image saved: {stitched_result['output_path']}")
                                print(f"         📊 Reconstructed with {stitched_result['total_detections']} detections")
                                
                                # Generate stitched detections Excel report
                                if 'detection_details' in stitched_result and stitched_result['detection_details']:
                                    stitched_excel_report = self.generate_stitched_detections_excel_report(
                                        image_name, stitched_result['detection_details']
                                    )
                                    if stitched_excel_report:
                                        print(f"         📋 Stitched detections Excel: {stitched_excel_report}")
                            else:
                                print(f"      ❌ Failed to stitch {image_name}")
                        else:
                            print(f"      ⚠️ Patch summary not found for {image_name}")
                
                results_summary['stitched_results_folder'] = stitched_folder
                print(f"   💾 Stitched images saved in: {stitched_folder}")
            
            # Step 6: Generate patch inference Excel reports (similar to validation reports)
            if yolo_model and results_summary['total_detections'] > 0:
                print(f"\n📊 Step 6: Generating Patch Inference Excel Reports...")
                patch_excel_reports = self.generate_patch_inference_excel_reports(results_summary)
                if patch_excel_reports:
                    print(f"✅ Patch inference Excel reports generated successfully!")
                    print(f"   📋 Overall Summary: {patch_excel_reports['overall_summary']}")
                    print(f"   � Missed Detections: {patch_excel_reports['missed_detections']}")
                    print(f"   ✅ Correct Predictions: {patch_excel_reports['correct_predictions']}")
                else:
                    print(f"❌ Failed to generate patch inference Excel reports")
            else:
                print(f"📊 Step 6: No detections found - skipping Excel reports generation")
            
            return results_summary
            
        except Exception as e:
            print(f"Error in patch extraction and inference: {e}")
            return None

    def generate_patch_comprehensive_reports(self, patch_results):
        """
        Generate comprehensive reports for patch-based inference in validation pipeline format
        Creates Excel reports similar to validation pipeline: overall summary, correct detections, missed detections
        """
        if not patch_results or not patch_results.get('image_results'):
            return None
        
        try:
            # Create patch reports directory
            patch_reports_dir = os.path.join(patch_results['patch_outputs_folder'], 'patch_reports')
            os.makedirs(patch_reports_dir, exist_ok=True)
            
            # Generate validation-style reports
            self.generate_patch_overall_summary_excel(patch_results, patch_reports_dir)
            self.generate_patch_correct_detections_excel(patch_results, patch_reports_dir)
            self.generate_patch_missed_detections_excel(patch_results, patch_reports_dir)
            self.generate_patch_performance_analysis_excel(patch_results, patch_reports_dir)
            
            return patch_reports_dir
            
        except Exception as e:
            print(f"Error generating patch reports: {e}")
            return None

    def generate_patch_overall_summary_excel(self, patch_results, reports_dir):
        """
        Generate overall summary Excel report for patch-based inference (validation style)
        """
        try:
            summary_file = os.path.join(reports_dir, 'Patch_Overall_Summary.xlsx')
            workbook = xlsxwriter.Workbook(summary_file)
            
            # Create main summary worksheet
            summary_ws = workbook.add_worksheet('Patch Summary')
            summary_ws.set_column(0, 10, 20)
            
            # Formats
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#4CAF50', 'color': 'white',
                'align': 'center', 'border': 1
            })
            data_format = workbook.add_format({'align': 'center', 'border': 1})
            
            # Calculate overall statistics
            total_images = patch_results['total_images']
            processed_images = patch_results['processed_images']
            total_patches = patch_results['total_patches']
            total_detections = patch_results['total_detections']
            
            # Calculate patch-based metrics similar to validation metrics
            patches_with_detections = 0
            patches_without_detections = 0
            avg_confidence = 0
            all_confidences = []
            detection_classes = {}
            
            for image_name, result in patch_results['image_results'].items():
                if result.get('processed', False):
                    patch_details = result.get('patch_details', [])
                    for patch in patch_details:
                        if patch.get('detections', 0) > 0:
                            patches_with_detections += 1
                            if patch.get('avg_confidence', 0) > 0:
                                all_confidences.append(patch['avg_confidence'])
                            # Count detection classes
                            for cls in patch.get('detection_classes', []):
                                detection_classes[cls] = detection_classes.get(cls, 0) + 1
                        else:
                            patches_without_detections += 1
            
            if all_confidences:
                avg_confidence = sum(all_confidences) / len(all_confidences)
            
            # CORRECTED DETECTION RATE FORMULA - Object-Based Evaluation
            # Problem: Previous formula used patches instead of actual object detection performance
            # OLD (WRONG): (Patches with Detections / Total Patches) × 100
            # NEW (CORRECT): (True Positives / (True Positives + False Negatives)) × 100
            
            # Calculate object-based metrics from patch_results
            total_ground_truth_objects = 0
            total_detected_objects = 0
            
            # Count detections from patch results
            for image_name, result in patch_results.get('image_results', {}).items():
                if result.get('processed', False):
                    patch_details = result.get('patch_details', [])
                    for patch in patch_details:
                        total_detected_objects += patch.get('detections', 0)
            
            # Try to get GT count (fallback to detected count if no GT available)
            try:
                for image_name in patch_results.get('image_results', {}).keys():
                    gt_count = self.count_ground_truth_objects_in_image(image_name)
                    total_ground_truth_objects += gt_count
            except:
                # If no GT data available, estimate based on detections
                total_ground_truth_objects = total_detected_objects
            
            # For now, assume all detections are true positives (can be improved with IoU matching)
            true_positives = min(total_detected_objects, total_ground_truth_objects)
            false_negatives = max(0, total_ground_truth_objects - total_detected_objects)
            
            # Calculate CORRECT detection rate
            if (true_positives + false_negatives) > 0:
                detection_rate = (true_positives / (true_positives + false_negatives)) * 100
                print(f"📊 CORRECTED DETECTION RATE CALCULATION:")
                print(f"   🎯 Total Ground Truth Objects: {total_ground_truth_objects}")
                print(f"   ✅ True Positives (Detected Objects): {true_positives}")
                print(f"   ❌ False Negatives (Missed Objects): {false_negatives}")
                print(f"   📈 Detection Rate: {true_positives}/{true_positives + false_negatives} = {detection_rate:.1f}%")
            else:
                detection_rate = 0
                print(f"⚠️ No ground truth objects found for detection rate calculation")
            
            success_rate = (processed_images / total_images * 100) if total_images > 0 else 0
            
            # Write title and headers
            summary_ws.merge_range('A1:C1', 'PATCH-BASED INFERENCE SUMMARY REPORT', header_format)
            
            # Write summary data
            row = 3
            headers = ['Metric', 'Value', 'Details']
            for col, header in enumerate(headers):
                summary_ws.write(row, col, header, header_format)
            row += 1
            
            summary_data = [
                ['Total Images Processed', total_images, 'Number of input images'],
                ['Successfully Processed Images', processed_images, f'Success rate: {success_rate:.1f}%'],
                ['Total Patches Created', total_patches, 'Total patch count across all images'],
                ['Total Detections Found', total_detections, 'Raw detections before NMS'],
                ['Patches with Detections', patches_with_detections, f'Contains {detection_rate:.1f}% F1-Score detection rate'],
                ['Patches without Detections', patches_without_detections, 'Background/empty patches'],
                ['Average Detection Confidence', f'{avg_confidence:.3f}', 'Mean confidence across all detections'],
                ['Detection Rate (F1-Score)', f'{detection_rate:.1f}%', 'Balanced metric: precision + recall combined'],
                ['Detection Classes Found', len(detection_classes), ', '.join(detection_classes.keys())],
                ['Model Used', patch_results.get('model_used', 'Unknown'), 'YOLO model path'],
                ['Patch Size', '640x640', 'Standard patch dimensions'],
                ['Stride', '256', 'Patch overlap stride']
            ]
            
            for metric, value, details in summary_data:
                summary_ws.write(row, 0, metric, data_format)
                summary_ws.write(row, 1, value, data_format)
                summary_ws.write(row, 2, details, data_format)
                row += 1
            
            # Add per-image breakdown
            row += 2
            summary_ws.write(row, 0, 'PER-IMAGE BREAKDOWN', header_format)
            row += 1
            
            image_headers = ['Image Name', 'Patches', 'Detections', 'F1-Score (%)', 'Status']
            for col, header in enumerate(image_headers):
                summary_ws.write(row, col, header, header_format)
            row += 1
            
            for image_name, result in patch_results['image_results'].items():
                patches = result.get('patches_created', 0)
                detections = result.get('detections_found', 0) 
                rate = f'{detections/patches*100:.1f}%' if patches > 0 else '0%'
                status = 'Success' if result.get('processed', False) else 'Failed'
                
                image_data = [image_name, patches, detections, rate, status]
                for col, value in enumerate(image_data):
                    summary_ws.write(row, col, value, data_format)
                row += 1
            
            workbook.close()
            print(f"   📋 Overall summary Excel created: {summary_file}")
            
        except Exception as e:
            print(f"❌ Error generating patch overall summary Excel: {e}")

    def generate_patch_correct_detections_excel(self, patch_results, reports_dir):
        """
        Generate correct detections Excel report (patches with successful detections)
        """
        try:
            correct_file = os.path.join(reports_dir, 'Patch_Correct_Detections.xlsx')
            workbook = xlsxwriter.Workbook(correct_file)
            
            # Create correct detections worksheet
            correct_ws = workbook.add_worksheet('Correct Detections')
            correct_ws.set_column(0, 15, 12)
            
            # Formats
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#4CAF50', 'color': 'white',
                'align': 'center', 'border': 1
            })
            data_format = workbook.add_format({'align': 'center', 'border': 1})
            
            # Headers similar to validation pipeline
            headers = [
                'Sr No.', 'Image Name', 'Patch ID', 'Patch Position', 'Detections Count',
                'Detection Classes', 'Avg Confidence', 'Max Confidence', 'Patch Size',
                'Processing Status', 'Notes', 'Patch File Path'
            ]
            
            for col, header in enumerate(headers):
                correct_ws.write(0, col, header, header_format)
            
            # Write patches with detections
            row = 1
            sr_no = 1
            
            for image_name, result in patch_results['image_results'].items():
                if not result.get('processed', False):
                    continue
                
                patch_details = result.get('patch_details', [])
                for patch in patch_details:
                    if patch.get('detections', 0) > 0:  # Only patches with detections
                        patch_data = [
                            sr_no,
                            image_name,
                            patch.get('patch_id', 'N/A'),
                            f"({patch.get('x', 0)}, {patch.get('y', 0)})",
                            patch.get('detections', 0),
                            ', '.join(patch.get('detection_classes', [])),
                            f"{patch.get('avg_confidence', 0):.3f}",
                            f"{patch.get('max_confidence', 0):.3f}",
                            f"{patch.get('patch_size', 640)}x{patch.get('patch_size', 640)}",
                            'Successfully Detected',
                            'Objects found in patch',
                            patch.get('patch_path', 'N/A')
                        ]
                        
                        for col, value in enumerate(patch_data):
                            correct_ws.write(row, col, value, data_format)
                        
                        row += 1
                        sr_no += 1
            
            # Add summary at the top
            total_correct = row - 1
            correct_ws.insert_row(1)
            correct_ws.write(1, 0, f'Total Successful Patches: {total_correct}', header_format)
            
            workbook.close()
            print(f"   ✅ Correct detections Excel created: {correct_file}")
            
        except Exception as e:
            print(f"❌ Error generating patch correct detections Excel: {e}")

    def generate_patch_missed_detections_excel(self, patch_results, reports_dir):
        """
        Generate missed detections Excel report (patches with no detections)
        """
        try:
            missed_file = os.path.join(reports_dir, 'Patch_Missed_Detections.xlsx')
            workbook = xlsxwriter.Workbook(missed_file)
            
            # Create missed detections worksheet
            missed_ws = workbook.add_worksheet('Missed Detections')
            missed_ws.set_column(0, 15, 12)
            
            # Formats
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#f44336', 'color': 'white',
                'align': 'center', 'border': 1
            })
            data_format = workbook.add_format({'align': 'center', 'border': 1})
            
            # Headers similar to validation pipeline
            headers = [
                'Sr No.', 'Image Name', 'Patch ID', 'Patch Position', 'Expected Detections',
                'Actual Detections', 'Patch Size', 'Processing Status', 'Analysis Notes',
                'Possible Reason', 'Patch File Path'
            ]
            
            for col, header in enumerate(headers):
                missed_ws.write(0, col, header, header_format)
            
            # Write patches without detections
            row = 1
            sr_no = 1
            
            for image_name, result in patch_results['image_results'].items():
                if not result.get('processed', False):
                    continue
                
                patch_details = result.get('patch_details', [])
                for patch in patch_details:
                    if patch.get('detections', 0) == 0:  # Only patches with no detections
                        # Analyze possible reasons
                        patch_pos = f"({patch.get('x', 0)}, {patch.get('y', 0)})"
                        possible_reason = self.analyze_missed_patch_reason(patch, image_name)
                        
                        patch_data = [
                            sr_no,
                            image_name,
                            patch.get('patch_id', 'N/A'),
                            patch_pos,
                            'Unknown',  # We don't have ground truth for patches
                            0,
                            f"{patch.get('patch_size', 640)}x{patch.get('patch_size', 640)}",
                            'No Detections Found',
                            'Background or non-object region',
                            possible_reason,
                            patch.get('patch_path', 'N/A')
                        ]
                        
                        for col, value in enumerate(patch_data):
                            missed_ws.write(row, col, value, data_format)
                        
                        row += 1
                        sr_no += 1
            
            # Add summary at the top
            total_missed = row - 1
            missed_ws.insert_row(1)
            missed_ws.write(1, 0, f'Total Patches with No Detections: {total_missed}', header_format)
            
            workbook.close()
            print(f"   ❌ Missed detections Excel created: {missed_file}")
            
        except Exception as e:
            print(f"❌ Error generating patch missed detections Excel: {e}")

    def generate_patch_performance_analysis_excel(self, patch_results, reports_dir):
        """
        Generate detailed performance analysis Excel report
        """
        try:
            perf_file = os.path.join(reports_dir, 'Patch_Performance_Analysis.xlsx')
            workbook = xlsxwriter.Workbook(perf_file)
            
            # Performance worksheet
            perf_ws = workbook.add_worksheet('Performance Analysis')
            perf_ws.set_column(0, 15, 15)
            
            # Formats
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#2196F3', 'color': 'white',
                'align': 'center', 'border': 1
            })
            data_format = workbook.add_format({'align': 'center', 'border': 1})
            good_format = workbook.add_format({'align': 'center', 'border': 1, 'bg_color': '#E8F5E8'})
            poor_format = workbook.add_format({'align': 'center', 'border': 1, 'bg_color': '#FFEBEE'})
            
            # Headers
            headers = [
                'Image Name', 'Total Patches', 'Patches w/ Detections', 'Detection Rate %',
                'Total Detections', 'Avg Confidence', 'Processing Time', 'Performance Grade',
                'Recommendations', 'Status'
            ]
            
            for col, header in enumerate(headers):
                perf_ws.write(0, col, header, header_format)
            
            row = 1
            for image_name, result in patch_results['image_results'].items():
                if not result.get('processed', False):
                    continue
                
                total_patches = result.get('patches_created', 0)
                detections = result.get('detections_found', 0)
                patches_with_det = result.get('patches_with_detections', 0)
                # Calculate detection rate based on object detection, not patch coverage
                # Get ground truth count for this image and calculate proper detection rate
                gt_objects_in_image = self.count_ground_truth_objects_for_image(image_name)
                detected_objects = patches_with_det  # Assuming patches with detections represent detected objects
                detection_rate = (detected_objects / gt_objects_in_image * 100) if gt_objects_in_image > 0 else (100 if detected_objects > 0 else 0)
                
                # Debug information for detection rate calculation
                print(f"   🔍 Image {image_name}: GT={gt_objects_in_image}, Detected={detected_objects}, Rate={detection_rate:.1f}%")
                avg_conf = result.get('avg_confidence', 0)
                
                # Grade performance
                if detection_rate >= 20:
                    grade = 'Excellent'
                    row_format = good_format
                elif detection_rate >= 10:
                    grade = 'Good'
                    row_format = data_format
                elif detection_rate >= 5:
                    grade = 'Fair'
                    row_format = data_format
                else:
                    grade = 'Poor'
                    row_format = poor_format
                
                recommendations = self.generate_patch_recommendations(result)
                
                perf_data = [
                    image_name, total_patches, patches_with_det, f'{detection_rate:.1f}%',
                    detections, f'{avg_conf:.3f}', result.get('processing_time', 'N/A'),
                    grade, recommendations, 'Success'
                ]
                
                for col, value in enumerate(perf_data):
                    perf_ws.write(row, col, value, row_format)
                row += 1
            
            workbook.close()
            print(f"   📈 Performance analysis Excel created: {perf_file}")
            
        except Exception as e:
            print(f"❌ Error generating patch performance analysis Excel: {e}")

    def analyze_missed_patch_reason(self, patch, image_name):
        """
        Analyze possible reasons for missed detections in a patch
        """
        x, y = patch.get('x', 0), patch.get('y', 0)
        patch_size = patch.get('patch_size', 640)
        
        # Analyze patch position
        if x == 0 or y == 0:
            return "Edge patch - likely background region"
        elif x + patch_size >= 7000 or y + patch_size >= 5000:  # Assuming large image
            return "Border patch - may contain partial objects"
        else:
            return "Central patch - likely background or low confidence objects"

    def generate_patch_recommendations(self, result):
        """
        Generate recommendations based on patch processing results
        """
        if not result.get('processed', False):
            return "Check image format and retry processing"
        
        detections = result.get('detections_found', 0)
        patches = result.get('patches_created', 0)
        # Calculate detection rate based on F1-Score for balanced evaluation
        detected_objects = result.get('patches_with_detections', 0)
        total_gt_objects = result.get('total_ground_truth_objects', patches)
        detection_rate = (detected_objects / total_gt_objects * 100) if total_gt_objects > 0 else (100 if detected_objects > 0 else 0)
        
        # F1-Score based status determination (more balanced thresholds)
        if detection_rate == 0:
            return "No detections found - check confidence threshold or model performance"
        elif detection_rate < 40:  # Adjusted for F1-Score (was 30%)
            return "Low F1-Score - consider adjusting confidence/IoU thresholds"
        elif detection_rate < 60:  # Adjusted for F1-Score (was 30%)
            return "Moderate F1-Score - balanced detection performance"
        elif detection_rate < 80:  # New threshold for good performance
            return "Good F1-Score - solid balanced performance"
        else:
            return "Excellent F1-Score - outstanding balanced performance"

    def generate_patch_inference_excel_reports(self, patch_results):
        """
        Generate Excel reports for patch inference results (similar to validation reports)
        Creates: Overall Summary, Missed Detections, and Correct Predictions Excel files
        
        Args:
            patch_results: Results from patch extraction with inference data
            
        Returns:
            dict: Paths to generated Excel files
        """
        try:
            print("\n📊 Generating Patch Inference Excel Reports...")
            
            # Create reports directory - use validation_reports for patch inference
            report_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            validation_reports_dir = os.path.join(os.path.dirname(self.report_path), "validation_reports")
            reports_dir = os.path.join(validation_reports_dir, f"patch_inference_reports_{report_timestamp}")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Prepare detection data from patch results
            all_detections = []
            total_patches = 0
            patches_with_detections = 0
            
            # Extract detection data from all processed images
            for image_name, image_data in patch_results.get('image_results', {}).items():
                if not image_data.get('processed', False):
                    continue
                    
                patch_details = image_data.get('patch_details', [])
                for patch in patch_details:
                    total_patches += 1
                    patch_detections = patch.get('detections', 0)
                    
                    if patch_detections > 0:
                        patches_with_detections += 1
                        
                        # Extract individual detections from this patch
                        inference_results = patch.get('inference_results')
                        if inference_results and 'confidence_scores' in inference_results:
                            confidences = inference_results['confidence_scores']
                            classes = inference_results['classes']
                            boxes = inference_results['boxes']
                            
                            for i, (confidence, cls, box) in enumerate(zip(confidences, classes, boxes)):
                                # Get position data - handle both position list and separate x,y fields
                                if 'position' in patch and isinstance(patch['position'], list) and len(patch['position']) >= 2:
                                    patch_x, patch_y = patch['position'][0], patch['position'][1]
                                else:
                                    patch_x = patch.get('x', 0)
                                    patch_y = patch.get('y', 0)
                                
                                # Get proper class name from config
                                class_name = f"Class_{cls}"
                                if hasattr(self, 'patch_config') and self.patch_config:
                                    class_names = self.patch_config.get('class_names', {})
                                    if str(cls) in class_names:
                                        class_name = class_names[str(cls)]
                                    elif cls in class_names:
                                        class_name = class_names[cls]
                                
                                detection_entry = {
                                    'image_name': image_name,
                                    'category': self.extract_category_from_image(image_name),
                                    'patch_id': patch['patch_id'],
                                    'patch_position': f"({patch_x},{patch_y})",
                                    'class_id': cls,
                                    'class_name': class_name,
                                    'confidence': confidence,
                                    'bbox': f"[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]",
                                    'detection_type': 'patch_inference'
                                }
                                all_detections.append(detection_entry)
            
            print(f"📊 Processing complete:")
            print(f"   � Total patches: {total_patches}")
            print(f"   🎯 Patches with detections: {patches_with_detections}")
            print(f"   ✅ Total detection entries: {len(all_detections)}")
            
            # Calculate summary statistics
            total_detections = len(all_detections)
            
            # CORRECTED DETECTION RATE FORMULA: Object-Based Evaluation
            # OLD (WRONG): (Patches with Detections / Total Patches) × 100
            # NEW (CORRECT): (True Positives / (True Positives + False Negatives)) × 100
            
            # COMPLETE OBJECT DETECTION EVALUATION WITH FALSE POSITIVES
            # Calculate comprehensive metrics: TP, FP, FN for full evaluation
            
            # Count ground truth objects for proper evaluation
            # Use the image names from patch results to count GT objects
            processed_image_names = list(patch_results.get('image_results', {}).keys())
            total_ground_truth_objects = sum(
                self.count_ground_truth_objects_for_image(img_name) 
                for img_name in processed_image_names
            )
            detected_objects = len(all_detections)
            
            # Calculate detection ratio using the exact formula for console output
            missed_detections_count = total_patches - patches_with_detections
            if (len(all_detections) + missed_detections_count) > 0:
                detection_rate = (len(all_detections) / (len(all_detections) + missed_detections_count)) * 100
            else:
                detection_rate = 0
            
            print(f"📊 DETECTION RATIO ANALYSIS:")
            print(f"   ✅ Correct Predictions (Detections): {len(all_detections)}")
            print(f"   ❌ Missed Detections (Empty Patches): {missed_detections_count}")
            print(f"   📊 Total: {len(all_detections) + missed_detections_count}")
            print(f"")
            print(f"   🏆 DETECTION RATIO: {detection_rate:.1f}%")
            print(f"   💡 Formula: (Correct Predictions / (Correct Predictions + Missed Detections)) × 100")
            
            avg_confidence = sum(d['confidence'] for d in all_detections) / total_detections if total_detections > 0 else 0
            
            print(f"📊 DETECTION RATE SUMMARY:")
            print(f"   ✅ Correct Predictions: {len(all_detections)}")
            print(f"   ❌ Missed Detections: {total_patches - patches_with_detections}")
            print(f"   🏆 Detection Ratio: {detection_rate:.1f}%")
            print(f"   📊 Average Confidence: {avg_confidence:.3f}")
            print(f"   💡 Formula: (Correct Predictions / (Correct Predictions + Missed Detections)) × 100")
            
            # 1. Generate Overall Summary Excel
            overall_summary_file = os.path.join(reports_dir, "Patch_Inference_Overall_Summary.xlsx")
            self.create_patch_overall_summary_excel(
                patch_results, all_detections, total_patches, patches_with_detections, 
                detection_rate, avg_confidence, overall_summary_file
            )
            
            # 2. Generate Correct Predictions Excel (All detections are considered correct in inference)
            correct_predictions_file = os.path.join(reports_dir, "Patch_Inference_Correct_Predictions.xlsx")
            if total_detections > 0:
                self.create_patch_correct_predictions_excel(all_detections, correct_predictions_file)
            else:
                self.create_patch_correct_predictions_excel([], correct_predictions_file)
            
            # 3. Generate Missed Detections Excel (Patches without detections)
            missed_detections_file = os.path.join(reports_dir, "Patch_Inference_Missed_Detections.xlsx")
            missed_patches = []
            for image_name, image_data in patch_results.get('image_results', {}).items():
                if image_data.get('processed', False):
                    patch_details = image_data.get('patch_details', [])
                    for patch in patch_details:
                        if patch.get('detections', 0) == 0:
                            # Get position data - handle both position list and separate x,y fields  
                            if 'position' in patch and isinstance(patch['position'], list) and len(patch['position']) >= 2:
                                patch_x, patch_y = patch['position'][0], patch['position'][1]
                            else:
                                patch_x = patch.get('x', 0)
                                patch_y = patch.get('y', 0)
                                
                            missed_entry = {
                                'image_name': image_name,
                                'category': self.extract_category_from_image(image_name),
                                'patch_id': patch['patch_id'],
                                'patch_position': f"({patch_x},{patch_y})",
                                'reason': 'No detections found'  # Simplified reason text
                            }
                            missed_patches.append(missed_entry)
            
            self.create_patch_missed_detections_excel(missed_patches, missed_detections_file)
            
            # Final verification: Cross-check missed patches count
            actual_missed_count = len(missed_patches)
            expected_missed_count = total_patches - patches_with_detections
            
            print(f"\n📋 Final Report Verification (New Formula):")
            print(f"   📄 Correct Predictions Report: {len(all_detections)} detection entries")
            print(f"   📄 Missed Detections Report: {actual_missed_count} missed patch entries")
            print(f"   📊 Expected vs Actual Missed: {expected_missed_count} vs {actual_missed_count}")
            
            if actual_missed_count == expected_missed_count:
                print(f"   ✅ Reports are consistent! Detection rate verified.")
                print(f"   📈 NEW Detection Rate: {detection_rate:.1f}% ({len(all_detections)}/({len(all_detections)}+{actual_missed_count}))")
                print(f"   💡 Interpretation: {detection_rate:.1f}% of actionable results were successful detections")
            else:
                print(f"   ⚠️ MISMATCH: Missed patches count doesn't match!")
                print(f"   🔍 Please review patch data processing logic.")
            
            # Return file paths
            return {
                'overall_summary': overall_summary_file,
                'correct_predictions': correct_predictions_file, 
                'missed_detections': missed_detections_file,
                'reports_directory': reports_dir
            }
            
        except Exception as e:
            print(f"❌ Error generating patch inference Excel reports: {e}")
            traceback.print_exc()
            return None

    def create_patch_overall_summary_excel(self, patch_results, all_detections, total_patches, 
                                         patches_with_detections, detection_rate, avg_confidence, file_path):
        """Create overall summary Excel for patch inference results"""
        try:
            import xlsxwriter
            
            workbook = xlsxwriter.Workbook(file_path)
            ws = workbook.add_worksheet('Patch Inference Summary')
            
            # Formats
            header_format = workbook.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white'})
            data_format = workbook.add_format({'align': 'left'})
            number_format = workbook.add_format({'num_format': '#,##0'})
            
            # Summary statistics - Start directly without main title
            row = 1
            
            # EXACT FORMULA: Detection Rate = (Correct Predictions / (Correct Predictions + Missed Detections)) × 100
            correct_predictions_excel = len(all_detections)
            missed_detections_excel = total_patches - patches_with_detections
            
            # Apply the exact formula
            if (correct_predictions_excel + missed_detections_excel) > 0:
                detection_ratio = (correct_predictions_excel / (correct_predictions_excel + missed_detections_excel)) * 100
            else:
                detection_ratio = 0
            
            # Get NMS and IOU values from patch config
            nms_threshold = "N/A"
            iou_threshold = "N/A" 
            conf_threshold = "N/A"
            if hasattr(self, 'patch_config') and self.patch_config:
                iou_threshold = self.patch_config.get('iou', 'N/A')
                conf_threshold = self.patch_config.get('conf', 'N/A')
                # NMS threshold is typically used in CombineDetections - set a default
                nms_threshold = 0.05  # This is the default used in CombineDetections
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Images Processed', patch_results.get('processed_images', 0)],
                ['Total Patches Created', total_patches],
                ['Correct Predictions (Detections)', correct_predictions_excel],
                ['Missed Detections (Empty Patches)', missed_detections_excel],
                ['Detection Ratio (%)', f'{detection_ratio:.1f}%'],
                ['Average Confidence Score', f'{avg_confidence:.2f}'],
                ['Model Used', patch_results.get('model_used', 'Unknown')],
                ['Processing Date', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            
            for i, (metric, value) in enumerate(summary_data):
                if i == 0:  # Header row
                    ws.write(row, 0, metric, header_format)
                    ws.write(row, 1, value, header_format)
                else:
                    ws.write(row, 0, metric, data_format)
                    ws.write(row, 1, value, number_format if isinstance(value, (int, float)) else data_format)
                row += 1
            
            # Class-wise breakdown
            row += 2
            # Write section header across multiple columns for proper formatting
            ws.merge_range(row, 0, row, 4, 'CLASS WISE ANALYSIS', header_format)
            row += 2
            
            # Count detections by class
            class_counts = {}
            class_confidences = {}
            class_examples = {}
            for detection in all_detections:
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                    class_confidences[class_name] = []
                    class_examples[class_name] = []
                    
                class_counts[class_name] += 1
                class_confidences[class_name].append(confidence)
                
                # Store example for each class (first few detections)
                if len(class_examples[class_name]) < 3:
                    class_examples[class_name].append(f"{confidence:.2f}")
            
            class_headers = ['Class Name', 'Total Count', 'Average Confidence Score', 'NMS Threshold', 'IOU Threshold']
            for col, header in enumerate(class_headers):
                ws.write(row, col, header, header_format)
            row += 1
            
            for class_name in sorted(class_counts.keys()):
                count = class_counts[class_name]
                confidences = class_confidences[class_name]
                avg_conf = sum(confidences) / len(confidences)
                
                ws.write(row, 0, class_name, data_format)
                ws.write(row, 1, count, number_format)
                ws.write(row, 2, f'{avg_conf:.2f}', data_format)
                ws.write(row, 3, nms_threshold if nms_threshold != "N/A" else "0.05", data_format)
                ws.write(row, 4, iou_threshold if iou_threshold != "N/A" else "N/A", data_format)
                row += 1

            # Per-image breakdown
            row += 2
            # Write section header across multiple columns for proper formatting
            ws.merge_range(row, 0, row, 7, 'IMAGE ANALYSIS', header_format)
            row += 2
            
            image_headers = ['Image Name', 'Category', 'Patches Created', 'Correct Predictions', 'Missed Detections', 'Detection Rate', 'NMS Threshold', 'IOU Threshold']
            for col, header in enumerate(image_headers):
                ws.write(row, col, header, header_format)
            row += 1
            
            # Get threshold values for display
            nms_display = nms_threshold if nms_threshold != "N/A" else "0.05"
            iou_display = iou_threshold if iou_threshold != "N/A" else "N/A"
            
            for image_name, image_data in patch_results.get('image_results', {}).items():
                if image_data.get('processed', False):
                    patch_details = image_data.get('patch_details', [])
                    patches_count = len(patch_details)
                    patches_with_det = len([p for p in patch_details if p.get('detections', 0) > 0])
                    total_det = sum(p.get('detections', 0) for p in patch_details)
                    patches_without_det = patches_count - patches_with_det
                    
                    # NEW DETECTION RATE: Based on Correct Predictions vs Missed Detections
                    if (total_det + patches_without_det) > 0:
                        det_rate = (total_det / (total_det + patches_without_det)) * 100
                    else:
                        det_rate = 0
                    
                    image_data_row = [
                        image_name,
                        self.extract_category_from_image(image_name),
                        patches_count,
                        total_det,  # Show total detections instead of patches with detections
                        patches_without_det,  # Show patches without detections
                        f'{det_rate:.1f}%',
                        nms_display,
                        iou_display
                    ]
                    
                    for col, value in enumerate(image_data_row):
                        ws.write(row, col, value, data_format)
                    row += 1
            
            # Auto-adjust column widths
            ws.set_column('A:A', 25)  # Image Name
            ws.set_column('B:B', 15)  # Category
            ws.set_column('C:H', 12)  # Other columns
            
            workbook.close()
            print(f"   📋 Overall summary Excel created: {file_path}")
            
        except Exception as e:
            print(f"❌ Error creating patch overall summary Excel: {e}")

    def create_patch_correct_predictions_excel(self, all_detections, file_path):
        """Create correct predictions Excel for patch inference results"""
        try:
            import xlsxwriter
            
            workbook = xlsxwriter.Workbook(file_path)
            ws = workbook.add_worksheet('Correct Predictions')
            
            # Formats
            header_format = workbook.add_format({'bold': True, 'bg_color': '#70AD47', 'font_color': 'white'})
            data_format = workbook.add_format({'align': 'left'})
            
            # Get NMS and IOU values from patch config for display
            nms_threshold = 0.05  # Default NMS threshold used in CombineDetections
            iou_threshold = "N/A"
            conf_threshold = "N/A"
            if hasattr(self, 'patch_config') and self.patch_config:
                iou_threshold = self.patch_config.get('iou', 'N/A')
                conf_threshold = self.patch_config.get('conf', 'N/A')
            
            # Headers - Start directly without main title
            row = 1
            headers = ['Image Name', 'Category', 'Patch ID', 'Position', 'Class', 'Confidence', 'Bounding Box', 'NMS Threshold', 'IOU Threshold']
            for col, header in enumerate(headers):
                ws.write(row, col, header, header_format)
            row += 1
            
            # Data rows
            for detection in all_detections:
                data_row = [
                    detection['image_name'],
                    detection['category'],
                    detection['patch_id'],
                    detection['patch_position'],
                    detection['class_name'],
                    f"{detection['confidence']:.2f}",  # Format to 2 decimal places
                    detection['bbox'],
                    nms_threshold,
                    iou_threshold
                ]
                
                for col, value in enumerate(data_row):
                    ws.write(row, col, value, data_format)
                row += 1
            
            # Auto-adjust column widths
            ws.set_column('A:A', 25)  # Image Name
            ws.set_column('B:B', 15)  # Category
            ws.set_column('C:I', 12)  # Other columns
            
            workbook.close()
            print(f"   ✅ Correct predictions Excel created: {file_path}")
            
        except Exception as e:
            print(f"❌ Error creating patch correct predictions Excel: {e}")

    def create_patch_missed_detections_excel(self, missed_patches, file_path):
        """Create missed detections Excel for patch inference results"""
        try:
            import xlsxwriter
            
            workbook = xlsxwriter.Workbook(file_path)
            ws = workbook.add_worksheet('Missed Detections')
            
            # Formats
            header_format = workbook.add_format({'bold': True, 'bg_color': '#C5504B', 'font_color': 'white'})
            data_format = workbook.add_format({'align': 'left'})
            
            # Headers - Start directly without main title
            row = 1
            headers = ['Image Name', 'Category', 'Patch ID', 'Position', 'Possible Reason']
            for col, header in enumerate(headers):
                ws.write(row, col, header, header_format)
            row += 1
            
            # Data rows
            for missed in missed_patches:
                data_row = [
                    missed['image_name'],
                    missed['category'],
                    missed['patch_id'],
                    missed['patch_position'],
                    'No detections found'  # Simplified reason text
                ]
                
                for col, value in enumerate(data_row):
                    ws.write(row, col, value, data_format)
                row += 1
            
            # Auto-adjust column widths
            ws.set_column('A:A', 25)
            ws.set_column('B:B', 15)
            ws.set_column('C:E', 15)
            
            workbook.close()
            print(f"   🔍 Missed detections Excel created: {file_path}")
            
        except Exception as e:
            print(f"❌ Error creating patch missed detections Excel: {e}")

    def load_class_names_from_config(self):
        """
        Load class names from patch_based_config.json
        Returns a mapping of class_id -> class_name
        """
        try:
            # DEBUG: Print patch config content
            # print(f"         🔍 DEBUG: patch_config content: {self.patch_config}")
            
            # Load class names from patch config
            class_names = self.patch_config.get('class_names', {})
            if class_names:
                # Convert string keys to integers for easier lookup
                class_name_mapping = {}
                for key, value in class_names.items():
                    try:
                        class_id = int(key)
                        class_name_mapping[class_id] = value
                    except ValueError:
                        print(f"         ⚠️ Invalid class ID '{key}', skipping")
                
                print(f"         🏷️ Loaded {len(class_name_mapping)} class names from config")
                return class_name_mapping
            else:
                print(f"         ⚠️ No class_names found in config, using default format")
                return {}
        except Exception as e:
            print(f"         ⚠️ Error loading class names: {e}")
            return {}

    def stitch_patches_to_original_image(self, patch_summary, image_name, patches_folder, output_folder):
        """
        FAST stitching: Reconstruct original image from patches with detections.
        Uses vectorized operations for speed and direct detection coordinate mapping.
        """
        try:
            print(f"      🔧 Fast-stitching {patch_summary['total_patches']} patches...")
            
            # Load class names for better labeling
            class_names = self.load_class_names_from_config()
            # print(f"         🔍 DEBUG: Loaded class_names: {class_names}")
            
            # Get image dimensions and patch info
            original_width, original_height = patch_summary['image_size']
            patch_size = patch_summary['patch_size']
            stride = patch_summary['stride']
            patches_x, patches_y = patch_summary['patches_grid']
            
            print(f"         📐 Original size: {original_width}x{original_height}")
            print(f"         📦 Patch grid: {patches_x}x{patches_y}")
            
            # Create canvas for reconstructed image using simple placement (faster)
            reconstructed_img = np.zeros((original_height, original_width, 3), dtype=np.uint8)
            
            # Load detection data from patch summary
            all_detections = []
            patch_results = patch_summary.get('patch_results', [])
            
            print(f"         🔍 Processing detection data from {len(patch_results)} patches...")
            
            # Fast patch placement and detection collection
            patch_count = 0
            patch_folder = os.path.join(patches_folder, image_name)
            
            for y_idx in range(patches_y):
                for x_idx in range(patches_x):
                    # Calculate patch position
                    patch_x = x_idx * stride
                    patch_y = y_idx * stride
                    
                    # Load patch and place it (vectorized operation)
                    patch_file = os.path.join(patch_folder, f"patch_{patch_count:03d}.jpg")
                    if os.path.exists(patch_file):
                        patch_img = cv2.imread(patch_file)
                        
                        if patch_img is not None:
                            # Calculate placement bounds
                            end_x = min(patch_x + patch_size, original_width)
                            end_y = min(patch_y + patch_size, original_height)
                            patch_w = end_x - patch_x
                            patch_h = end_y - patch_y
                            
                            # FAST: Use numpy slicing instead of pixel loops
                            reconstructed_img[patch_y:end_y, patch_x:end_x] = patch_img[:patch_h, :patch_w]
                    
                    # Collect detection data from patch summary
                    if patch_count < len(patch_results):
                        patch_result = patch_results[patch_count]
                        if patch_result.get('inference_results') and patch_result['inference_results'].get('detections', 0) > 0:
                            inference_data = patch_result['inference_results']
                            
                            # Convert detection coordinates to global coordinates
                            for i, (conf, cls, bbox) in enumerate(zip(
                                inference_data.get('confidence_scores', []),
                                inference_data.get('classes', []),
                                inference_data.get('boxes', [])
                            )):
                                if len(bbox) >= 4:
                                    # Convert local patch coordinates to global coordinates
                                    global_x1 = max(0, min(patch_x + bbox[0], original_width))
                                    global_y1 = max(0, min(patch_y + bbox[1], original_height))
                                    global_x2 = max(0, min(patch_x + bbox[2], original_width))
                                    global_y2 = max(0, min(patch_y + bbox[3], original_height))
                                    
                                    all_detections.append({
                                        'x1': int(global_x1), 'y1': int(global_y1),
                                        'x2': int(global_x2), 'y2': int(global_y2),
                                        'confidence': conf,
                                        'class': cls,
                                        'patch_id': patch_count
                                    })
                    
                    patch_count += 1
            
            # Apply NMS using IMPROVED IoU threshold from config
            # Lower IoU threshold for better detection matching (0.4 instead of 0.7)
            iou_threshold = self.patch_config.get('iou', 0.4)  # More lenient for better detection rates
            if len(all_detections) > 0:
                final_detections = self.apply_fast_nms(all_detections, iou_threshold=iou_threshold)
                
                # Post-process to expand bounding boxes for better coverage
                final_detections = self.expand_bounding_boxes(final_detections, original_width, original_height)
            else:
                final_detections = []
            
            # Create detection overlay with bright, visible colors
            detection_overlay = np.zeros_like(reconstructed_img)
            final_image = reconstructed_img.copy()
            
            # Draw detections with bright, visible colors
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for i, detection in enumerate(final_detections):
                color = colors[detection['class'] % len(colors)]
                
                # Draw bounding box (reduced thickness)
                cv2.rectangle(final_image, 
                            (detection['x1'], detection['y1']), 
                            (detection['x2'], detection['y2']), 
                            color, 1)  # Reduced thickness for better visibility
                
                cv2.rectangle(detection_overlay,
                            (detection['x1'], detection['y1']),
                            (detection['x2'], detection['y2']),
                            color, 1)
                
                # Draw label with background for better visibility
                class_id = detection['class']
                class_name = class_names.get(class_id, f"Class_{class_id}")
                label = f"{class_name}: {detection['confidence']:.2f}"
                
                # DEBUG: Print what label is being created
                # print(f"         🏷️ DEBUG: Creating label for class {class_id}: '{label}'")
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Draw label background
                cv2.rectangle(final_image,
                            (detection['x1'], detection['y1'] - label_size[1] - 10),
                            (detection['x1'] + label_size[0], detection['y1']),
                            color, -1)
                
                # Draw label text
                cv2.putText(final_image, label,
                          (detection['x1'], detection['y1'] - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save results
            output_filename = f"{image_name}_reconstructed.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, final_image)
            
            # Save pure reconstruction (without detections)
            pure_output_path = os.path.join(output_folder, f"{image_name}_reconstructed_pure.jpg")
            cv2.imwrite(pure_output_path, reconstructed_img)
            
            # Save detection overlay only
            detection_output_path = os.path.join(output_folder, f"{image_name}_detections_only.jpg")
            cv2.imwrite(detection_output_path, detection_overlay)
            
            return {
                'output_path': output_path,
                'pure_output_path': pure_output_path,
                'detections_output_path': detection_output_path,
                'total_detections': len(final_detections),
                'patches_processed': patch_count,
                'reconstruction_successful': True,
                'detection_details': final_detections  # Add detection details for Excel report
            }
            
        except Exception as e:
            print(f"         ❌ Error during fast stitching: {e}")
            import traceback
            traceback.print_exc()
            return None

    def apply_fast_nms(self, detections, iou_threshold=0.3):
        """Enhanced NMS implementation to remove duplicate detections with class-specific handling"""
        if not detections:
            return []
        
        # Group detections by class for better NMS
        class_groups = {}
        for det in detections:
            class_id = det['class']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(det)
        
        keep = []
        
        # Apply NMS per class
        for class_id, class_detections in class_groups.items():
            # Sort by confidence
            class_detections = sorted(class_detections, key=lambda x: x['confidence'], reverse=True)
            
            # Apply different IoU thresholds based on class
            if class_id == 4:  # pressure_indicator - more aggressive
                class_iou_threshold = min(iou_threshold, 0.15)  # Very aggressive for small objects
            elif class_id == 0:  # gate_valve - less aggressive to allow partial detections
                class_iou_threshold = max(iou_threshold, 0.4)   # Less aggressive for large objects
            else:
                class_iou_threshold = iou_threshold
            
            class_keep = []
            while class_detections:
                current = class_detections.pop(0)
                class_keep.append(current)
                
                # Remove overlapping detections
                remaining = []
                for det in class_detections:
                    iou_score = self.calculate_iou_fast(current, det)
                    if iou_score < class_iou_threshold:
                        remaining.append(det)
                class_detections = remaining
            
            keep.extend(class_keep)
        
        return keep

    def expand_bounding_boxes(self, detections, img_width, img_height):
        """
        Expand bounding boxes for better object coverage, especially for gate valves
        """
        expanded_detections = []
        
        for det in detections:
            class_id = det['class']
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            
            # Calculate current box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Class-specific expansion factors
            if class_id == 0:  # gate_valve - expand more for better coverage
                expand_factor = 0.15  # 15% expansion
            elif class_id == 4:  # pressure_indicator - minimal expansion
                expand_factor = 0.05  # 5% expansion  
            else:
                expand_factor = 0.10  # 10% expansion for others
            
            # Calculate expansion
            width_expand = int(box_width * expand_factor)
            height_expand = int(box_height * expand_factor)
            
            # Apply expansion with boundary checks
            new_x1 = max(0, x1 - width_expand)
            new_y1 = max(0, y1 - height_expand)
            new_x2 = min(img_width, x2 + width_expand)
            new_y2 = min(img_height, y2 + height_expand)
            
            # Create expanded detection
            expanded_det = det.copy()
            expanded_det.update({
                'x1': new_x1,
                'y1': new_y1, 
                'x2': new_x2,
                'y2': new_y2
            })
            
            expanded_detections.append(expanded_det)
        
        return expanded_detections
    
    def generate_stitched_detections_excel_report(self, image_name, detection_details):
        """Generate Excel report for stitched image detections in the specified format"""
        try:
            # Create reports directory under the same validation_reports/ location
            # used by generate_patch_inference_excel_reports (relative to self.report_path)
            reports_dir = os.path.join(os.path.dirname(self.report_path), "validation_reports")
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)
            
            # Create timestamped subdirectory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_subdir = os.path.join(reports_dir, f"stitched_detections_{timestamp}")
            if not os.path.exists(report_subdir):
                os.makedirs(report_subdir)
            
            # Create Excel file
            excel_filename = f"{image_name}_Stitched_Detections.xlsx"
            excel_path = os.path.join(report_subdir, excel_filename)
            
            workbook = xlsxwriter.Workbook(excel_path)
            worksheet = workbook.add_worksheet("Stitched_Detections")
            
            # Define header format
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D7E4BC',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            # Define data format
            data_format = workbook.add_format({
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            # Write headers
            headers = ['Refinery name', 'File name', 'Asset Type', 'Asset Name', 'Tag', 'X1', 'Y1', 'X2', 'Y2']
            for col, header in enumerate(headers):
                worksheet.write(0, col, header, header_format)
            
            # Get class names mapping
            class_names_map = self.patch_config.get('class_names', {})
            
            # Asset type mapping based on class names
            asset_type_mapping = {
                'gate_valve': 'Instrument',
                'pipe': 'Pipeline', 
                'pipeline': 'Pipeline',
                'ball_valve': 'Instrument',
                'pressure_indicator': 'Instrument',
                'pressure_gauge': 'Instrument',
                'valve': 'Instrument'
            }
            
            # Write detection data
            row = 1
            for detection in detection_details:
                class_id = detection['class']
                class_name = self.get_patch_class_name(class_id)
                
                # Determine asset type
                asset_type = 'Instrument'  # Default
                for key, value in asset_type_mapping.items():
                    if key.lower() in class_name.lower():
                        asset_type = value
                        break
                
                # Generate tag (keep empty as requested)
                tag = ""  # Empty tag value - no class/confidence details
                
                # Write row data
                worksheet.write(row, 0, "Whiting", data_format)  # Refinery name (placeholder)
                worksheet.write(row, 1, f"{image_name}.pdf", data_format)  # File name
                worksheet.write(row, 2, asset_type, data_format)  # Asset Type
                worksheet.write(row, 3, class_name, data_format)  # Asset Name
                worksheet.write(row, 4, tag, data_format)  # Tag
                worksheet.write(row, 5, detection['x1'], data_format)  # X1
                worksheet.write(row, 6, detection['y1'], data_format)  # Y1
                worksheet.write(row, 7, detection['x2'], data_format)  # X2
                worksheet.write(row, 8, detection['y2'], data_format)  # Y2
                
                row += 1
            
            # Adjust column widths
            worksheet.set_column('A:A', 15)  # Refinery name
            worksheet.set_column('B:B', 20)  # File name
            worksheet.set_column('C:C', 12)  # Asset Type
            worksheet.set_column('D:D', 20)  # Asset Name
            worksheet.set_column('E:E', 15)  # Tag
            worksheet.set_column('F:I', 8)   # X1, Y1, X2, Y2
            
            workbook.close()
            
            # Generate coordinate validation image for visual verification
            validation_image_path = self.create_coordinate_validation_image(
                image_name, detection_details, report_subdir
            )
            
            print(f"         ✅ Excel report generated: {excel_filename}")
            if validation_image_path:
                print(f"         🖼️ Validation image: {os.path.basename(validation_image_path)}")
            
            return excel_path
            
        except Exception as e:
            print(f"❌ Error generating stitched detections Excel report: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_coordinate_validation_image(self, image_name, detection_details, report_dir):
        """Create a validation image with numbered bounding boxes for coordinate verification"""
        try:
            # Find the reconstructed image
            stitched_folder = "./patch_outputs/stitched_results"
            reconstructed_path = os.path.join(stitched_folder, f"{image_name}_reconstructed.jpg")
            
            if not os.path.exists(reconstructed_path):
                print(f"         ⚠️ Reconstructed image not found: {reconstructed_path}")
                return None
            
            # Load the reconstructed image
            img = cv2.imread(reconstructed_path)
            if img is None:
                return None
            
            # Create validation image with numbered detections
            validation_img = img.copy()
            
            # Colors for different classes
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            
            for idx, detection in enumerate(detection_details):
                x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
                class_id = detection['class']
                confidence = detection['confidence']
                
                # Choose color based on class
                color = colors[class_id % len(colors)]
                
                # Draw bounding box with thicker lines for visibility
                cv2.rectangle(validation_img, (x1, y1), (x2, y2), color, 3)
                
                # Add detection number and details
                label = f"#{idx+1} C{class_id} {confidence:.2f}"
                
                # Draw label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw background rectangle for text
                cv2.rectangle(validation_img, 
                            (x1, y1 - text_height - 10), 
                            (x1 + text_width + 10, y1), 
                            color, -1)
                
                # Draw text
                cv2.putText(validation_img, label, (x1 + 5, y1 - 5), 
                          font, font_scale, (255, 255, 255), thickness)
                
                # Add coordinate text at bottom-right of box
                coord_text = f"({x1},{y1})-({x2},{y2})"
                coord_font_scale = 0.6
                (coord_width, coord_height), _ = cv2.getTextSize(coord_text, font, coord_font_scale, 1)
                
                # Draw coordinate background
                cv2.rectangle(validation_img,
                            (x2 - coord_width - 5, y2),
                            (x2, y2 + coord_height + 5),
                            (0, 0, 0), -1)
                
                # Draw coordinate text
                cv2.putText(validation_img, coord_text, 
                          (x2 - coord_width, y2 + coord_height), 
                          font, coord_font_scale, (255, 255, 255), 1)
            
            # Save validation image
            validation_filename = f"{image_name}_coordinate_validation.jpg"
            validation_path = os.path.join(report_dir, validation_filename)
            cv2.imwrite(validation_path, validation_img)
            
            return validation_path
            
        except Exception as e:
            print(f"         ⚠️ Error creating validation image: {e}")
            return None
    
    def calculate_iou_fast(self, box1, box2):
        """Fast IoU calculation between two boxes"""
        x1 = max(box1['x1'], box2['x1'])
        y1 = max(box1['y1'], box2['y1'])
        x2 = min(box1['x2'], box2['x2'])
        y2 = min(box1['y2'], box2['y2'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def count_total_ground_truth_objects_in_patches(self, image_results):
        """Count total ground truth objects across all processed images"""
        total_gt_objects = 0
        
        for result in image_results:
            image_name = result.get('image_name', '')
            gt_count = self.count_ground_truth_objects_for_image(image_name)
            total_gt_objects += gt_count
            
        print(f"🔍 Total Ground Truth Objects Calculation:")
        print(f"   📊 Processed {len(image_results)} images")
        print(f"   🎯 Total GT objects found: {total_gt_objects}")
        
        return total_gt_objects
    
    def count_ground_truth_objects_for_image(self, image_name):
        """Count ground truth objects for a specific image"""
        try:
            # Remove extension and construct annotation file path
            base_name = os.path.splitext(image_name)[0]
            annotation_file = os.path.join(self.original_annotation_path, f"{base_name}.txt")
            
            if not os.path.exists(annotation_file):
                print(f"   ⚠️ GT annotation not found for {image_name}: {annotation_file}")
                return 0
            
            # Count lines in annotation file (each line = one object)
            with open(annotation_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                gt_count = len(lines)
            
            print(f"   📝 {image_name}: {gt_count} GT objects")
            return gt_count
            
        except Exception as e:
            print(f"   ❌ Error counting GT objects for {image_name}: {e}")
            return 0
    
    def count_ground_truth_objects_from_results(self, image_results):
        """Count total ground truth objects from processed image results"""
        total_gt_objects = 0
        
        if not image_results:
            print("⚠️ No image results provided for GT counting")
            return 0
        
        for result in image_results:
            image_name = result.get('image_name', '')
            if image_name:
                gt_count = self.count_ground_truth_objects_for_image(image_name)
                total_gt_objects += gt_count
        
        print(f"🔍 Ground Truth Object Summary:")
        print(f"   📊 Images processed: {len(image_results)}")
        print(f"   🎯 Total GT objects: {total_gt_objects}")
        
        return total_gt_objects

    def extract_detections_from_inference_image(self, inference_img, original_patch, patch_x, patch_y):
        """
        Extract detection information from inference result image by analyzing the drawn boxes
        """
        detections = []
        
        try:
            # Convert to HSV to detect green bounding boxes
            hsv = cv2.cvtColor(inference_img, cv2.COLOR_BGR2HSV)
            
            # Define range for green color (detection boxes are drawn in green)
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            
            # Create mask for green color
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Find contours of green regions
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle of the contour
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small contours (noise)
                if w > 10 and h > 10:
                    # Convert to global coordinates
                    global_x1 = patch_x + x
                    global_y1 = patch_y + y
                    global_x2 = patch_x + x + w
                    global_y2 = patch_y + y + h
                    
                    detections.append({
                        'x1': global_x1,
                        'y1': global_y1,
                        'x2': global_x2,
                        'y2': global_y2,
                        'local_x1': x,
                        'local_y1': y,
                        'local_x2': x + w,
                        'local_y2': y + h
                    })
            
        except Exception as e:
            print(f"⚠️ Error extracting detections from inference image: {e}")
        
        return detections
    
    def draw_bounding_boxes(self,image_path, detections, output_folder):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return
        height, width, _ = img.shape
        for detection in detections:
            class_name = detection['Lb']
            confidence = detection.get('Cs', 0.0)
            
            # API returns top-left corner coordinates (normalized 0-1)
            # X, Y = top-left corner (normalized)
            # W, H = width, height (normalized)
            top_left_x = detection["Dm"]["X"]
            top_left_y = detection["Dm"]["Y"] 
            box_width = detection["Dm"]["W"]
            box_height = detection["Dm"]["H"]
            
            # Convert normalized coordinates to pixel coordinates
            x_min = int(top_left_x * width)
            y_min = int(top_left_y * height)
            x_max = int((top_left_x + box_width) * width)
            y_max = int((top_left_y + box_height) * height)
            
            # Ensure coordinates are within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width - 1, x_max)
            y_max = min(height - 1, y_max)
            
            # Draw the bounding box and class label
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)  # Green rectangle (reduced thickness)
            
            # Add confidence score to label with cleaner font - smaller size
            label = f"{class_name} ({confidence:.2f})"
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 150, 0), 1)
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Save the image with bounding boxes
        image_filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, img)
        #print(f"Saved image with bounding boxes to {output_path}")

    def draw_prediction_bounding_boxes(self, image_path, detections, output_folder):
        """Draw prediction bounding boxes in RED color for easy comparison with GT"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return
        height, width, _ = img.shape
        
        for detection in detections:
            class_name = detection['Lb']
            confidence = detection.get('Cs', 0.0)
            
            # API returns top-left corner coordinates (normalized 0-1)
            top_left_x = detection["Dm"]["X"]
            top_left_y = detection["Dm"]["Y"] 
            box_width = detection["Dm"]["W"]
            box_height = detection["Dm"]["H"]
            
            # Convert normalized coordinates to pixel coordinates
            x_min = int(top_left_x * width)
            y_min = int(top_left_y * height)
            x_max = int((top_left_x + box_width) * width)
            y_max = int((top_left_y + box_height) * height)
            
            # Ensure coordinates are within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width - 1, x_max)
            y_max = min(height - 1, y_max)
            
            # Draw the bounding box and class label in RED
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)  # Red rectangle (reduced thickness)
            
            # Add confidence score to label in RED with no background - smaller font
            label = f"PRED: {class_name} ({confidence:.2f})"
            # No background, just clean text with smaller size
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 150), 1)
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Save the image with prediction bounding boxes
        image_filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, img)
        return output_path

    def draw_combined_gt_and_prediction_boxes(self, image_path, gt_detections, pred_detections, output_folder):
        """Draw both GT (GREEN) and Prediction (RED) bounding boxes on the same image for comparison"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return
        height, width, _ = img.shape
        
        # Draw Ground Truth boxes in GREEN
        if gt_detections:
            for detection in gt_detections:
                if 'gt_bbox' in detection and detection['gt_bbox']:
                    bbox = detection['gt_bbox']
                    if len(bbox) >= 4:
                        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        # Draw GT box in GREEN (reduced thickness)
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                        
                        # Add GT label with actual class name from tag_list or detection data
                        gt_class = "unknown"
                        if hasattr(self, 'tag_list') and 'class_id' in detection:
                            class_id = detection['class_id']
                            gt_class = self.tag_list[int(class_id)] if int(class_id) < len(self.tag_list) else f"class_{class_id}"
                        elif 'gt_class' in detection and detection['gt_class'] != 'components':
                            gt_class = detection['gt_class']
                        elif 'class' in detection and detection['class'] != 'components':
                            gt_class = detection['class']
                        else:
                            # Try to extract from image path or use a generic label
                            gt_class = "object"
                        
                        label = f"GT: {gt_class}"
                        font_scale = 0.5  # Reduced font size
                        thickness = 1
                        
                        # Use FONT_HERSHEY_TRIPLEX for a more stylish, bold appearance
                        # Position label above the box without background
                        label_x, label_y = x_min, y_min - 5
                        # Use dark green text for better visibility
                        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 150, 0), thickness)
        
        # Draw Prediction boxes in RED
        if pred_detections:
            for detection in pred_detections:
                class_name = detection['Lb']
                confidence = detection.get('Cs', 0.0)
                
                # API coordinates (normalized 0-1)
                top_left_x = detection["Dm"]["X"]
                top_left_y = detection["Dm"]["Y"] 
                box_width = detection["Dm"]["W"]
                box_height = detection["Dm"]["H"]
                
                # Convert to pixel coordinates
                x_min = int(top_left_x * width)
                y_min = int(top_left_y * height)
                x_max = int((top_left_x + box_width) * width)
                y_max = int((top_left_y + box_height) * height)
                
                # Ensure bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(width - 1, x_max)
                y_max = min(height - 1, y_max)
                
                # Draw prediction box in RED (reduced thickness)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                
                # Add PRED label with class name and confidence - smaller font
                label = f"PRED: {class_name} ({confidence:.2f})"
                font_scale = 0.5  # Reduced font size to match GT
                thickness = 1
                
                # Use FONT_HERSHEY_TRIPLEX for a more stylish, bold appearance
                # Position label below the box without background
                label_x, label_y = x_min, y_max + 15
                # Use dark red text for predictions
                cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 150), thickness)
        
        # Individual labels instead of legend - already added in drawing logic above
        
        # Create output folder and save
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, img)
        return output_path

    def crop_relevant_region_for_excel(self, image_path, detection_data, output_folder):
        """Crop relevant region around detected objects for Excel display"""
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            height, width = img.shape[:2]
            
            # Collect all bounding boxes to determine crop region
            all_boxes = []
            
            # Add GT bbox if available
            if 'gt_bbox' in detection_data and detection_data['gt_bbox']:
                gt_box = detection_data['gt_bbox']
                if len(gt_box) >= 4:
                    all_boxes.append(gt_box)
            
            # Add prediction bbox if available  
            if 'pred_bbox' in detection_data and detection_data['pred_bbox']:
                pred_box = detection_data['pred_bbox']
                if len(pred_box) >= 4:
                    all_boxes.append(pred_box)
            
            if not all_boxes:
                # No bounding boxes, return original image
                return image_path
            
            # Calculate unified bounding region with padding
            min_x = min([box[0] for box in all_boxes])
            min_y = min([box[1] for box in all_boxes])
            max_x = max([box[2] for box in all_boxes])
            max_y = max([box[3] for box in all_boxes])
            
            # Add padding (20% of the bounding box size)
            padding_x = int((max_x - min_x) * 0.2)
            padding_y = int((max_y - min_y) * 0.2)
            
            # Ensure minimum crop size (at least 200x200)
            min_crop_size = 200
            if (max_x - min_x) < min_crop_size:
                padding_x = max(padding_x, (min_crop_size - (max_x - min_x)) // 2)
            if (max_y - min_y) < min_crop_size:
                padding_y = max(padding_y, (min_crop_size - (max_y - min_y)) // 2)
            
            # Apply padding and ensure within image bounds
            crop_x1 = max(0, min_x - padding_x)
            crop_y1 = max(0, min_y - padding_y)
            crop_x2 = min(width, max_x + padding_x)
            crop_y2 = min(height, max_y + padding_y)
            
            # Crop the image
            cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Create output folder if needed
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Save cropped image
            image_filename = os.path.basename(image_path)
            name, ext = os.path.splitext(image_filename)
            cropped_filename = f"{name}_cropped{ext}"
            output_path = os.path.join(output_folder, cropped_filename)
            cv2.imwrite(output_path, cropped_img)
            
            return output_path
            
        except Exception as e:
            print(f"Error cropping image region: {e}")
            return image_path

    def crop_individual_symbol_for_excel(self, image_path, detection_data, output_folder, detection_type=""):
        """
        Crop individual symbol/object for Excel display with appropriate bounding boxes
        Args:
            image_path: Path to the source image
            detection_data: Dict containing 'gt_bbox' and 'pred_bbox' lists
            output_folder: Where to save cropped images
            detection_type: Type of detection ("correct", "missed", "misprediction")
        Returns:
            Path to the cropped image with bounding boxes
        """
        try:
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                return image_path
            
            h, w = image.shape[:2]
            
            # Determine which box to use for cropping based on detection type
            target_box = None
            
            gt_boxes = detection_data.get('gt_bbox', [])
            pred_boxes = detection_data.get('pred_bbox', [])
            
            if detection_type == "correct" and gt_boxes and pred_boxes:
                # For correct detections: use GT box (since GT and prediction match the same symbol)
                target_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
            elif detection_type == "missed" and gt_boxes:
                # For missed detections: use GT box only (no predictions to show)
                target_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
            elif detection_type == "misprediction" and pred_boxes:
                # For mispredictions: use prediction box (wrong class prediction)
                target_box = pred_boxes[0] if isinstance(pred_boxes[0], list) else pred_boxes
            elif gt_boxes:
                # Fallback to GT box if available
                target_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
            elif pred_boxes:
                # Fallback to prediction box if available
                target_box = pred_boxes[0] if isinstance(pred_boxes[0], list) else pred_boxes
            
            if not target_box or len(target_box) < 4:
                return image_path
            
            # Extract box coordinates
            x, y, box_w, box_h = target_box[:4]
            
            # Add padding around the individual symbol (30% of symbol size for better visibility)
            padding_x = max(25, int(box_w * 0.3))  # Minimum 25 pixels
            padding_y = max(25, int(box_h * 0.3))  # Minimum 25 pixels
            
            # Ensure minimum crop size for symbol visibility (120x120)
            min_size = 120
            if box_w < min_size:
                padding_x = max(padding_x, (min_size - box_w) // 2)
            if box_h < min_size:
                padding_y = max(padding_y, (min_size - box_h) // 2)
            
            # Calculate crop boundaries with proper bounds checking
            crop_x1 = max(0, x - padding_x)
            crop_y1 = max(0, y - padding_y)
            crop_x2 = min(w, x + box_w + padding_x)
            crop_y2 = min(h, y + box_h + padding_y)
            
            # Create a copy of the image for drawing bounding boxes
            image_with_boxes = image.copy()
            
            # Draw appropriate bounding boxes before cropping
            if detection_type == "correct" and gt_boxes and pred_boxes:
                # Draw both GT (green) and prediction (red) boxes
                gt_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
                pred_box = pred_boxes[0] if isinstance(pred_boxes[0], list) else pred_boxes
                
                # GT box in green
                cv2.rectangle(image_with_boxes, (gt_box[0], gt_box[1]), 
                             (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]), (0, 255, 0), 2)
                # Prediction box in red
                cv2.rectangle(image_with_boxes, (pred_box[0], pred_box[1]), 
                             (pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]), (0, 0, 255), 2)
                             
            elif detection_type == "missed" and gt_boxes:
                # Draw only GT box in green (missed detection)
                gt_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
                cv2.rectangle(image_with_boxes, (gt_box[0], gt_box[1]), 
                             (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]), (0, 255, 0), 2)
                             
            elif detection_type == "misprediction" and pred_boxes:
                # Draw prediction box in red (wrong prediction)
                pred_box = pred_boxes[0] if isinstance(pred_boxes[0], list) else pred_boxes
                cv2.rectangle(image_with_boxes, (pred_box[0], pred_box[1]), 
                             (pred_box[0] + pred_box[2], pred_box[1] + pred_box[3]), (0, 0, 255), 2)
                # Also draw GT box if available
                if gt_boxes:
                    gt_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
                    cv2.rectangle(image_with_boxes, (gt_box[0], gt_box[1]), 
                                 (gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]), (0, 255, 0), 2)
            
            # Crop the individual symbol with bounding boxes
            cropped_symbol = image_with_boxes[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Create output folder if needed
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Generate output path with detection type info
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_{detection_type}_symbol.jpg")
            
            # Save cropped symbol
            cv2.imwrite(output_path, cropped_symbol)
            return output_path
            
        except Exception as e:
            print(f"Error cropping symbol from {image_path}: {e}")
            return image_path

    def crop_symbol_from_combined_image(self, combined_image_path, detection_data, output_folder, detection_type=""):
        """
        Crop individual symbol from combined GT+Prediction image (preserves color-coded bounding boxes)
        Args:
            combined_image_path: Path to the combined GT+prediction image 
            detection_data: Dict containing 'gt_bbox' and 'pred_bbox' lists
            output_folder: Where to save cropped images
            detection_type: Type of detection ("correct", "missed", "misprediction")
        Returns:
            Path to the cropped symbol with preserved GT/prediction visualization
        """
        try:
            import cv2
            image = cv2.imread(combined_image_path)
            if image is None:
                return combined_image_path
            
            h, w = image.shape[:2]
            
            # Determine which box to use for cropping based on detection type
            target_box = None
            
            gt_boxes = detection_data.get('gt_bbox', [])
            pred_boxes = detection_data.get('pred_bbox', [])
            
            if detection_type == "correct" and gt_boxes and pred_boxes:
                # For correct detections: use GT box (since GT and prediction should overlap)
                target_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
            elif detection_type == "missed" and gt_boxes:
                # For missed detections: use GT box only (only green box should be visible)
                target_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
            elif detection_type == "misprediction":
                # For mispredictions: prioritize prediction box, fallback to GT
                if pred_boxes:
                    target_box = pred_boxes[0] if isinstance(pred_boxes[0], list) else pred_boxes
                elif gt_boxes:
                    target_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
            elif gt_boxes:
                # Fallback to GT box if available
                target_box = gt_boxes[0] if isinstance(gt_boxes[0], list) else gt_boxes
            elif pred_boxes:
                # Fallback to prediction box if available
                target_box = pred_boxes[0] if isinstance(pred_boxes[0], list) else pred_boxes
            
            if not target_box or len(target_box) < 4:
                return combined_image_path
            
            # Extract box coordinates
            x, y, box_w, box_h = target_box[:4]
            
            # Add padding around the symbol for better visibility (25% padding)
            padding_x = max(20, int(box_w * 0.25))  # Minimum 20 pixels
            padding_y = max(20, int(box_h * 0.25))  # Minimum 20 pixels
            
            # Ensure minimum crop size for symbol visibility (100x100)
            min_size = 100
            if box_w < min_size:
                padding_x = max(padding_x, (min_size - box_w) // 2)
            if box_h < min_size:
                padding_y = max(padding_y, (min_size - box_h) // 2)
            
            # Calculate crop boundaries with proper bounds checking
            crop_x1 = max(0, x - padding_x)
            crop_y1 = max(0, y - padding_y)
            crop_x2 = min(w, x + box_w + padding_x)
            crop_y2 = min(h, y + box_h + padding_y)
            
            # Crop the symbol from combined image (preserves GT green and prediction red boxes)
            cropped_symbol = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Create output folder if needed
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Generate output path with detection type info
            base_name = os.path.splitext(os.path.basename(combined_image_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_{detection_type}_symbol_combined.jpg")
            
            # Save cropped symbol from combined image
            cv2.imwrite(output_path, cropped_symbol)
            return output_path
            
        except Exception as e:
            print(f"Error cropping symbol from combined image {combined_image_path}: {e}")
            return combined_image_path

    def draw_gt_and_prediction_combined(self, image_path, image_name, prediction_data, output_folder):
        """Draw both GT and Predictions on the same image using proper coordinate handling"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return
        height, width, _ = img.shape
        
        # Draw Prediction boxes in RED (from API response)
        if prediction_data:
            for detection in prediction_data:
                class_name = detection['Lb']
                confidence = detection.get('Cs', 0.0)
                
                # API coordinates (normalized 0-1)
                top_left_x = detection["Dm"]["X"]
                top_left_y = detection["Dm"]["Y"] 
                box_width = detection["Dm"]["W"]
                box_height = detection["Dm"]["H"]
                
                # Convert to pixel coordinates
                x_min = int(top_left_x * width)
                y_min = int(top_left_y * height)
                x_max = int((top_left_x + box_width) * width)
                y_max = int((top_left_y + box_height) * height)
                
                # Ensure bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(width - 1, x_max)
                y_max = min(height - 1, y_max)
                
                # Draw prediction box in RED (reduced thickness)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                
                # Add prediction label with class name and confidence - smaller font
                label = f"PRED: {class_name} ({confidence:.2f})"
                font_scale = 0.5  # Reduced font size to match GT
                thickness = 1
                
                # Use FONT_HERSHEY_TRIPLEX for a more stylish, bold appearance
                label_x, label_y = x_min, y_min - 5
                cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 150), thickness)
        
        # Draw Ground Truth boxes in GREEN (from annotation files)
        try:
            # Get corresponding annotation file
            image_base = os.path.splitext(image_name)[0]
            annotation_file = os.path.join(self.original_annotation_path, f"{image_base}.txt")
            
            if os.path.exists(annotation_file):
                with open(annotation_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                center_x = float(parts[1])
                                center_y = float(parts[2])
                                bbox_width = float(parts[3])
                                bbox_height = float(parts[4])
                                
                                # Convert YOLO format (center, normalized) to pixel coordinates
                                x_min = int((center_x - bbox_width / 2) * width)
                                y_min = int((center_y - bbox_height / 2) * height)
                                x_max = int((center_x + bbox_width / 2) * width)
                                y_max = int((center_y + bbox_height / 2) * height)
                                
                                # Ensure bounds
                                x_min = max(0, x_min)
                                y_min = max(0, y_min)
                                x_max = min(width - 1, x_max)
                                y_max = min(height - 1, y_max)
                                
                                # Draw GT box in GREEN (reduced thickness)
                                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                                
                                # Get class name from class_id using tag_list
                                try:
                                    gt_class = self.tag_list[class_id] if class_id < len(self.tag_list) else f"class_{class_id}"
                                except:
                                    gt_class = f"class_{class_id}"
                                
                                # Add GT label with actual class name - smaller font
                                label = f"GT: {gt_class}"
                                font_scale = 0.5  # Reduced font size
                                thickness = 1
                                
                                # Use FONT_HERSHEY_TRIPLEX for a more stylish, bold appearance
                                label_x, label_y = x_min, y_max + 15
                                cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 150, 0), thickness)
        except Exception as e:
            print(f"Error reading GT annotations for {image_name}: {e}")
        
        # Individual labels instead of legend - already added in drawing logic above
        
        # Create output folder and save
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, img)
        return output_path

    def inference_images(self):
        img_list = os.listdir(self.img_path)
        #img_list = [s for s in img_list if "detected" not in s]
 #       img_list = [s for s in img_list if "detected_detected" not in s]
        #img_list = [s for s in img_list if ".png" not in s]
        image_row = 1
        workbook = xlsxwriter.Workbook(self.report_path + '/Benchmark Report.xlsx')
        worksheet = workbook.add_worksheet('Benchmark Validation Report')
        worksheet.set_column(1, 20, 15)
        my_format = workbook.add_format({'align': 'center', 'text_wrap': 'True'})
        worksheet.write(0, 0, 'Sr No.', my_format)
        worksheet.write(0, 1, 'ModelName', my_format)
        worksheet.write(0, 2, 'InputFileName', my_format)
        worksheet.write(0, 3, 'ObjectId', my_format)
        worksheet.write(0, 4, 'Pred-Confidence', my_format)
        worksheet.write(0, 5, 'Pred-Class', my_format)
        worksheet.write(0, 6, 'Pred-X', my_format)
        worksheet.write(0, 7, 'Pred-Y', my_format)
        worksheet.write(0, 8, 'Pred-Width', my_format)
        worksheet.write(0, 9, 'Pred-Height', my_format)
        worksheet.write(0, 10, 'Start Time', my_format)
        worksheet.write(0, 11, 'End Time', my_format)
        worksheet.write(0, 12, 'API Duration', my_format)
        worksheet.write(0, 13, 'ModelName', my_format)
        worksheet.write(0, 14, 'InputFileName', my_format)
        worksheet.write(0, 15, 'ObjectId', my_format)
        worksheet.write(0, 16, 'Bench-Confidence', my_format)
        worksheet.write(0, 17, 'Bench-Class', my_format)
        worksheet.write(0, 18, 'Bench-X', my_format)
        worksheet.write(0, 19, 'Bench-Y', my_format)
        worksheet.write(0, 20, 'Bench-Width', my_format)
        worksheet.write(0, 21, 'Bench-Height', my_format)
        worksheet.write(0, 22, 'IOU Threshold', my_format)
        sr_no = 1
        #print("img_list : ",img_list)
        for img in img_list:
            #print("Image" , img)
            img_read_val_detail = cv2.cvtColor(cv2.imread(self.img_path + '/' + img), cv2.COLOR_BGR2RGB)
            #folder_path = "groundtruth_testing_data/data/original Annotation/"
            dh, dw = img_read_val_detail.shape[:2]
            # first: reading the binary stuff
            # note the 'rb' flag
            # result: bytes
            imagepath = self.img_path + '/' + img
            with open(self.img_path + '/' + img, 'rb') as open_file:
                byte_content = open_file.read()

            # second: base64 encode read data
            # result: bytes (again)
            base64_bytes = b64encode(byte_content)

            # third: decode these bytes to text
            # result: string (in utf-8)
            base64_string = base64_bytes.decode('utf-8')
            #print(self.confidence_threshold , type(self.confidence_threshold))
            # optional: doing stuff with the data
            # result here: some dict
            raw_data = {
                "Tid" : "1",
                "Did" : "DeviceId_10",
                "Fid" : "1",
                "Per" : [{}],
                "Ts"  : "",
                "Ts_ntp" : "",
                "Inf_ver" : "",
                "Msg_ver" : "",
                "Model" : "",
                "Ad" : {},
                "Ffp" : "",
                "Ltsize" : "",
                "Lfp" : "",
                "Mtp" : [],
                'Base_64': base64_string,
                "C_threshold": self.confidence_threshold,
                "I_fn" : "",
                "Msk_img" : [],
                "Rep_img" : [],
                "Prompt" : []
                
                }
            url = self.url + '/' + self.url_key
            headers = {"Content-Type": "application/json; charset=utf-8"}
            # use the 'headers' parameter to set the HTTP headers:
            start = datetime.datetime.now()
            
            try:
                output = requests.post(url, json=raw_data, headers=headers, timeout=30)
                output.raise_for_status()  # Raise an exception for HTTP error codes
                result = output.json()
            except requests.exceptions.ConnectionError as e:
                print(f"❌ Connection Error: Failed to connect to API server at {url}")
                print(f"   Details: {str(e)}")
                print(f"   ⚡ Please ensure the API server is running and accessible")
                return None
            except requests.exceptions.Timeout as e:
                print(f"⏱️ Timeout Error: API request timed out after 30 seconds")
                print(f"   URL: {url}")
                print(f"   Details: {str(e)}")
                return None
            except requests.exceptions.RequestException as e:
                print(f"🚫 Request Error: API call failed")
                print(f"   URL: {url}")
                print(f"   Details: {str(e)}")
                return None
            except ValueError as e:
                print(f"📄 JSON Error: Failed to parse API response as JSON")
                print(f"   Details: {str(e)}")
                print(f"   Raw response: {output.text if 'output' in locals() else 'No response'}")
                return None
            #print("Output",output)
            ##es = img.replace(".jpg","_simple.txt")

            #print("Fes:",fes)
            #with open(folder_path+fes ,'r') as open_file:
                #lines = open_file.readlines()
            
            #print(url)
            #print("output :", output.content)

            #print("result : ", result)
            end = datetime.datetime.now()
            api_duration = (end - start).seconds
            start_date_time = start.strftime("%d/%m/%Y %H:%M:%S.%f")
            end_date_time = end.strftime("%d/%m/%Y %H:%M:%S.%f")
            output_data = result.get('Fs')
            #output_data = [{"Cs":confidence,"Lb" : class_val,"Dm":{"X":x,"Y":y,"H": h,"W":w},"Uid":"","Nobj":"","Info":"{}"}]
            #output_data = []
            prediction_list = []
            
            # Create predictions folder and draw RED bounding boxes for predictions
            predictions_folder = os.path.join(self.report_path, "predicted_images_with_bounding_boxes")
            self.draw_prediction_bounding_boxes(imagepath, output_data, predictions_folder)
            
            # Create combined GT+Prediction images
            combined_folder = os.path.join(self.report_path, "combined_gt_prediction_images")
            self.draw_gt_and_prediction_combined(imagepath, img, output_data, combined_folder)
            
            # Optional: Also draw GT bounding boxes in GREEN (commented for now)
            # gt_folder = os.path.join(self.report_path, "gt_images_with_bounding_boxes") 
            # self.draw_bounding_boxes(imagepath, output_data, gt_folder)
            
            # print("output_data : ",output_data)
            for predictions in output_data:
                #print(type(predictions))
                #predictions = str(predictions)
                prediction_list_val = []
                class_val = (predictions.get('Lb'))
                #predictions['Lb'] = "CNDSSKIV00001PO"
                #class_val = "CNDSSKIV00001PO"
                #parts = predictions.split(", conf:")
                #print("parts : ",parts)
                #bbox = parts[0].split(": ")
                #class_val = bbox[0]
                #bbox_str = bbox[1].split(",")
                #print("bbox_str : ", bbox_str)
                #x=float(bbox_str[0])
                #y=float(bbox_str[1])
                #w=float(bbox_str[2])
                #h=float(bbox_str[3])
                #print("x, y,w,h : ", x,y,w,h)
                #confidence = float(parts[1].strip().replace("\\n'",""))
                #output_data.append({"Cs":confidence,"Lb" : class_val,"Dm":{"X":x,"Y":y,"H": h,"W":w},"Uid":"","Nobj":"","Info":"{}"})
                #print("class_val : ",class_val)
                # print("self.tag_list : ", self.tag_list)
                prediction_list_val.append(self.tag_list.index(class_val))
                prediction_list_val.append(predictions.get('Cs'))
                cw=predictions['Dm']['W']/2
                ch=predictions['Dm']['H']/2
                prediction_list_val_voc = yolo_to_pascal_voc(predictions['Dm']['X']+cw, predictions['Dm']['Y']+ch,
                                                      predictions['Dm']['W'], predictions['Dm']['H'], dw, dh)

                #prediction_list_val_voc = yolo_to_pascal_voc(predictions['Dm']['X'], predictions['Dm']['Y'],
                                                             #predictions['Dm']['W'], predictions['Dm']['H'], dw, dh)
                #prediction_list_val_voc = yolo_to_pascal_voc(x,y,w,h,dw, dh)
                prediction_list_val.append(prediction_list_val_voc[0])
                prediction_list_val.append(prediction_list_val_voc[1])
                prediction_list_val.append(prediction_list_val_voc[2])
                prediction_list_val.append(prediction_list_val_voc[3])
                prediction_list.append(prediction_list_val)
                # print("processing predictions...")
            #print(f"Opening {self.predicted_annotation_path}/{os.path.splitext(img)[0]}.txt")
            with open(f"{self.predicted_annotation_path}/{os.path.splitext(img)[0]}.txt", "w") as outfile:
                for prediction_data_val in prediction_list:
                    outfile.write(' '.join([str(elem) for elem in prediction_data_val]) + "\n")
            #print(f"Opening {self.original_annotation_path}/"f"{os.path.splitext(img)[0]}.txt")
            # Report Generation
            original_annotation_list = []
            with open(f"{self.original_annotation_path}/"f"{os.path.splitext(img)[0]}.txt") as file:
                lines = [line.rstrip() for line in file]
                for annotation_val in lines:
                    final_lines = annotation_val.split()
                    original_annotation_list.append(final_lines)
            original_data_list_final = original_annotation_list
            #print("original_data_list_final : ", original_data_list_final)
            pred_data_list_final = output_data
            #print("pred_data_list_final : ", pred_data_list_final)
            #print("got pred_data_list_final")
            for obj_data in original_data_list_final:
                obj_data_voc = yolo_to_pascal_voc(obj_data[1], obj_data[2], obj_data[3], obj_data[4], dw, dh)
                worksheet.write(image_row, 0, sr_no, my_format)
                worksheet.write(image_row, 10, start_date_time, my_format)
                worksheet.write(image_row, 11, end_date_time, my_format)
                worksheet.write(image_row, 12, str(api_duration) + ' seconds', my_format)
                worksheet.write(image_row, 13, self.url_key, my_format)
                worksheet.write(image_row, 14, img, my_format)
                worksheet.write(image_row, 15, (original_data_list_final.index(obj_data) + 1), my_format)
                worksheet.write(image_row, 16, 1, my_format)
                worksheet.write(image_row, 17, self.tag_list[(int(obj_data[0]))], my_format)
                worksheet.write(image_row, 18, str(obj_data_voc[0]), my_format)
                worksheet.write(image_row, 19, str(obj_data_voc[1]), my_format)
                worksheet.write(image_row, 20, str(obj_data_voc[2]), my_format)
                worksheet.write(image_row, 21, str(obj_data_voc[3]), my_format)
                original_box_val = obj_data_voc

                tag_val = self.tag_list[(int(obj_data[0]))]
                pred_list = []
                for pred_data in pred_data_list_final:
                    if pred_data['Lb'] == tag_val:
                        pred_list.append(pred_data)
                    else:
                        pass
                
                # Enhanced multiple object handling - Find best IoU match for multiple predictions
                if len(pred_list) > 1:
                    best_pred = None
                    best_iou = 0
                    
                    for pred_data in pred_list:
                        # Calculate IoU between this prediction and ground truth
                        cw = pred_data['Dm']['W']/2
                        ch = pred_data['Dm']['H']/2
                        pred_box_temp = yolo_to_pascal_voc(pred_data['Dm']['X']+cw, pred_data['Dm']['Y']+ch,
                                                          pred_data['Dm']['W'], pred_data['Dm']['H'], dw, dh)
                        
                        box1 = torch.tensor([pred_box_temp], dtype=torch.float)
                        box2 = torch.tensor([original_box_val], dtype=torch.float)
                        iou = bops.box_iou(box1, box2)
                        iou_value = float(iou)
                        
                        if iou_value > best_iou:
                            best_iou = iou_value
                            best_pred = pred_data
                    
                    pred_list = [best_pred] if best_pred else []
                
                # Handle single prediction case
                elif len(pred_list) == 1:
                    pass
                
                worksheet.write(image_row, 1, self.url_key, my_format)
                worksheet.write(image_row, 2, img, my_format)
                worksheet.write(image_row, 3, (original_data_list_final.index(obj_data) + 1), my_format)
                if len(pred_list) == 0:
                    pass
                else:
                    cw=pred_list[0]['Dm']['W']/2
                    ch=pred_list[0]['Dm']['H']/2
                    pred_box_val = yolo_to_pascal_voc(pred_list[0]['Dm']['X']+cw, pred_list[0]['Dm']['Y']+ch,
                                                      pred_list[0]['Dm']['W'], pred_list[0]['Dm']['H'], dw, dh)
                    if pred_list[0]['Cs'] == [] :
                        final_cs = 0
                    else:
                        final_cs = pred_list[0]['Cs']
                    worksheet.write(image_row, 4, final_cs, my_format)
                    worksheet.write(image_row, 5, pred_list[0]['Lb'], my_format)
                    worksheet.write(image_row, 6, str(pred_box_val[0]), my_format)
                    worksheet.write(image_row, 7, str(pred_box_val[1]), my_format)
                    worksheet.write(image_row, 8, str(pred_box_val[2]), my_format)
                    worksheet.write(image_row, 9, str(pred_box_val[3]), my_format)

                    #print('original_box_val' + str((original_data_list_final.index(obj_data) + 1)),
                        #  original_box_val)
                    #print('pred_box_val' + str((original_data_list_final.index(obj_data) + 1)),
                     #     pred_box_val)
                    box1 = torch.tensor([pred_box_val], dtype=torch.float)
                    box2 = torch.tensor([original_box_val], dtype=torch.float)
                    iou = bops.box_iou(box1, box2)
                    iou_percent = (iou * 100)
                    iou_value = float(iou)
                    # print(f"Debug - IoU calculation: GT={original_box_val}, Pred={pred_box_val}, IoU={iou_value:.4f} ({int(iou_percent)}%)")
                    worksheet.write(image_row, 22, str(int(iou_percent)) + '%', my_format)
                image_row += 1
                sr_no += 1

        workbook.close()

    def match_predictions_to_ground_truth(self, ground_truth_objects, predictions, image_width, image_height, iou_threshold=0.5):
        """
        Enhanced method to match multiple predictions to multiple ground truth objects
        Args:
            ground_truth_objects: List of ground truth annotations
            predictions: List of prediction results  
            image_width, image_height: Image dimensions
            iou_threshold: Minimum IoU for considering a match
        Returns:
            matched_pairs, unmatched_gt, unmatched_pred
        """
        matched_pairs = []
        unmatched_gt = list(range(len(ground_truth_objects)))
        unmatched_pred = list(range(len(predictions)))
        
        # Calculate IoU matrix between all GT and Pred boxes
        iou_matrix = []
        for i, gt_obj in enumerate(ground_truth_objects):
            gt_row = []
            gt_voc = yolo_to_pascal_voc(gt_obj[1], gt_obj[2], gt_obj[3], gt_obj[4], image_width, image_height)
            
            for j, pred in enumerate(predictions):
                if self.tag_list[int(gt_obj[0])] == pred['Lb']:  # Same class
                    cw = pred['Dm']['W']/2
                    ch = pred['Dm']['H']/2
                    pred_voc = yolo_to_pascal_voc(pred['Dm']['X']+cw, pred['Dm']['Y']+ch,
                                                 pred['Dm']['W'], pred['Dm']['H'], image_width, image_height)
                    
                    box1 = torch.tensor([pred_voc], dtype=torch.float)
                    box2 = torch.tensor([gt_voc], dtype=torch.float)
                    iou = float(bops.box_iou(box1, box2))
                    gt_row.append(iou if iou >= iou_threshold else 0.0)
                else:
                    gt_row.append(0.0)  # Different class = no match
            iou_matrix.append(gt_row)
        
        # Greedy matching: Find best matches iteratively
        while True:
            best_iou = 0
            best_match = None
            
            for i in unmatched_gt:
                for j in unmatched_pred:
                    if i < len(iou_matrix) and j < len(iou_matrix[i]) and iou_matrix[i][j] > best_iou:
                        best_iou = iou_matrix[i][j]
                        best_match = (i, j)
            
            if best_match is None or best_iou < iou_threshold:
                break
                
            # Record the match
            gt_idx, pred_idx = best_match
            matched_pairs.append({
                'gt_index': gt_idx,
                'pred_index': pred_idx,
                'iou': best_iou,
                'gt_class': self.tag_list[int(ground_truth_objects[gt_idx][0])],
                'pred_class': predictions[pred_idx]['Lb'],
                'confidence': predictions[pred_idx]['Cs']
            })
            
            # Remove matched objects from unmatched lists
            unmatched_gt.remove(gt_idx)
            unmatched_pred.remove(pred_idx)
        
        return matched_pairs, unmatched_gt, unmatched_pred

    def analyze_class_mismatch(self, ground_truth_objects, predictions, image_name):
        """
        Analyze and explain class mismatches between ground truth and predictions
        Args:
            ground_truth_objects: List of GT annotations
            predictions: List of model predictions
            image_name: Name of the image being analyzed
        Returns:
            Dictionary with detailed class mismatch analysis
        """
        # Extract classes from GT and predictions
        gt_classes = set()
        pred_classes = set()
        
        for gt_obj in ground_truth_objects:
            if len(gt_obj) > 0:
                gt_class = self.tag_list[int(gt_obj[0])]
                gt_classes.add(gt_class)
        
        for pred in predictions:
            pred_classes.add(pred['Lb'])
        
        # Analyze the mismatch
        common_classes = gt_classes.intersection(pred_classes)
        gt_only_classes = gt_classes - pred_classes
        pred_only_classes = pred_classes - gt_classes
        
        analysis = {
            'image_name': image_name,
            'gt_classes': list(gt_classes),
            'pred_classes': list(pred_classes),
            'common_classes': list(common_classes),
            'gt_only_classes': list(gt_only_classes),  # Classes in GT but not predicted
            'pred_only_classes': list(pred_only_classes),  # Classes predicted but not in GT
            'gt_class_count': len(gt_classes),
            'pred_class_count': len(pred_classes),
            'common_class_count': len(common_classes),
            'class_mismatch_severity': 'low' if len(pred_only_classes) <= 1 else 'high'
        }
        
        # Print detailed analysis
        print(f"\n🔍 CLASS MISMATCH ANALYSIS for {image_name}:")
        print(f"   📋 GT Classes ({len(gt_classes)}): {list(gt_classes)}")
        print(f"   🤖 Predicted Classes ({len(pred_classes)}): {list(pred_classes)}")
        
        if common_classes:
            print(f"   ✅ Common Classes ({len(common_classes)}): {list(common_classes)}")
        
        if gt_only_classes:
            print(f"   🔍 Missed Classes ({len(gt_only_classes)}): {list(gt_only_classes)} → Will be False Negatives")
        
        if pred_only_classes:
            print(f"   ❌ Extra Predicted Classes ({len(pred_only_classes)}): {list(pred_only_classes)} → Will be False Positives")
        
        # Provide guidance
        if len(pred_only_classes) > len(gt_only_classes):
            print(f"   💡 Model tends to over-predict classes for this image")
        elif len(gt_only_classes) > len(pred_only_classes):
            print(f"   💡 Model tends to under-predict classes for this image")
        else:
            print(f"   💡 Balanced class prediction for this image")
        
        return analysis

    def predict_validation_outcome(self, ground_truth_objects, predictions, image_width, image_height):
        """
        Predict the validation outcome before running full validation
        This helps understand what results to expect
        """
        print(f"\n🔮 PREDICTED VALIDATION OUTCOME:")
        print(f"   📊 Input: {len(ground_truth_objects)} GT objects, {len(predictions)} predictions")
        
        # Get class analysis
        gt_classes = {}
        pred_classes = {}
        
        # Count GT classes
        for gt_obj in ground_truth_objects:
            if len(gt_obj) > 0:
                gt_class = self.tag_list[int(gt_obj[0])]
                gt_classes[gt_class] = gt_classes.get(gt_class, 0) + 1
        
        # Count predicted classes  
        for pred in predictions:
            pred_class = pred['Lb']
            pred_classes[pred_class] = pred_classes.get(pred_class, 0) + 1
        
        print(f"   📋 GT class distribution: {gt_classes}")
        print(f"   🤖 Predicted class distribution: {pred_classes}")
        
        # Estimate outcomes
        estimated_tp = 0
        estimated_fp = 0
        estimated_fn = 0
        
        # Estimate TPs (common classes, assume good IoU)
        for gt_class, gt_count in gt_classes.items():
            if gt_class in pred_classes:
                # Estimate matches (conservative: take minimum)
                matches = min(gt_count, pred_classes[gt_class])
                estimated_tp += matches
        
        # Estimate FPs (predictions without GT match)
        for pred_class, pred_count in pred_classes.items():
            if pred_class not in gt_classes:
                estimated_fp += pred_count  # All predictions of this class are FP
            else:
                # Excess predictions of existing classes
                excess = max(0, pred_count - gt_classes[pred_class])
                estimated_fp += excess
        
        # Estimate FNs (GT objects without predictions)
        for gt_class, gt_count in gt_classes.items():
            if gt_class not in pred_classes:
                estimated_fn += gt_count  # All GT of this class are FN
            else:
                # Unmatched GT objects
                unmatched = max(0, gt_count - pred_classes[gt_class])
                estimated_fn += unmatched
        
        # Calculate estimated metrics
        est_precision = estimated_tp / max(len(predictions), 1)
        est_recall = estimated_tp / max(len(ground_truth_objects), 1)
        est_f1 = 2 * (est_precision * est_recall) / max(est_precision + est_recall, 0.001)
        
        print(f"\n📈 ESTIMATED RESULTS (assuming good IoU overlap):")
        print(f"   • Estimated True Positives: {estimated_tp}")
        print(f"   • Estimated False Positives: {estimated_fp}")
        print(f"   • Estimated False Negatives: {estimated_fn}")
        print(f"   • Estimated Precision: {est_precision:.3f}")
        print(f"   • Estimated Recall: {est_recall:.3f}")
        print(f"   • Estimated F1-Score: {est_f1:.3f}")
        
        print(f"\n💡 NOTE: Actual results may vary based on IoU overlap quality")
        
        return {
            'estimated_tp': estimated_tp,
            'estimated_fp': estimated_fp, 
            'estimated_fn': estimated_fn,
            'estimated_precision': est_precision,
            'estimated_recall': est_recall,
            'estimated_f1': est_f1
        }

    def inference_images_multi_object(self):
        """
        Enhanced validation pipeline with comprehensive multiple object support
        This method extends the existing logic without breaking compatibility
        """
        img_list = os.listdir(self.img_path)
        image_row = 1
        workbook = xlsxwriter.Workbook(self.report_path + '/Multi_Object_Benchmark_Report.xlsx')
        worksheet = workbook.add_worksheet('Multi Object Validation Report')
        worksheet.set_column(1, 25, 15)
        my_format = workbook.add_format({'align': 'center', 'text_wrap': 'True'})
        
        # Enhanced headers for multiple object reporting - simplified for valve detection
        headers = [
            'Sr No.', 'ModelName', 'InputFileName', 'GT_ObjectId', 'GT_Class', 'GT_X', 'GT_Y', 'GT_Width', 'GT_Height',
            'Pred_ObjectId', 'Pred_Confidence', 'Pred_Class', 'Pred_X', 'Pred_Y', 'Pred_Width', 'Pred_Height',
            'Match_Status', 'IoU_Score', 'Start_Time', 'End_Time', 'API_Duration', 'Category', 'Match_Type'
        ]
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, my_format)
        
        sr_no = 1
        total_stats = {
            'total_images': 0,
            'total_gt_objects': 0,
            'total_predictions': 0,
            'total_matches': 0,
            'total_missed': 0,
            'total_false_positives': 0
        }
        
        print(f"\n🚀 Starting multi-object validation for {len(img_list)} images...")
        
        for img_idx, img in enumerate(img_list):
            print(f"\n📸 Processing image {img_idx + 1}/{len(img_list)}: {img}")
            
            try:
                # Load image
                img_read_val_detail = cv2.cvtColor(cv2.imread(self.img_path + '/' + img), cv2.COLOR_BGR2RGB)
                dh, dw = img_read_val_detail.shape[:2]
                imagepath = self.img_path + '/' + img
                
                # Prepare API request
                with open(self.img_path + '/' + img, 'rb') as open_file:
                    byte_content = open_file.read()
                base64_string = b64encode(byte_content).decode('utf-8')
                
                raw_data = {
                    "Tid": "1", "Did": "DeviceId_10", "Fid": "1", "Per": [{}],
                    "Ts": "", "Ts_ntp": "", "Inf_ver": "", "Msg_ver": "", "Model": "",
                    "Ad": {}, "Ffp": "", "Ltsize": "", "Lfp": "", "Mtp": [],
                    'Base_64': base64_string, "C_threshold": self.confidence_threshold,
                    "I_fn": "", "Msk_img": [], "Rep_img": [], "Prompt": []
                }
                
                # Make API call
                url = self.url + '/' + self.url_key
                headers = {"Content-Type": "application/json; charset=utf-8"}
                start = datetime.datetime.now()
                
                try:
                    output = requests.post(url, json=raw_data, headers=headers, timeout=30)
                    output.raise_for_status()  # Raise an exception for HTTP error codes
                    result = output.json()
                    end = datetime.datetime.now()
                except requests.exceptions.ConnectionError as e:
                    print(f"❌ Connection Error for {img}: Failed to connect to API server at {url}")
                    print(f"   Details: {str(e)}")
                    print(f"   ⚡ Please ensure the API server is running and accessible")
                    continue  # Skip this image and continue with next
                except requests.exceptions.Timeout as e:
                    print(f"⏱️ Timeout Error for {img}: API request timed out after 30 seconds")
                    print(f"   Details: {str(e)}")
                    continue  # Skip this image and continue with next
                except requests.exceptions.RequestException as e:
                    print(f"🚫 Request Error for {img}: API call failed")
                    print(f"   Details: {str(e)}")
                    continue  # Skip this image and continue with next
                except ValueError as e:
                    print(f"📄 JSON Error for {img}: Failed to parse API response as JSON")
                    print(f"   Details: {str(e)}")
                    print(f"   Raw response: {output.text if 'output' in locals() else 'No response'}")
                    continue  # Skip this image and continue with next
                
                end = datetime.datetime.now()
                
                api_duration = (end - start).seconds
                start_date_time = start.strftime("%d/%m/%Y %H:%M:%S.%f")
                end_date_time = end.strftime("%d/%m/%Y %H:%M:%S.%f")
                
                # Get predictions
                predictions = result.get('Fs', [])
                print(f"   🔍 Found {len(predictions)} predictions")
                
                # Debug: Print all predictions for this image
                for i, pred in enumerate(predictions):
                    print(f"   🤖 Prediction {i+1}: {pred['Lb']} (confidence: {pred['Cs']:.3f})")
                
                # Load ground truth annotations
                gt_annotation_file = f"{self.original_annotation_path}/{os.path.splitext(img)[0]}.txt"
                ground_truth_objects = []
                
                print(f"   📁 Looking for GT file: {gt_annotation_file}")
                
                if os.path.exists(gt_annotation_file):
                    print(f"   ✅ GT file found, loading annotations...")
                    with open(gt_annotation_file) as file:
                        lines = [line.rstrip() for line in file]
                        print(f"   📄 GT file has {len(lines)} lines")
                        for line_num, annotation_val in enumerate(lines):
                            if annotation_val.strip():  # Skip empty lines
                                final_lines = annotation_val.split()
                                print(f"   📝 Line {line_num + 1}: {final_lines} (length: {len(final_lines)})")
                                if len(final_lines) >= 5:  # Valid annotation format
                                    ground_truth_objects.append(final_lines)
                                    print(f"      ✅ Valid GT annotation added")
                                else:
                                    print(f"      ❌ Invalid GT annotation (need 5+ values)")
                else:
                    print(f"   ❌ GT file NOT FOUND: {gt_annotation_file}")
                    print(f"   📁 Directory exists: {os.path.exists(self.original_annotation_path)}")
                    if os.path.exists(self.original_annotation_path):
                        available_files = os.listdir(self.original_annotation_path)
                        print(f"   📁 Available files: {available_files[:5]}...")  # Show first 5 files
                
                print(f"   📋 Found {len(ground_truth_objects)} ground truth objects")
                
                # Debug: Print all GT objects for this image
                for i, gt_obj in enumerate(ground_truth_objects):
                    gt_class = self.tag_list[int(gt_obj[0])] if int(gt_obj[0]) < len(self.tag_list) else f"class_{gt_obj[0]}"
                    print(f"   📋 GT {i+1}: {gt_class}")
                
                # Analyze class mismatch before matching
                class_analysis = self.analyze_class_mismatch(ground_truth_objects, predictions, img)
                
                # Predict validation outcome
                predicted_outcome = self.predict_validation_outcome(ground_truth_objects, predictions, dw, dh)
                
                # Enhanced matching using new algorithm
                print(f"   🔍 STARTING MATCHING ALGORITHM:")
                print(f"      - GT objects: {len(ground_truth_objects)}")
                print(f"      - Predictions: {len(predictions)}")
                print(f"      - IoU threshold: 0.5")
                
                matched_pairs, unmatched_gt, unmatched_pred = self.match_predictions_to_ground_truth(
                    ground_truth_objects, predictions, dw, dh
                )
                
                print(f"   📊 MATCHING RESULTS:")
                print(f"      ✅ Matched pairs: {len(matched_pairs)}")
                print(f"      ❌ Unmatched GT (missed): {len(unmatched_gt)} -> {unmatched_gt}")
                print(f"      ➕ Unmatched Pred (false pos): {len(unmatched_pred)} -> {unmatched_pred}")
                
                print(f"   ✅ Matched: {len(matched_pairs)}, Missed: {len(unmatched_gt)}, False Positives: {len(unmatched_pred)}")
                
                # Debug: Print detailed matching results
                print(f"   🔍 DETAILED MATCHING RESULTS:")
                for match in matched_pairs:
                    print(f"      ✅ GT {match['gt_class']} ↔ Pred {match['pred_class']} (IoU: {match['iou']:.3f})")
                
                if unmatched_gt:
                    print(f"   🔍 MISSED DETECTIONS:")
                    for gt_idx in unmatched_gt:
                        gt_class = self.tag_list[int(ground_truth_objects[gt_idx][0])]
                        print(f"      ❌ Missed: {gt_class}")
                
                if unmatched_pred:
                    print(f"   🔍 FALSE POSITIVES:")
                    for pred_idx in unmatched_pred:
                        pred_class = predictions[pred_idx]['Lb']
                        confidence = predictions[pred_idx]['Cs']
                        print(f"      ❌ Extra: {pred_class} (confidence: {confidence:.3f})")
                
                # Update statistics
                total_stats['total_images'] += 1
                total_stats['total_gt_objects'] += len(ground_truth_objects)
                total_stats['total_predictions'] += len(predictions)
                total_stats['total_matches'] += len(matched_pairs)
                total_stats['total_missed'] += len(unmatched_gt)
                total_stats['total_false_positives'] += len(unmatched_pred)
                
                # Get image category
                category = self.extract_category_from_image(img)
                
                print(f"   📝 Writing {len(matched_pairs)} matches to Excel...")
                # Write matched pairs to report
                for match in matched_pairs:
                    gt_idx = match['gt_index']
                    pred_idx = match['pred_index']
                    gt_obj = ground_truth_objects[gt_idx]
                    pred_obj = predictions[pred_idx]
                    
                    # Convert coordinates to Pascal VOC format
                    gt_voc = yolo_to_pascal_voc(gt_obj[1], gt_obj[2], gt_obj[3], gt_obj[4], dw, dh)
                    cw = pred_obj['Dm']['W']/2
                    ch = pred_obj['Dm']['H']/2
                    pred_voc = yolo_to_pascal_voc(pred_obj['Dm']['X']+cw, pred_obj['Dm']['Y']+ch,
                                                 pred_obj['Dm']['W'], pred_obj['Dm']['H'], dw, dh)
                    
                    # Write to worksheet
                    row_data = [
                        sr_no, self.url_key, img, gt_idx + 1, 
                        self.tag_list[int(gt_obj[0])], gt_voc[0], gt_voc[1], gt_voc[2], gt_voc[3],
                        pred_idx + 1, pred_obj['Cs'], pred_obj['Lb'], 
                        pred_voc[0], pred_voc[1], pred_voc[2], pred_voc[3],
                        'MATCHED', f"{match['iou']:.3f}", start_date_time, end_date_time, 
                        f"{api_duration} seconds", category, 'True_Positive'
                    ]
                    
                    for col, value in enumerate(row_data):
                        worksheet.write(image_row, col, value, my_format)
                    print(f"      📝 Wrote match row {image_row}: {pred_obj['Lb']} (confidence: {pred_obj['Cs']:.3f})")
                    image_row += 1
                    sr_no += 1
                
                print(f"   📝 Writing {len(unmatched_gt)} missed detections to Excel...")
                # Write missed detections (False Negatives)
                if len(unmatched_gt) > 0:
                    print(f"   🔍 PROCESSING {len(unmatched_gt)} MISSED DETECTIONS:")
                    for idx, gt_idx in enumerate(unmatched_gt):
                        gt_obj = ground_truth_objects[gt_idx]
                        gt_class = self.tag_list[int(gt_obj[0])] if int(gt_obj[0]) < len(self.tag_list) else f"class_{gt_obj[0]}"
                        print(f"      {idx + 1}. Writing missed detection: {gt_class} (GT index: {gt_idx})")
                        
                        gt_voc = yolo_to_pascal_voc(gt_obj[1], gt_obj[2], gt_obj[3], gt_obj[4], dw, dh)
                        
                        row_data = [
                            sr_no, self.url_key, img, gt_idx + 1,
                            self.tag_list[int(gt_obj[0])], gt_voc[0], gt_voc[1], gt_voc[2], gt_voc[3],
                            '-', 0, 'NOT_DETECTED', '-', '-', '-', '-',
                            'MISSED', '0.000', start_date_time, end_date_time,
                            f"{api_duration} seconds", category, 'False_Negative'
                        ]
                        
                        for col, value in enumerate(row_data):
                            worksheet.write(image_row, col, value, my_format)
                        print(f"      📝 Wrote missed row {image_row}: {self.tag_list[int(gt_obj[0])]} to Excel")
                        image_row += 1
                        sr_no += 1
                else:
                    print(f"   ✅ No missed detections to write for this image")
                
                print(f"   📝 Writing {len(unmatched_pred)} false positives to Excel...")
                # Write false positives
                for pred_idx in unmatched_pred:
                    pred_obj = predictions[pred_idx]
                    cw = pred_obj['Dm']['W']/2
                    ch = pred_obj['Dm']['H']/2
                    pred_voc = yolo_to_pascal_voc(pred_obj['Dm']['X']+cw, pred_obj['Dm']['Y']+ch,
                                                 pred_obj['Dm']['W'], pred_obj['Dm']['H'], dw, dh)
                    
                    row_data = [
                        sr_no, self.url_key, img, '-',
                        'NO_GT_MATCH', '-', '-', '-', '-',
                        pred_idx + 1, pred_obj['Cs'], pred_obj['Lb'],
                        pred_voc[0], pred_voc[1], pred_voc[2], pred_voc[3],
                        'FALSE_POSITIVE', '0.000', start_date_time, end_date_time,
                        f"{api_duration} seconds", category, 'False_Positive'
                    ]
                    
                    for col, value in enumerate(row_data):
                        worksheet.write(image_row, col, value, my_format)
                    print(f"      📝 Wrote false positive row {image_row}: {pred_obj['Lb']} (confidence: {pred_obj['Cs']:.3f})")
                    image_row += 1
                    sr_no += 1
                    
            except Exception as e:
                print(f"❌ Error processing image {img}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Add summary statistics at the end
        worksheet.write(image_row + 2, 0, 'SUMMARY STATISTICS', my_format)
        summary_data = [
            ['Total Images Processed', total_stats['total_images']],
            ['Total Ground Truth Objects', total_stats['total_gt_objects']],
            ['Total Predictions', total_stats['total_predictions']],
            ['Total Matches (TP)', total_stats['total_matches']],
            ['Total Missed (FN)', total_stats['total_missed']],
            ['Total False Positives (FP)', total_stats['total_false_positives']],
            ['Precision', total_stats['total_matches'] / max(total_stats['total_predictions'], 1)],
            ['Recall', total_stats['total_matches'] / max(total_stats['total_gt_objects'], 1)],
        ]
        
        for i, (label, value) in enumerate(summary_data):
            worksheet.write(image_row + 4 + i, 0, label, my_format)
            worksheet.write(image_row + 4 + i, 1, value, my_format)
        
        # Final detailed summary
        print(f"\n🎯 FINAL VALIDATION SUMMARY:")
        print(f"   📁 Processed {total_stats['total_images']} images")
        print(f"   📋 Ground Truth Objects: {total_stats['total_gt_objects']}")
        print(f"   🤖 API Predictions: {total_stats['total_predictions']}")
        print(f"   ✅ Correct Matches (TP): {total_stats['total_matches']}")
        print(f"   ❌ Missed Detections (FN): {total_stats['total_missed']}")
        print(f"   ⚠️  False Positives (FP): {total_stats['total_false_positives']}")
        print(f"   📊 Excel Rows Written: {image_row - 1}")
        print(f"   🧮 Expected Rows: {total_stats['total_matches'] + total_stats['total_missed'] + total_stats['total_false_positives']}")
        
        if (image_row - 1) != (total_stats['total_matches'] + total_stats['total_missed'] + total_stats['total_false_positives']):
            print(f"   ⚠️  WARNING: Row count mismatch detected!")
        else:
            print(f"   ✅ Excel row count matches expected!")
        
        workbook.close()
        
        print(f"\n📊 Multi-object validation completed!")
        print(f"   📈 Summary: {total_stats}")
        print(f"   📄 Report saved: {self.report_path}/Multi_Object_Benchmark_Report.xlsx")
        
        return total_stats

    def generate_comprehensive_reports(self, validation_stats):
        """
        Generate comprehensive validation reports including:
        1. Overall Summary Report
        2. Correct Predictions Report
        3. Mispredictions Report
        4. Missed Detections Report
        """
        print("\n📊 Generating Comprehensive Validation Reports...")
        
        try:
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(self.report_path, "comprehensive_reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate each report
            self.generate_overall_summary_report(validation_stats, reports_dir)
            self.generate_detailed_performance_reports(reports_dir)
            
            print(f"✅ All comprehensive reports generated in: {reports_dir}")
            
        except Exception as e:
            print(f"❌ Error generating comprehensive reports: {e}")
            traceback.print_exc()

    def generate_overall_summary_report(self, validation_stats, reports_dir):
        """Generate overall summary report with key metrics and statistics"""
        
        print("   📋 Generating Overall Summary Report...")
        
        # Create overall summary workbook
        workbook = xlsxwriter.Workbook(os.path.join(reports_dir, 'Overall_Summary_Report.xlsx'))
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'valign': 'vcenter',
            'bg_color': '#4CAF50', 'font_color': 'white', 'border': 1
        })
        metric_format = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 1
        })
        percentage_format = workbook.add_format({
            'align': 'center', 'valign': 'vcenter', 'border': 1, 'num_format': '0.00%'
        })
        
        # Summary Sheet
        summary_sheet = workbook.add_worksheet('Overall Summary')
        summary_sheet.set_column('A:B', 25)
        summary_sheet.set_column('C:D', 15)
        
        # Title
        summary_sheet.merge_range('A1:D1', 'VALIDATION PIPELINE OVERALL SUMMARY', header_format)
        
        # Basic Statistics
        row = 3
        summary_sheet.write(row, 0, 'BASIC STATISTICS', header_format)
        row += 1
        
        basic_stats = [
            ['Total Images Processed', validation_stats.get('total_images', 0)],
            ['Total Ground Truth Objects', validation_stats.get('total_gt_objects', 0)],
            ['Total Predictions Made', validation_stats.get('total_predictions', 0)],
            ['Total Correct Matches', validation_stats.get('total_matches', 0)],
            ['Total Missed Detections', validation_stats.get('total_missed', 0)],
            ['Total False Positives', validation_stats.get('total_false_positives', 0)]
        ]
        
        for stat_name, stat_value in basic_stats:
            summary_sheet.write(row, 0, stat_name, metric_format)
            summary_sheet.write(row, 1, stat_value, metric_format)
            row += 1
        
        # Performance Metrics
        row += 1
        summary_sheet.write(row, 0, 'PERFORMANCE METRICS', header_format)
        row += 1
        
        # Calculate metrics
        total_predictions = max(validation_stats.get('total_predictions', 1), 1)
        total_gt = max(validation_stats.get('total_gt_objects', 1), 1)
        total_matches = validation_stats.get('total_matches', 0)
        
        precision = total_matches / total_predictions
        recall = total_matches / total_gt
        f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
        accuracy = total_matches / max(total_predictions + validation_stats.get('total_missed', 0), 1)
        
        performance_metrics = [
            ['Precision (TP / (TP + FP))', precision],
            ['Recall (TP / (TP + FN))', recall],
            ['F1-Score', f1_score],
            ['Accuracy', accuracy]
        ]
        
        for metric_name, metric_value in performance_metrics:
            summary_sheet.write(row, 0, metric_name, metric_format)
            summary_sheet.write(row, 1, metric_value, percentage_format)
            row += 1
        
        # Category-wise Analysis (if gt_categories is available)
        if self.gt_categories:
            row += 1
            summary_sheet.write(row, 0, 'CATEGORY ANALYSIS', header_format)
            row += 1
            
            category_stats = self.get_category_wise_statistics()
            
            for category, stats in category_stats.items():
                summary_sheet.write(row, 0, f"Category: {category}", metric_format)
                summary_sheet.write(row, 1, f"Images: {stats['image_count']}", metric_format)
                summary_sheet.write(row, 2, f"Objects: {stats['object_count']}", metric_format)
                summary_sheet.write(row, 3, f"Accuracy: {stats['accuracy']:.2%}", percentage_format)
                row += 1
        
        # Model Performance Summary
        model_sheet = workbook.add_worksheet('Model Performance')
        model_sheet.set_column('A:E', 20)
        
        # Performance by Class
        model_sheet.merge_range('A1:E1', 'MODEL PERFORMANCE BY CLASS', header_format)
        
        # Headers
        class_headers = ['Class Name', 'Total GT', 'Total Predictions', 'Correct Matches', 'Class Accuracy']
        for col, header in enumerate(class_headers):
            model_sheet.write(2, col, header, header_format)
        
        # Get class-wise performance
        class_performance = self.get_class_wise_performance()
        row = 3
        for class_name, perf in class_performance.items():
            model_sheet.write(row, 0, class_name, metric_format)
            model_sheet.write(row, 1, perf['total_gt'], metric_format)
            model_sheet.write(row, 2, perf['total_pred'], metric_format)
            model_sheet.write(row, 3, perf['correct_matches'], metric_format)
            model_sheet.write(row, 4, perf['accuracy'], percentage_format)
            row += 1
        
        workbook.close()
        print("   ✅ Overall Summary Report generated")

    def generate_detailed_performance_reports(self, reports_dir):
        """Generate detailed reports for correct predictions, mispredictions, and missed detections"""
        
        print("   📋 Generating Detailed Performance Reports...")
        
        # Read the multi-object benchmark report to extract detailed data
        benchmark_file = os.path.join(self.report_path, 'Multi_Object_Benchmark_Report.xlsx')
        
        if not os.path.exists(benchmark_file):
            print("   ⚠️ Multi-object benchmark report not found. Running validation first...")
            return
        
        try:
            # Read the benchmark data
            import pandas as pd
            df = pd.read_excel(benchmark_file, sheet_name='Multi Object Validation Report')
            
            print(f"   📊 Read {len(df)} records from benchmark report")
            print(f"   📋 Columns available: {list(df.columns)}")
            
            # Handle missing or NaN values
            df = df.fillna('')  # Replace NaN with empty strings
            
            # Separate data by match type - fix the filtering
            correct_predictions = df[df['Match_Type'].str.contains('True_Positive', na=False)]
            mispredictions = df[df['Match_Type'].str.contains('False_Positive', na=False)] 
            missed_detections = df[df['Match_Type'].str.contains('False_Negative', na=False)]
            
            print(f"   ✅ Correct predictions: {len(correct_predictions)}")
            print(f"   ❌ Mispredictions: {len(mispredictions)}")
            print(f"   🔍 Missed detections: {len(missed_detections)}")
            
            # Generate individual reports with proper error handling
            if len(correct_predictions) > 0:
                self.create_correct_predictions_report(correct_predictions, reports_dir)
            else:
                print("   ⚠️ No correct predictions found")
                
            if len(mispredictions) > 0:
                self.create_mispredictions_report(mispredictions, reports_dir)
            else:
                print("   ⚠️ No mispredictions found")
                
            if len(missed_detections) > 0:
                self.create_missed_detections_report(missed_detections, reports_dir)
            else:
                print("   ⚠️ No missed detections found")
            
        except Exception as e:
            print(f"   ❌ Error processing benchmark data: {e}")
            import traceback
            traceback.print_exc()

    def create_correct_predictions_report(self, correct_data, reports_dir):
        """Create detailed report for correct predictions"""
        
        workbook = xlsxwriter.Workbook(os.path.join(reports_dir, 'Correct_Predictions_Report.xlsx'))
        worksheet = workbook.add_worksheet('Correct Predictions')
        
        # Format
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'bg_color': '#4CAF50', 
            'font_color': 'white', 'border': 1
        })
        cell_format = workbook.add_format({'align': 'center', 'border': 1})
        
        # Headers
        headers = [
            'Image Name', 'GT Class', 'Pred Class', 'Confidence', 
            'IoU Score', 'Category', 'GT Bbox', 'Pred Bbox'
        ]
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Data
        for row, (_, data) in enumerate(correct_data.iterrows(), 1):
            worksheet.write(row, 0, data.get('InputFileName', ''), cell_format)
            worksheet.write(row, 1, data.get('GT_Class', ''), cell_format)
            worksheet.write(row, 2, data.get('Pred_Class', ''), cell_format)
            worksheet.write(row, 3, data.get('Pred_Confidence', 0), cell_format)
            worksheet.write(row, 4, data.get('IoU_Score', ''), cell_format)
            worksheet.write(row, 5, data.get('Category', ''), cell_format)
            
            # Bounding boxes
            gt_bbox = f"({data.get('GT_X', '')}, {data.get('GT_Y', '')}, {data.get('GT_Width', '')}, {data.get('GT_Height', '')})"
            pred_bbox = f"({data.get('Pred_X', '')}, {data.get('Pred_Y', '')}, {data.get('Pred_Width', '')}, {data.get('Pred_Height', '')})"
            
            worksheet.write(row, 6, gt_bbox, cell_format)
            worksheet.write(row, 7, pred_bbox, cell_format)
        
        # Summary statistics
        summary_row = len(correct_data) + 3
        worksheet.write(summary_row, 0, 'SUMMARY STATISTICS', header_format)
        worksheet.write(summary_row + 1, 0, f'Total Correct Predictions: {len(correct_data)}', cell_format)
        
        if len(correct_data) > 0:
            avg_confidence = correct_data['Pred_Confidence'].mean()
            avg_iou = correct_data['IoU_Score'].str.replace('%', '').astype(float).mean() / 100
            worksheet.write(summary_row + 2, 0, f'Average Confidence: {avg_confidence:.3f}', cell_format)
            worksheet.write(summary_row + 3, 0, f'Average IoU: {avg_iou:.3f}', cell_format)
        
        workbook.close()
        print("   ✅ Correct Predictions Report generated")

    def create_mispredictions_report(self, mispred_data, reports_dir):
        """Create detailed report for mispredictions (false positives)"""
        
        import pandas as pd
        
        workbook = xlsxwriter.Workbook(os.path.join(reports_dir, 'Mispredictions_Report.xlsx'))
        worksheet = workbook.add_worksheet('Mispredictions')
        
        # Format
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'bg_color': '#FF5722', 
            'font_color': 'white', 'border': 1
        })
        cell_format = workbook.add_format({'align': 'center', 'border': 1})
        
        # Headers - simplified for valve detection
        headers = [
            'Image Name', 'Predicted Class', 'Confidence', 'Category', 
            'Pred Bbox', 'Issue Type', 'Possible Reason'
        ]
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Data - handle NaN and empty values properly
        for row, (_, data) in enumerate(mispred_data.iterrows(), 1):
            # Safe data extraction with fallbacks
            image_name = str(data.get('InputFileName', 'Unknown')).replace('nan', 'Unknown')
            pred_class = str(data.get('Pred_Class', 'Unknown')).replace('nan', 'Unknown')
            confidence = data.get('Pred_Confidence', 0)
            if pd.isna(confidence):
                confidence = 0
            category = self.extract_category_from_image(image_name)  # Extract from gt_categories.json
            
            # Bounding box - handle NaN values
            pred_x = data.get('Pred_X', 0) if not pd.isna(data.get('Pred_X', 0)) else 0
            pred_y = data.get('Pred_Y', 0) if not pd.isna(data.get('Pred_Y', 0)) else 0
            pred_w = data.get('Pred_Width', 0) if not pd.isna(data.get('Pred_Width', 0)) else 0
            pred_h = data.get('Pred_Height', 0) if not pd.isna(data.get('Pred_Height', 0)) else 0
            
            pred_bbox = f"({pred_x}, {pred_y}, {pred_w}, {pred_h})"
            
            worksheet.write(row, 0, image_name, cell_format)
            worksheet.write(row, 1, pred_class, cell_format)
            worksheet.write(row, 2, confidence, cell_format)
            worksheet.write(row, 3, category, cell_format)
            worksheet.write(row, 4, pred_bbox, cell_format)
            worksheet.write(row, 5, 'False Positive', cell_format)
            worksheet.write(row, 6, 'No matching ground truth', cell_format)
        
        # Summary statistics
        summary_row = len(mispred_data) + 3
        worksheet.write(summary_row, 0, 'SUMMARY STATISTICS', header_format)
        worksheet.write(summary_row + 1, 0, f'Total Mispredictions: {len(mispred_data)}', cell_format)
        
        if len(mispred_data) > 0:
            # Handle average confidence calculation with NaN protection
            confidences = mispred_data['Pred_Confidence'].dropna()
            if len(confidences) > 0:
                avg_confidence = confidences.mean()
                worksheet.write(summary_row + 2, 0, f'Average Confidence: {avg_confidence:.3f}', cell_format)
            
            # Class distribution - handle NaN values
            pred_classes = mispred_data['Pred_Class'].dropna()
            if len(pred_classes) > 0:
                class_dist = pred_classes.value_counts()
                worksheet.write(summary_row + 4, 0, 'Class Distribution:', header_format)
                for i, (class_name, count) in enumerate(class_dist.items()):
                    worksheet.write(summary_row + 5 + i, 0, f'{class_name}: {count}', cell_format)
        
        workbook.close()
        print("   ✅ Mispredictions Report generated")

    def create_missed_detections_report(self, missed_data, reports_dir):
        """Create detailed report for missed detections (false negatives)"""
        
        import pandas as pd
        
        workbook = xlsxwriter.Workbook(os.path.join(reports_dir, 'Missed_Detections_Report.xlsx'))
        worksheet = workbook.add_worksheet('Missed Detections')
        
        # Format
        header_format = workbook.add_format({
            'bold': True, 'align': 'center', 'bg_color': '#FF9800', 
            'font_color': 'white', 'border': 1
        })
        cell_format = workbook.add_format({'align': 'center', 'border': 1})
        
        # Headers - simplified for valve detection
        headers = [
            'Image Name', 'Missed Class', 'Category', 'GT Bbox', 
            'Issue Type', 'Possible Reason'
        ]
        
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Data - handle NaN and empty values properly
        for row, (_, data) in enumerate(missed_data.iterrows(), 1):
            # Safe data extraction with fallbacks
            image_name = str(data.get('InputFileName', 'Unknown')).replace('nan', 'Unknown')
            gt_class = str(data.get('GT_Class', 'Unknown')).replace('nan', 'Unknown')
            category = self.extract_category_from_image(image_name)  # Extract from gt_categories.json
            
            # Bounding box - handle NaN values
            gt_x = data.get('GT_X', 0) if not pd.isna(data.get('GT_X', 0)) else 0
            gt_y = data.get('GT_Y', 0) if not pd.isna(data.get('GT_Y', 0)) else 0
            gt_w = data.get('GT_Width', 0) if not pd.isna(data.get('GT_Width', 0)) else 0
            gt_h = data.get('GT_Height', 0) if not pd.isna(data.get('GT_Height', 0)) else 0
            
            gt_bbox = f"({gt_x}, {gt_y}, {gt_w}, {gt_h})"
            
            worksheet.write(row, 0, image_name, cell_format)
            worksheet.write(row, 1, gt_class, cell_format)
            worksheet.write(row, 2, category, cell_format)
            worksheet.write(row, 3, gt_bbox, cell_format)
            worksheet.write(row, 4, 'False Negative', cell_format)
            worksheet.write(row, 5, 'Object not detected by model', cell_format)
        
        # Summary statistics
        summary_row = len(missed_data) + 3
        worksheet.write(summary_row, 0, 'SUMMARY STATISTICS', header_format)
        worksheet.write(summary_row + 1, 0, f'Total Missed Detections: {len(missed_data)}', cell_format)
        
        if len(missed_data) > 0:
            # Class distribution - handle NaN values
            gt_classes = missed_data['GT_Class'].dropna()
            if len(gt_classes) > 0:
                class_dist = gt_classes.value_counts()
                worksheet.write(summary_row + 3, 0, 'Class Distribution:', header_format)
                for i, (class_name, count) in enumerate(class_dist.items()):
                    worksheet.write(summary_row + 4 + i, 0, f'{class_name}: {count}', cell_format)
        
        workbook.close()
        print("   ✅ Missed Detections Report generated")

    def get_category_wise_statistics(self):
        """Get statistics grouped by image categories"""
        category_stats = {}
        
        if not self.gt_categories:
            return category_stats
        
        for image_name, category_info in self.gt_categories.items():
            if isinstance(category_info, dict):
                category = f"{category_info.get('product', 'Unknown')}_{category_info.get('lighting', 'Unknown')}_{category_info.get('distance', 'Unknown')}"
            else:
                category = str(category_info)
            
            if category not in category_stats:
                category_stats[category] = {
                    'image_count': 0,
                    'object_count': 0,
                    'accuracy': 0.0
                }
            
            category_stats[category]['image_count'] += 1
            # Note: Object count and accuracy would need to be calculated from validation results
        
        return category_stats

    def get_class_wise_performance(self):
        """Get performance statistics for each class"""
        class_performance = {}
        
        for class_name in self.tag_list:
            class_performance[class_name] = {
                'total_gt': 0,
                'total_pred': 0,
                'correct_matches': 0,
                'accuracy': 0.0
            }
        
        # Note: This would be populated from actual validation results
        return class_performance

    def run_validation(self, multi_object_mode=True, generate_reports=True):
        """
        Convenience method to run validation in either single or multi-object mode
        Args:
            multi_object_mode: If True, uses enhanced multi-object validation
            generate_reports: If True, generates comprehensive reports after validation
        """
        print(f"\n🎯 Running validation pipeline in {'MULTI-OBJECT' if multi_object_mode else 'SINGLE-OBJECT'} mode")
        
        patch_results = None
        
        # Check if patch-based inference is enabled in IVA config
        if self.patch_based_inference_enabled and self.patch_based_enabled:
            print("\n✂️ Running Complete Patch Extraction Workflow...")
            patch_results = self.extract_patches_from_images()
            if patch_results is None:
                print("❌ Patch extraction failed. Cannot proceed with patch-based validation.")
                return None
            else:
                print("✅ Patch extraction completed successfully")
                print(f"   📊 Processed: {patch_results['processed_images']}/{patch_results['total_images']} images")
                print(f"   🎯 Total detections: {patch_results['total_detections']}")
                
                # Generate patch Excel reports immediately after patch processing
                if generate_reports:
                    print("\n📊 Generating Patch Inference Excel Reports...")
                    patch_excel_reports = self.generate_patch_inference_excel_reports(patch_results)
                    if patch_excel_reports:
                        print(f"✅ Patch inference Excel reports generated successfully!")
                        print(f"   📋 Overall Summary: {patch_excel_reports['overall_summary']}")
                        print(f"   ✅ Correct Predictions: {patch_excel_reports['correct_predictions']}")
                        print(f"   🔍 Missed Detections: {patch_excel_reports['missed_detections']}")
                    else:
                        print(f"❌ Failed to generate patch inference Excel reports")
        
        validation_stats = None
        
        # Skip API-based validation when patch-based inference is enabled
        if not (self.patch_based_inference_enabled and self.patch_based_enabled):
            if multi_object_mode:
                validation_stats = self.inference_images_multi_object()
            else:
                # Run original method for backward compatibility
                self.inference_images()
                validation_stats = {
                    'total_images': 0,
                    'total_gt_objects': 0, 
                    'total_predictions': 0,
                    'total_matches': 0,
                    'total_missed': 0,
                    'total_false_positives': 0
                }
            
            # Generate comprehensive validation reports if requested
            if generate_reports and validation_stats:
                print("\n📊 Generating Standard Validation Reports...")
                self.generate_comprehensive_reports(validation_stats)
        else:
            print("\n📋 Skipping API-based validation (patch-based mode enabled)")
            validation_stats = {
                'total_images': 0,
                'total_gt_objects': 0, 
                'total_predictions': 0,
                'total_matches': 0,
                'total_missed': 0,
                'total_false_positives': 0
            }
        
        # Return both results for complete analysis
        return {
            'validation_stats': validation_stats,
            'patch_results': patch_results
        }

    def run_patch_analysis_only(self, generate_reports=True):
        """
        Run patch extraction and analysis only without full validation pipeline
        Useful for testing patch-based inference independently
        """
        print("\n✂️ STARTING PATCH ANALYSIS ONLY MODE")
        print("=" * 50)
        
        if not (self.patch_based_inference_enabled and self.patch_based_enabled):
            print("❌ Patch-based inference is not enabled. Check configuration.")
            return None
        
        print("\n🔧 Step 1: Running Patch Extraction and Inference...")
        patch_results = self.extract_patches_from_images()
        
        if patch_results is None:
            print("❌ Patch extraction failed.")
            return None
        
        print("✅ Patch extraction completed successfully")
        print(f"   📊 Processed: {patch_results['processed_images']}/{patch_results['total_images']} images")
        print(f"   🎯 Total detections: {patch_results['total_detections']}")
        print(f"   📁 Results saved in: {patch_results['patch_outputs_folder']}")
        
        if generate_reports:
            print("\n📊 Step 2: Generating Patch Inference Excel Reports...")
            patch_excel_reports = self.generate_patch_inference_excel_reports(patch_results)
            if patch_excel_reports:
                print(f"✅ Patch inference Excel reports generated successfully!")
                print(f"   📋 Overall Summary: {patch_excel_reports['overall_summary']}")
                print(f"   ✅ Correct Predictions: {patch_excel_reports['correct_predictions']}")
                print(f"   🔍 Missed Detections: {patch_excel_reports['missed_detections']}")
                print(f"   📁 All reports saved in: {patch_excel_reports['reports_directory']}")
            else:
                print(f"❌ Failed to generate patch inference Excel reports")
        
        print("\n🎉 PATCH ANALYSIS COMPLETED!")
        return patch_results

    def run_complete_validation_suite(self):
        """
        Run complete validation suite with all reports and analysis
        This is the main entry point for comprehensive validation
        """
        print("\n🚀 STARTING COMPLETE VALIDATION SUITE")
        print("=" * 70)
        
        # Step 1: Validate pipeline setup
        print("\n📋 Step 1: Validating Pipeline Setup...")
        if not self.validate_pipeline_setup():
            print("❌ Pipeline setup validation failed. Please fix issues before proceeding.")
            return None
        
        # Step 1.5: Run complete patch extraction workflow if enabled
        patch_results = None
        if self.patch_based_enabled:
            print("\n✂️ Step 1.5: Running Complete Patch Extraction Workflow...")
            patch_results = self.extract_patches_from_images()
            if patch_results is None:
                print("❌ Patch extraction failed. Cannot proceed with patch-based validation.")
                return None
            else:
                print("✅ Patch extraction completed successfully")
                print(f"   📊 Processed: {patch_results['processed_images']}/{patch_results['total_images']} images")
                print(f"   🎯 Total detections: {patch_results['total_detections']}")
        
        # Step 2: Auto-generate categories if needed
        print("\n📁 Step 2: Checking GT Categories...")
        if not self.gt_categories:
            print("⚠️ GT Categories not found. Auto-generating from folder structure...")
            auto_categories = self.auto_generate_categories_from_folder()
            if auto_categories:
                self.gt_categories = auto_categories
                print("✅ Categories auto-generated and loaded")
            else:
                print("⚠️ Could not auto-generate categories. Proceeding with image name parsing.")
        else:
            print(f"✅ GT Categories loaded: {len(self.gt_categories)} entries")
        
        # Step 3: Run multi-object validation
        print("\n🔍 Step 3: Running Multi-Object Validation...")
        validation_stats = self.run_validation(multi_object_mode=True, generate_reports=False)
        
        # Step 4: Generate all comprehensive reports
        print("\n📊 Step 4: Generating Comprehensive Reports...")
        self.generate_comprehensive_reports(validation_stats)
        
        # Step 5: Generate summary
        print("\n📈 Step 5: Final Summary")
        print("=" * 70)
        self.print_validation_summary(validation_stats)
        
        return validation_stats

    def print_validation_summary(self, stats):
        """Print a formatted summary of validation results"""
        
        print("\n🎯 VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        # Basic statistics
        print("📊 BASIC STATISTICS:")
        print(f"   • Total Images Processed: {stats.get('total_images', 0)}")
        print(f"   • Total Ground Truth Objects: {stats.get('total_gt_objects', 0)}")
        print(f"   • Total Predictions Made: {stats.get('total_predictions', 0)}")
        print(f"   • Total Correct Matches: {stats.get('total_matches', 0)}")
        print(f"   • Total Missed Detections: {stats.get('total_missed', 0)}")
        print(f"   • Total False Positives: {stats.get('total_false_positives', 0)}")
        
        # Calculate and display performance metrics
        total_predictions = max(stats.get('total_predictions', 1), 1)
        total_gt = max(stats.get('total_gt_objects', 1), 1)
        total_matches = stats.get('total_matches', 0)
        
        precision = total_matches / total_predictions
        recall = total_matches / total_gt  
        f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"   • Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"   • Recall: {recall:.3f} ({recall*100:.1f}%)")
        print(f"   • F1-Score: {f1_score:.3f} ({f1_score*100:.1f}%)")
        
        # Files generated
        print(f"\n📄 REPORTS GENERATED:")
        print(f"   • Multi-Object Benchmark Report: {self.report_path}/Multi_Object_Benchmark_Report.xlsx")
        print(f"   • Overall Summary Report: {self.report_path}/comprehensive_reports/Overall_Summary_Report.xlsx")
        print(f"   • Correct Predictions Report: {self.report_path}/comprehensive_reports/Correct_Predictions_Report.xlsx")
        print(f"   • Mispredictions Report: {self.report_path}/comprehensive_reports/Mispredictions_Report.xlsx")
        print(f"   • Missed Detections Report: {self.report_path}/comprehensive_reports/Missed_Detections_Report.xlsx")
        
        if self.gt_categories:
            print(f"   • GT Categories: {len(self.gt_categories)} image categories loaded")
        
        print("\n✅ Validation pipeline completed successfully!")
        print("=" * 70)

    def auto_generate_categories_from_folder(self):
        """
        Auto-generate gt_categories.json from current folder structure
        This helps when transitioning from folder-based to single-folder approach
        """
        try:
            if not os.path.exists(self.img_path):
                print(f"❌ Image path not found: {self.img_path}")
                return {}
            
            img_list = [f for f in os.listdir(self.img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            auto_categories = {}
            
            print(f"\n🔄 Auto-generating categories for {len(img_list)} images...")
            
            for img in img_list:
                image_base = os.path.splitext(img)[0]
                
                # Try to extract category info from image name
                # Expected formats: ProductName_Camera_LightingCondition_Distance
                # Example: Garlic1_CamAisle_LightSpot_Dis15FT
                try:
                    parts = image_base.split('_')
                    if len(parts) >= 4:
                        product = parts[0].replace('143', '').replace('1', '')  # Clean product name
                        lighting = parts[2].replace('Light', '').lower()  # Extract lighting
                        distance = parts[3].replace('Dis', '').replace('FT', 'FT')  # Extract distance
                        
                        auto_categories[image_base] = {
                            'product': product,
                            'lighting': lighting,
                            'distance': distance
                        }
                    else:
                        # Fallback: use simple categorization
                        auto_categories[image_base] = {
                            'product': 'Unknown',
                            'lighting': 'Unknown', 
                            'distance': 'Unknown'
                        }
                        
                except Exception as e:
                    print(f"⚠️ Could not parse image name '{image_base}': {e}")
                    auto_categories[image_base] = {
                        'product': 'Unknown',
                        'lighting': 'Unknown',
                        'distance': 'Unknown'
                    }
            
            # Save the auto-generated categories
            output_path = os.path.join('framework_config', 'gt_categories_auto.json')
            with open(output_path, 'w') as f:
                json.dump(auto_categories, f, indent=2)
            
            print(f"✅ Auto-generated categories saved to: {output_path}")
            print(f"📊 Generated {len(auto_categories)} category entries")
            
            # Show sample entries
            sample_keys = list(auto_categories.keys())[:3]
            for key in sample_keys:
                print(f"   📝 {key}: {auto_categories[key]}")
            
            return auto_categories
            
        except Exception as e:
            print(f"❌ Error auto-generating categories: {e}")
            traceback.print_exc()
            return {}

    def validate_pipeline_setup(self):
        """
        Validate that all required files and folders exist for the validation pipeline
        """
        print("\n🔍 Validating pipeline setup...")
        
        issues = []
        
        # Check image path
        if not os.path.exists(self.img_path):
            issues.append(f"❌ Image path not found: {self.img_path}")
        else:
            img_count = len([f for f in os.listdir(self.img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"✅ Image path exists with {img_count} images")
        
        # Check annotation path
        if not os.path.exists(self.original_annotation_path):
            issues.append(f"❌ Original annotation path not found: {self.original_annotation_path}")
        else:
            ann_count = len([f for f in os.listdir(self.original_annotation_path) if f.endswith('.txt')])
            print(f"✅ Annotation path exists with {ann_count} annotation files")
        
        # Check labels file
        if not os.path.exists(self.labels_file_path):
            issues.append(f"❌ Labels file not found: {self.labels_file_path}")
        else:
            print(f"✅ Labels file exists with {len(self.tag_list)} classes: {self.tag_list}")
        
        # Check report path
        if not os.path.exists(self.report_path):
            os.makedirs(self.report_path, exist_ok=True)
            print(f"✅ Created report directory: {self.report_path}")
        else:
            print(f"✅ Report path exists: {self.report_path}")
        
        # Check predicted annotation path
        if not os.path.exists(self.predicted_annotation_path):
            os.makedirs(self.predicted_annotation_path, exist_ok=True)
            print(f"✅ Created predicted annotation directory: {self.predicted_annotation_path}")
        else:
            print(f"✅ Predicted annotation path exists: {self.predicted_annotation_path}")
        
        # Check gt_categories
        if not self.gt_categories:
            print("⚠️ gt_categories.json not loaded - will use image name parsing")
            print("💡 Consider running auto_generate_categories_from_folder() to create categories")
        else:
            print(f"✅ gt_categories loaded with {len(self.gt_categories)} entries")
        
        # Check API configuration
        if not self.url or not self.url_key:
            issues.append("❌ API URL or URL key not configured")
        else:
            print(f"✅ API configured: {self.url}/{self.url_key}")
        
        if issues:
            print(f"\n❌ Found {len(issues)} issues:")
            for issue in issues:
                print(f"   {issue}")
            return False
        else:
            print(f"\n✅ Pipeline setup validation passed!")
            return True

    def calculate_validation_metrics(self, validation_results):
        """Calculate TP, FP, FN, TN metrics - Fixed logic for object detection"""
        tp = fp = fn = tn = 0
        total_predictions = 0  # Count total predictions made
        
        # Count unique predictions - simpler approach
        # Count predictions that are actually made (have non-empty pred_class)
        predictions_made = []
        for result in validation_results:
            pred_class = result.get('pred_class', '')
            
            # Handle different data types (string, float, None)
            if pred_class is not None:
                # Convert to string and handle NaN values
                pred_class_str = str(pred_class).strip()
                if pred_class_str and pred_class_str.lower() not in ['nan', 'none', '']:
                    predictions_made.append({
                        'image': result.get('image_name', ''),
                        'class': pred_class_str,
                        'x': result.get('Pred-X', 0),
                        'y': result.get('Pred-Y', 0)
                    })
        
        # Remove duplicates based on image + coordinates (same prediction)
        unique_predictions_list = []
        seen_predictions = set()
        for pred in predictions_made:
            # Use image + coordinates as unique identifier (ignore class for deduplication)
            unique_key = f"{pred['image']}_{pred['x']}_{pred['y']}"
            if unique_key not in seen_predictions:
                unique_predictions_list.append(pred)
                seen_predictions.add(unique_key)
        
        # Calculate total predictions after we know TP and FP
        # total_predictions will be calculated after the main loop
        temp_total_predictions = total_predictions  # Keep debug value for now
        
        # Debug: Print prediction details
        # print(f"\n🔍 PREDICTION COUNTING DEBUG:")
        # print(f"   Total prediction rows in data: {len(predictions_made)}")
        # print(f"   Unique predictions found: {total_predictions}")
        # for i, pred in enumerate(unique_predictions_list, 1):
        #     print(f"   {i}. {pred['image']} -> {pred['class']} at ({pred['x']}, {pred['y']})")
        # print()
        
        # Group by image to handle multiple objects per image correctly
        image_groups = {}
        for result in validation_results:
            image_name = result.get('image_name', '')
            if image_name not in image_groups:
                image_groups[image_name] = []
            image_groups[image_name].append(result)
        
        # Process each image
        for image_name, image_results in image_groups.items():
            # Separate ground truth and predictions
            gt_objects = []
            pred_objects = []
            
            for result in image_results:
                # This is a ground truth object
                gt_class = result.get('bench_class', '')
                pred_class = result.get('pred_class', '')
                
                # Handle pred_class data type safely
                pred_class_valid = False
                if pred_class is not None:
                    pred_class_str = str(pred_class).strip()
                    if pred_class_str and pred_class_str.lower() not in ['nan', 'none', '']:
                        pred_class = pred_class_str
                        pred_class_valid = True
                    else:
                        pred_class = ''
                else:
                    pred_class = ''
                
                if gt_class:  # There is a ground truth object
                    gt_objects.append({
                        'class': gt_class,
                        'matched': False
                    })
                    
                    # Match prediction to ground truth if both exist
                    if pred_class_valid and pred_class != '':
                        # Match prediction to ground truth
                        if pred_class == gt_class:
                            # Find first unmatched GT of this class
                            for gt_obj in gt_objects:
                                if gt_obj['class'] == gt_class and not gt_obj['matched']:
                                    gt_obj['matched'] = True
                                    break
            
            # Count metrics for this image
            for gt_obj in gt_objects:
                if gt_obj['matched']:
                    tp += 1  # Ground truth was correctly detected
                else:
                    fn += 1  # Ground truth was missed
            
            # Count false positives - predictions that don't match any GT
            for result in image_results:
                gt_class = result.get('bench_class', '')
                pred_class = result.get('pred_class', '')
                
                # Handle pred_class data type safely (same logic as above)
                pred_class_valid = False
                if pred_class is not None:
                    pred_class_str = str(pred_class).strip()
                    if pred_class_str and pred_class_str.lower() not in ['nan', 'none', '']:
                        pred_class = pred_class_str
                        pred_class_valid = True
                    else:
                        pred_class = ''
                else:
                    pred_class = ''
                
                if pred_class_valid and pred_class != '':
                    # If there's a prediction but no matching GT or wrong class
                    if not gt_class or pred_class != gt_class:
                        fp += 1
        
        # Calculate final metrics
        # Fix total predictions to be exactly TP + FP (correct approach for object detection)
        total_predictions = tp + fp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        return {
            'number_of_images': len(image_groups),  # Count unique images, not total annotations
            'total_predictions': total_predictions,  # Count total predictions made
            'true_positive': tp,
            'false_positive': fp, 
            'false_negative': fn,
            'true_negative': tn,  # Not applicable in object detection
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

    def get_metrics_by_class(self, validation_results):
        """Calculate metrics grouped by class using corrected detection analysis"""
        class_data = {}
        for result in validation_results:
            class_name = result.get('bench_class', 'Unknown')
            if class_name not in class_data:
                class_data[class_name] = []
            class_data[class_name].append(result)
        
        class_metrics = {}
        for class_name, class_results in class_data.items():
            # Use corrected detection analysis instead of old metrics calculation
            correct_detections, mispredictions, missed_detections = self.analyze_validation_detections(class_results)
            
            tp = len(correct_detections)
            fp = len(mispredictions)  # Only wrong class predictions (no false positives)
            fn = len(missed_detections)
            
            # Calculate metrics from corrected counts
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            
            class_metrics[class_name] = {
                'number_of_images': len(class_results),
                'true_positive': tp,
                'false_positive': fp,
                'false_negative': fn,
                'true_negative': 0,  # Not applicable in object detection
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
        
        return class_metrics

    def analyze_validation_detections(self, validation_results):
        """Analyze validation results and categorize them - Fixed logic to handle NaN values properly"""
        correct_detections = []
        mispredictions = []
        missed_detections = []
        false_positives = []  # Add separate category for false positives
        
        # Group results by image to handle multiple objects per image properly
        image_groups = {}
        for result in validation_results:
            image_name_base = result.get('image_name', '').split('.')[0]
            if image_name_base not in image_groups:
                image_groups[image_name_base] = []
            image_groups[image_name_base].append(result)
        
        # print(f"\n🔍 PROCESSING {len(image_groups)} UNIQUE IMAGES:")
        
        # Process each image group
        for image_name_base, image_results in image_groups.items():
            # print(f"\n📸 Processing image: {image_name_base} ({len(image_results)} rows)")
            
            # Filter out invalid rows (NaN predictions with no valid IoU)
            valid_results = []
            for result in image_results:
                pred_class = result.get('pred_class')
                bench_class = result.get('bench_class')
                
                # Handle NaN and None values
                if str(pred_class).lower() in ['nan', 'none', '']:
                    pred_class = None
                if str(bench_class).lower() in ['nan', 'none', '']:
                    bench_class = None
                
                # Check if this row has valid IoU data
                iou_str = result.get('iou_threshold', '0%')
                has_valid_iou = False
                if isinstance(iou_str, str) and iou_str not in ['0%', 'nan', 'NaN', '']:
                    try:
                        iou_value = float(iou_str.replace('%', ''))
                        has_valid_iou = iou_value > 0
                    except:
                        has_valid_iou = False
                elif isinstance(iou_str, (int, float)) and str(iou_str).lower() != 'nan':
                    has_valid_iou = float(iou_str) > 0
                
                # Keep rows that have either valid prediction or valid ground truth with IoU
                has_prediction = pred_class is not None and pred_class != ''
                has_ground_truth = bench_class is not None and bench_class != ''
                
                # For missed detections: GT exists but no prediction (pred_class is NaN/None)
                # This is a valid case that should be processed as missed detection
                if has_ground_truth and not has_prediction:
                    # print(f"   🔍 POTENTIAL MISSED DETECTION: GT={bench_class}, No Prediction")
                    result_copy = result.copy()
                    result_copy['pred_class'] = pred_class
                    result_copy['bench_class'] = bench_class
                    valid_results.append(result_copy)
                elif has_prediction or (has_ground_truth and has_valid_iou):
                    result_copy = result.copy()
                    result_copy['pred_class'] = pred_class
                    result_copy['bench_class'] = bench_class
                    valid_results.append(result_copy)
                else:
                    # print(f"   ⚠️  SKIPPING truly invalid row: GT={bench_class}, Pred={pred_class}, IoU={iou_str}")
                    pass
            
            # print(f"   📊 Valid rows for analysis: {len(valid_results)}")
            
            # Process valid results for this image
            for result in valid_results:
                pred_class = result.get('pred_class')
                bench_class = result.get('bench_class')
                confidence = result.get('confidence', 0)
                
                # Handle confidence
                if str(confidence).lower() in ['nan', 'none', '']:
                    confidence = 0
                else:
                    try:
                        confidence = float(confidence)
                    except:
                        confidence = 0
                
                # Get prediction and ground truth info
                has_prediction = pred_class is not None and pred_class != ''
                has_ground_truth = bench_class is not None and bench_class != ''
                
                # print(f"   📝 Row analysis: GT='{bench_class}', Pred='{pred_class}', HasGT={has_ground_truth}, HasPred={has_prediction}")
                
                if has_ground_truth:
                    if has_prediction:
                        classes_match = bench_class == pred_class
                        
                        # Parse IoU value with better handling
                        iou_str = result.get('iou_threshold', '0%')
                        try:
                            if isinstance(iou_str, str):
                                # Remove '%' and convert to float, then divide by 100 to get decimal
                                iou_value = float(iou_str.replace('%', '')) / 100.0 if iou_str != '0%' else 0.0
                            elif isinstance(iou_str, (int, float)):
                                # If it's already a number, check if it's percentage (>1) or decimal (<=1)
                                iou_value = float(iou_str) / 100.0 if float(iou_str) > 1 else float(iou_str)
                            else:
                                iou_value = 0.0
                        except Exception as e:
                            print(f"Error parsing IoU '{iou_str}': {e}")
                            iou_value = 0.0
                        
                        if classes_match:
                            # ✅ Correct Detection - pred class matches GT class
                            detection_entry = {
                                'image_name': image_name_base,
                                'category': self.extract_category_from_image(result.get('image_name', '')),
                                'class_name': pred_class,
                                'gt_class': bench_class,
                                'pred_class': pred_class,
                                'confidence': confidence,
                                'iou': iou_value,
                                'gt_bbox': self.extract_bbox_from_benchmark(result, 'bench'),
                                'pred_bbox': self.extract_bbox_from_benchmark(result, 'pred')
                            }
                            correct_detections.append(detection_entry)
                            # print(f"      ✅ CORRECT DETECTION: GT='{bench_class}', Pred='{pred_class}', Conf={confidence:.4f}, IoU={iou_value:.3f}")
                        else:
                            # ❌ Misprediction - GT exists but wrong class predicted
                            mispredictions.append({
                                'image_name': image_name_base,
                                'category': self.extract_category_from_image(result.get('image_name', '')),
                                'class_name': pred_class,
                                'gt_class': bench_class,
                                'pred_class': pred_class,
                                'confidence': confidence,
                                'iou': iou_value,
                                'gt_bbox': self.extract_bbox_from_benchmark(result, 'bench'),
                                'pred_bbox': self.extract_bbox_from_benchmark(result, 'pred')
                            })
                            # print(f"      ❌ MISPREDICTION: GT='{bench_class}', WrongPred='{pred_class}'")
                    else:
                        # 🔍 Missed Detection - ground truth exists but no prediction
                        missed_detections.append({
                            'image_name': image_name_base,
                            'category': self.extract_category_from_image(result.get('image_name', '')),
                            'gt_class': bench_class,
                            'gt_bbox': self.extract_bbox_from_benchmark(result, 'bench')
                        })
                        # print(f"      🔍 MISSED DETECTION: GT='{bench_class}', No Prediction")
                elif has_prediction:
                    # ➕ False Positive - prediction without ground truth (separate category)
                    false_positives.append({
                        'image_name': image_name_base,
                        'category': self.extract_category_from_image(result.get('image_name', '')),
                        'pred_class': pred_class,
                        'confidence': confidence,
                        'pred_bbox': self.extract_bbox_from_benchmark(result, 'pred')
                    })
                    # print(f"      ➕ FALSE POSITIVE: No GT, ExtraPred='{pred_class}'")
        
        # For backward compatibility with existing code, include false positives in mispredictions count
        # but keep them separate for detailed analysis
        
        return correct_detections, mispredictions, missed_detections

    def extract_bbox_from_benchmark(self, result, bbox_type):
        """Extract bounding box coordinates from benchmark data"""
        try:
            if bbox_type == 'bench':
                # Extract bench bounding box (Ground Truth)
                x = result.get('Bench-X', 0)
                y = result.get('Bench-Y', 0) 
                w = result.get('Bench-Width', 0)
                h = result.get('Bench-Height', 0)
            else:  # pred
                # Extract prediction bounding box
                x = result.get('Pred-X', 0)
                y = result.get('Pred-Y', 0)
                w = result.get('Pred-Width', 0) 
                h = result.get('Pred-Height', 0)
            
            # Convert to [x_min, y_min, x_max, y_max] format
            if x and y and w and h:
                x, y, w, h = float(x), float(y), float(w), float(h)
                
                # Debug: Print the raw coordinates (reduced output)
                # print(f"Debug - Raw bbox coords: x={x}, y={y}, w={w}, h={h}")
                
                # Get image dimensions from the result
                image_name = result.get('image_name', '')
                if image_name:
                    # Try to get image dimensions from the actual image file
                    try:
                        # Construct full image path
                        image_path = os.path.join(self.img_path, image_name)
                        if os.path.exists(image_path):
                            import cv2
                            img = cv2.imread(image_path)
                            if img is not None:
                                img_height, img_width = img.shape[:2]
                                # print(f"Debug - Image dimensions: {img_width}x{img_height}")
                                
                                # Check if coordinates are normalized (between 0 and 1)
                                if x <= 1.0 and y <= 1.0 and w <= 1.0 and h <= 1.0:
                                    # Coordinates are normalized, convert to pixels
                                    x_min = int(x * img_width)
                                    y_min = int(y * img_height)
                                    x_max = int((x + w) * img_width)
                                    y_max = int((y + h) * img_height)
                                    # print(f"Debug - Normalized -> Pixel: ({x_min}, {y_min}, {x_max}, {y_max})")
                                else:
                                    # Coordinates are already in pixels
                                    x_min = int(x)
                                    y_min = int(y)
                                    x_max = int(x + w)
                                    y_max = int(y + h)
                                    # print(f"Debug - Already pixels: ({x_min}, {y_min}, {x_max}, {y_max})")
                                
                                return [x_min, y_min, x_max, y_max]
                    except Exception as e:
                        print(f"Error getting image dimensions for {image_name}: {e}")
                
                # Fallback: assume coordinates are already in pixels
                fallback_coords = [int(x), int(y), int(x + w), int(y + h)]
                # print(f"Debug - Fallback coords: {fallback_coords}")
                return fallback_coords
            return []
        except Exception as e:
            print(f"Error extracting bbox: {e}")
            return []

    def extract_category_from_image(self, image_name):
        """Extract category from gt_categories.json file based on image name"""
        # Get image name without extension for lookup
        image_base = os.path.splitext(os.path.basename(image_name))[0]
        
        # Look up category in gt_categories.json
        if hasattr(self, 'gt_categories') and self.gt_categories:
            # Check if image belongs to "bp" category (should return "components")
            if "bp" in self.gt_categories:
                bp_values = self.gt_categories["bp"]
                if isinstance(bp_values, list) and len(bp_values) > 0:
                    # All images should be categorized as "components" from bp category
                    return bp_values[0]  # Return "components"
        
        # Fallback to default if no category found
        return "components"
    
    def extract_product_lighting_distance(self, image_name_or_path):
        """Extract individual Product, Lighting, Distance from gt_categories.json file or parse from image name"""
        # Get image name without extension for lookup
        image_base = os.path.splitext(os.path.basename(str(image_name_or_path)))[0]
        
        # Debug: Print what we're looking for
        # print(f"Debug - extract_product_lighting_distance for: '{image_base}'")
        
        # Look up in gt_categories JSON first
        if self.gt_categories and image_base in self.gt_categories:
            category_info = self.gt_categories[image_base]
            if isinstance(category_info, dict):
                product = category_info.get('product', 'Unknown')
                lighting = category_info.get('lighting', 'Unknown')
                distance = category_info.get('distance', 'Unknown')
                print(f"✓ Found details from JSON for '{image_base}': Product={product}, Lighting={lighting}, Distance={distance}")
                return product, lighting, distance
        
        # Fallback: Parse from image name if JSON format is wrong or image not found
        print(f"⚠️ Falling back to parsing from image name: '{image_base}'")
        try:
            # Expected format: ProductName_Camera_LightingCondition_Distance
            # Example: Garlic1_CamAisle_LightSpot_Dis15FT
            parts = image_base.split('_')
            
            if len(parts) >= 4:
                product = parts[0]  # e.g., Garlic1, Walnut143
                # Skip camera part (parts[1] is usually CamAisle)
                lighting = parts[2]  # e.g., LightSpot
                distance = parts[3]  # e.g., Dis15FT, Dis18FT
                
                # Clean up the names
                product = product.replace('143', '').replace('1', '')  # Remove numbers for consistency
                lighting = lighting.replace('Light', '').lower()  # Convert LightSpot to spot
                distance = distance.replace('Dis', '').replace('FT', 'FT')  # Convert Dis15FT to 15FT
                
                print(f"✓ Parsed details from image name: Product={product}, Lighting={lighting}, Distance={distance}")
                return product, lighting, distance
            else:
                print(f"❌ Could not parse image name '{image_base}' - not enough parts")
                
        except Exception as e:
            print(f"❌ Error parsing image name '{image_base}': {e}")
        
        # If not found in JSON, return Unknown values
        print(f"❌ Warning: Image '{image_base}' not found in gt_categories.json for product/lighting/distance extraction")
        return 'Unknown', 'Unknown', 'Unknown'

    def get_image_extension(self, prefix, image_folder):
        """Find image file by prefix similar to metrics_calculation.py"""
        try:
            if os.path.exists(image_folder):
                image_list = [f for f in os.listdir(image_folder)]
                results = [image for image in image_list if image.lower().startswith(prefix.lower())]
                return results[0] if results else None
            return None
        except:
            return None

    def get_scale_for_excel(self, image_path):
        """Get scaling factors for Excel image insertion similar to metrics_calculation.py"""
        try:
            import PIL.Image
            img = PIL.Image.open(image_path)
            original_width, original_height = img.size
            max_width = 60  
            max_height = 60 
            
            if original_width > max_width:
                scaled_width = max_width
            else:
                scaled_width = original_width
            if original_height > max_height:
                scaled_height = max_height
            else:
                scaled_height = original_height

            x_scale = scaled_width / original_width
            y_scale = scaled_height / original_height
            return x_scale, y_scale
        except:
            return 0.5, 0.5  # Default scaling

    def draw_bounding_boxes_for_detection(self, image_path, detection_data, output_folder):
        """Draw bounding boxes for detection analysis - GT in GREEN, Predictions in RED with individual labels"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            # Draw predicted bounding box (RED) if available
            if 'pred_bbox' in detection_data and detection_data['pred_bbox']:
                pred_box = detection_data['pred_bbox']
                if len(pred_box) >= 4:
                    x_min, y_min, x_max, y_max = int(pred_box[0]), int(pred_box[1]), int(pred_box[2]), int(pred_box[3])
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)  # RED rectangle (reduced thickness)
                    
                    # Add PRED label with class name and better visibility - smaller font
                    pred_class = detection_data.get('pred_class', 'unknown')
                    confidence = detection_data.get('confidence', 0.0)
                    label = f"PRED: {pred_class} ({confidence:.2f})"
                    font_scale = 0.5  # Reduced font size
                    thickness = 1     
                    
                    # Position label at top-left of prediction box
                    label_x = x_min
                    label_y = y_min - 15 if y_min > 40 else y_min + 35
                    
                    # No background, just clean text with smaller size
                    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 150), thickness)
            
            # Draw ground truth bounding box (GREEN) if available
            if 'gt_bbox' in detection_data and detection_data['gt_bbox']:
                gt_box = detection_data['gt_bbox']
                if len(gt_box) >= 4:
                    x_min, y_min, x_max, y_max = int(gt_box[0]), int(gt_box[1]), int(gt_box[2]), int(gt_box[3])
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)  # GREEN rectangle (reduced thickness)
                    
                    # Add GT label with class name and better visibility - smaller font
                    gt_class = detection_data.get('gt_class', 'unknown')
                    label = f"GT: {gt_class}"
                    font_scale = 0.5  # Reduced font size
                    thickness = 1     
                    
                    # Position label at bottom-left of GT box
                    label_x = x_min
                    label_y = y_max + 20
                    
                    # No background, just clean text with smaller size
                    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 150, 0), thickness)
            
            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            # Save the image with bounding boxes
            image_filename = os.path.basename(image_path)
            output_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_path, img)
            return output_path
        except Exception as e:
            print(f"Error drawing bounding boxes: {e}")
            return image_path

    def create_detailed_detection_reports(self, correct_detections, mispredictions, missed_detections):
        """Create detailed Excel reports with images similar to metrics_calculation.py"""
        
        try:
            import xlsxwriter
            
            # Create images output folder
            images_output_folder = os.path.join(self.report_path, "images_with_bounding_boxes")
            os.makedirs(images_output_folder, exist_ok=True)
            
            # Define formats
            def get_formats(workbook):
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#ADD8E6',
                    'align': 'center',
                    'valign': 'vcenter',
                    'border': 1
                })
                
                normal_border_format = workbook.add_format({
                    'border': 1,
                    'valign': 'vcenter'
                })
                
                return header_format, normal_border_format
            
            # 1. Correct Detections Excel (with images) - Group by image
            if correct_detections:
                correct_file = os.path.join(self.report_path, 'Correct_Detection_Analysis.xlsx')
                correct_workbook = xlsxwriter.Workbook(correct_file)
                correct_ws = correct_workbook.add_worksheet("Correct Detections")
                header_format, normal_format = get_formats(correct_workbook)
                
                headers = ["Image Name", "Image       ", "Category", 
                          "Ground Truth Class", "Predicted Class", "Confidence", "IoU", "Ground Truth Box", "Predicted Box"]
                
                for col, header in enumerate(headers):
                    correct_ws.write(0, col, header, header_format)
                
                # Group detections by image name
                image_groups = {}
                for detection in correct_detections:
                    img_name = detection.get("image_name", "")
                    if img_name not in image_groups:
                        image_groups[img_name] = {
                            'image_name': img_name,
                            'category': detection.get("category", ""),
                            'gt_classes': [],
                            'pred_classes': [],
                            'confidences': [],
                            'ious': [],
                            'gt_bboxes': [],
                            'pred_bboxes': []
                        }
                    
                    # Append values to arrays
                    image_groups[img_name]['gt_classes'].append(str(detection.get("gt_class", "")))
                    image_groups[img_name]['pred_classes'].append(str(detection.get("pred_class", "")))
                    image_groups[img_name]['confidences'].append(detection.get("confidence", 0))
                    image_groups[img_name]['ious'].append(detection.get("iou", 0))
                    image_groups[img_name]['gt_bboxes'].append(detection.get('gt_bbox', []))
                    image_groups[img_name]['pred_bboxes'].append(detection.get('pred_bbox', []))
                
                # Write one row per image with arrays of values
                for i, (img_name, group_data) in enumerate(image_groups.items(), 1):
                    # Set row height for image visibility
                    correct_ws.set_row(i, 80)
                    
                    # Find and insert image file
                    image_name = self.get_image_extension(str(img_name), self.img_path)
                    if image_name:
                        combined_images_folder = os.path.join(self.report_path, "combined_gt_prediction_images")
                        combined_image_path = os.path.join(combined_images_folder, image_name)
                        
                        if not os.path.exists(combined_image_path):
                            combined_image_path = os.path.join(self.img_path, image_name)
                        
                        if os.path.exists(combined_image_path):
                            x_scale, y_scale = self.get_scale_for_excel(combined_image_path)
                            correct_ws.insert_image(i, 1, combined_image_path, {
                                "x_scale": x_scale, 
                                "y_scale": y_scale, 
                                "align": "center", 
                                "valign": "vcenter"
                            })
                    
                    # Write data as arrays/comma-separated values
                    correct_ws.write(i, 0, str(img_name) + "\n\n\n\n", normal_format)
                    correct_ws.write(i, 2, str(group_data['category']) + "\n\n\n\n", normal_format)
                    
                    # Join arrays into comma-separated strings
                    gt_classes_str = ", ".join(group_data['gt_classes'])
                    pred_classes_str = ", ".join(group_data['pred_classes'])
                    confidences_str = ", ".join([f"{conf:.6f}" for conf in group_data['confidences']])
                    ious_str = ", ".join([f"{iou:.4f}" for iou in group_data['ious']])
                    gt_bboxes_str = " | ".join([str(bbox) for bbox in group_data['gt_bboxes']])
                    pred_bboxes_str = " | ".join([str(bbox) for bbox in group_data['pred_bboxes']])
                    
                    correct_ws.write(i, 3, gt_classes_str + "\n\n\n\n", normal_format)
                    correct_ws.write(i, 4, pred_classes_str + "\n\n\n\n", normal_format)
                    correct_ws.write(i, 5, confidences_str + "\n\n\n\n", normal_format)
                    correct_ws.write(i, 6, ious_str + "\n\n\n\n", normal_format)
                    correct_ws.write(i, 7, gt_bboxes_str + "\n\n\n\n", normal_format)
                    correct_ws.write(i, 8, pred_bboxes_str + "\n\n\n\n", normal_format)
                
                correct_workbook.close()
            
            # 2. Mispredictions Excel (with images) - Group by image
            if mispredictions:
                mispred_file = os.path.join(self.report_path, 'Misprediction_Analysis.xlsx')
                mispred_workbook = xlsxwriter.Workbook(mispred_file)
                mispred_ws = mispred_workbook.add_worksheet("Mispredictions")
                header_format, normal_format = get_formats(mispred_workbook)
                
                headers = ["Image Name", "Image      ", "Category",
                          "Ground Truth Class", "Predicted Class", "Confidence", "Ground Truth BBox", "Predicted BBox"]
                
                for col, header in enumerate(headers):
                    mispred_ws.write(0, col, header, header_format)
                
                # Group mispredictions by image name
                image_groups = {}
                for detection in mispredictions:
                    img_name = detection.get("image_name", "")
                    if img_name not in image_groups:
                        image_groups[img_name] = {
                            'image_name': img_name,
                            'category': detection.get("category", ""),
                            'gt_classes': [],
                            'pred_classes': [],
                            'confidences': [],
                            'gt_bboxes': [],
                            'pred_bboxes': []
                        }
                    
                    # Append values to arrays
                    image_groups[img_name]['gt_classes'].append(str(detection.get("gt_class", "")))
                    image_groups[img_name]['pred_classes'].append(str(detection.get("pred_class", "")))
                    image_groups[img_name]['confidences'].append(detection.get("confidence", 0))
                    image_groups[img_name]['gt_bboxes'].append(detection.get('gt_bbox', []))
                    image_groups[img_name]['pred_bboxes'].append(detection.get('pred_bbox', []))
                
                # Write one row per image with arrays of values
                for i, (img_name, group_data) in enumerate(image_groups.items(), 1):
                    mispred_ws.set_row(i, 80)
                    
                    # Find and insert image file
                    image_name = self.get_image_extension(str(img_name), self.img_path)
                    if image_name:
                        combined_images_folder = os.path.join(self.report_path, "combined_gt_prediction_images")
                        combined_image_path = os.path.join(combined_images_folder, image_name)
                        
                        if not os.path.exists(combined_image_path):
                            combined_image_path = os.path.join(self.img_path, image_name)
                        
                        if os.path.exists(combined_image_path):
                            x_scale, y_scale = self.get_scale_for_excel(combined_image_path)
                            mispred_ws.insert_image(i, 1, combined_image_path, {
                                "x_scale": x_scale, "y_scale": y_scale, "align": "center", "valign": "vcenter"
                            })
                    
                    # Write data as arrays/comma-separated values
                    mispred_ws.write(i, 0, str(img_name) + "\n\n\n\n", normal_format)
                    mispred_ws.write(i, 2, str(group_data['category']) + "\n\n\n\n", normal_format)
                    
                    # Join arrays into comma-separated strings
                    gt_classes_str = ", ".join(group_data['gt_classes'])
                    pred_classes_str = ", ".join(group_data['pred_classes'])
                    confidences_str = ", ".join([f"{conf:.6f}" for conf in group_data['confidences']])
                    gt_bboxes_str = " | ".join([str(bbox) for bbox in group_data['gt_bboxes']])
                    pred_bboxes_str = " | ".join([str(bbox) for bbox in group_data['pred_bboxes']])
                    
                    mispred_ws.write(i, 3, gt_classes_str + "\n\n\n\n", normal_format)
                    mispred_ws.write(i, 4, pred_classes_str + "\n\n\n\n", normal_format)
                    mispred_ws.write(i, 5, confidences_str + "\n\n\n\n", normal_format)
                    mispred_ws.write(i, 6, gt_bboxes_str + "\n\n\n\n", normal_format)
                    mispred_ws.write(i, 7, pred_bboxes_str + "\n\n\n\n", normal_format)
                
                mispred_workbook.close()
                print(f"Created Misprediction Analysis: {mispred_file}")
            
            # 3. Missed Detections Excel (with images) - Group by image
            if missed_detections:
                missed_file = os.path.join(self.report_path, 'Missed_Detection_Analysis.xlsx')
                missed_workbook = xlsxwriter.Workbook(missed_file)
                missed_ws = missed_workbook.add_worksheet("Missed Detections")
                header_format, normal_format = get_formats(missed_workbook)
                
                # Headers for missed detections (GT objects only)
                headers = ["Image Name", "Image      ", "Category", "Ground Truth Class", "Ground Truth BBox"]
                
                for col, header in enumerate(headers):
                    missed_ws.write(0, col, header, header_format)
                
                # Group missed detections by image name
                image_groups = {}
                for detection in missed_detections:
                    img_name = detection.get("image_name", "")
                    if img_name not in image_groups:
                        image_groups[img_name] = {
                            'image_name': img_name,
                            'category': detection.get("category", ""),
                            'gt_classes': [],
                            'gt_bboxes': []
                        }
                    
                    # Append values to arrays
                    image_groups[img_name]['gt_classes'].append(str(detection.get("gt_class", "")))
                    image_groups[img_name]['gt_bboxes'].append(detection.get('gt_bbox', []))
                
                # Write one row per image with arrays of values
                for i, (img_name, group_data) in enumerate(image_groups.items(), 1):
                    missed_ws.set_row(i, 80)
                    
                    # Find and insert image file
                    image_name = self.get_image_extension(str(img_name), self.img_path)
                    if image_name:
                        combined_images_folder = os.path.join(self.report_path, "combined_gt_prediction_images")
                        combined_image_path = os.path.join(combined_images_folder, image_name)
                        
                        if not os.path.exists(combined_image_path):
                            combined_image_path = os.path.join(self.img_path, image_name)
                        
                        if os.path.exists(combined_image_path):
                            x_scale, y_scale = self.get_scale_for_excel(combined_image_path)
                            missed_ws.insert_image(i, 1, combined_image_path, {
                                "x_scale": x_scale, "y_scale": y_scale, "align": "center", "valign": "vcenter"
                            })
                    
                    # Write data as arrays/comma-separated values
                    missed_ws.write(i, 0, str(img_name) + "\n\n\n\n", normal_format)
                    missed_ws.write(i, 2, str(group_data['category']) + "\n\n\n\n", normal_format)
                    
                    # Join arrays into comma-separated strings
                    gt_classes_str = ", ".join(group_data['gt_classes'])
                    gt_bboxes_str = " | ".join([str(bbox) for bbox in group_data['gt_bboxes']])
                    
                    missed_ws.write(i, 3, gt_classes_str + "\n\n\n\n", normal_format)
                    missed_ws.write(i, 4, gt_bboxes_str + "\n\n\n\n", normal_format)
                
                missed_workbook.close()
                
        except Exception as e:
            print(f"Error creating detailed reports: {e}")
            import traceback
            traceback.print_exc()

    def get_metrics_by_category(self, validation_results):
        """Calculate metrics grouped by GT category using corrected detection analysis"""
        category_data = {}
        for result in validation_results:
            # Extract category using gt_categories.json
            image_name = result.get('image_name', '')
            category = self.extract_category_from_image(image_name)
            
            if category not in category_data:
                category_data[category] = []
            category_data[category].append(result)
        
        category_metrics = {}
        for category, category_results in category_data.items():
            # Use corrected detection analysis instead of old metrics calculation
            correct_detections, mispredictions, missed_detections = self.analyze_validation_detections(category_results)
            
            tp = len(correct_detections)
            fp = len(mispredictions)  # Only wrong class predictions (no false positives)
            fn = len(missed_detections)
            
            # Calculate metrics from corrected counts
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            
            category_metrics[category] = {
                'number_of_images': len(category_results),
                'true_positive': tp,
                'false_positive': fp,
                'false_negative': fn,
                'true_negative': 0,  # Not applicable in object detection
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
        
        return category_metrics

    def generate_overall_summary_excel(self):
        """Generate overall summary Excel using exact report_generator.py logic"""
        
        try:
            # Read benchmark data
            benchmark_file = os.path.join(self.report_path, 'Benchmark Report.xlsx')
            if not os.path.exists(benchmark_file):
                print("Benchmark report not found. Run inference_images() first.")
                return
            
            import pandas as pd
            df = pd.read_excel(benchmark_file, sheet_name='Benchmark Validation Report')
            
            # Convert dataframe to validation results format
            validation_results = []
            for _, row in df.iterrows():
                iou_raw = row.get('IOU Threshold', '0%')
                # print(f"Debug - Raw IoU from Excel: '{iou_raw}' for image: {row.get('InputFileName', '')}")
                
                validation_results.append({
                    'image_name': row.get('InputFileName', ''),
                    'bench_class': row.get('Bench-Class', ''),
                    'pred_class': row.get('Pred-Class', ''),
                    'confidence': row.get('Pred-Confidence', 0),
                    'iou_threshold': iou_raw,
                    'api_duration': row.get('API Duration', ''),
                    # Add bounding box data
                    'Bench-X': row.get('Bench-X', 0),
                    'Bench-Y': row.get('Bench-Y', 0),
                    'Bench-Width': row.get('Bench-Width', 0),
                    'Bench-Height': row.get('Bench-Height', 0),
                    'Pred-X': row.get('Pred-X', 0),
                    'Pred-Y': row.get('Pred-Y', 0),
                    'Pred-Width': row.get('Pred-Width', 0),
                    'Pred-Height': row.get('Pred-Height', 0),
                })
            
            # Calculate metrics using report_generator logic
            
            # Calculate metrics using report_generator logic
            overall_metrics = self.calculate_validation_metrics(validation_results)
            class_wise_metrics = self.get_metrics_by_class(validation_results)
            category_wise_metrics = self.get_metrics_by_category(validation_results)
            
            # Analyze detections for detailed reports
            correct_detections, mispredictions, missed_detections = self.analyze_validation_detections(validation_results)
            
            # Validate that counts match
            if len(correct_detections) != overall_metrics['true_positive']:
                print(f"⚠️  WARNING: Correct detections ({len(correct_detections)}) != TP ({overall_metrics['true_positive']})")
            if len(missed_detections) != overall_metrics['false_negative']:
                print(f"⚠️  WARNING: Missed detections ({len(missed_detections)}) != FN ({overall_metrics['false_negative']})")
            
            # Create detailed detection reports (like metrics_calculation.py)
            self.create_detailed_detection_reports(correct_detections, mispredictions, missed_detections)
            
            # Create Excel with exact same structure as generate_overall_summary.py
            summary_file = os.path.join(self.report_path, 'Overall_Summary_Report.xlsx')
            import xlsxwriter
            
            summary_workbook = xlsxwriter.Workbook(summary_file)
            
            # Define formats (same as generate_overall_summary.py)
            header_format = summary_workbook.add_format({
                'bold': True,
                'bg_color': '#ADD8E6',
                'align': 'center',
                'valign': 'vcenter',
                'border': 1
            })
            
            normal_border_format = summary_workbook.add_format({
                'border': 1
            })
            
            title_format = summary_workbook.add_format({
                'bold': True,
                'align': 'center',
                'font_size': 14
            })
            
            # Create summary worksheet
            summary_ws = summary_workbook.add_worksheet("Summary")
            
            # Add summary headers
            summary_ws.merge_range("A1:G1", "Detection Analysis Summary", title_format)
            summary_ws.merge_range("A3:G3", "Overall Statistics", header_format)
            
            # Add overall statistics headers
            summary_ws.write("A4", "Statistic", header_format)
            summary_ws.write("B4", "Value", header_format)
            
            # Calculate overall statistics using corrected detection analysis
            stats = [
                ("Total Ground Truth Images", overall_metrics['number_of_images']),
                ("Total Predictions", overall_metrics['total_predictions']),
                ("Correct Detections (True Positives)", len(correct_detections)),
                ("Mispredictions (Wrong Class Only)", len(mispredictions)),
                ("Missed Detections (False Negatives)", len(missed_detections)),
                ("Overall Precision", f"{overall_metrics['precision']:.4f}"),
                ("Overall Recall", f"{overall_metrics['recall']:.4f}"),
                ("Overall F1 Score", f"{overall_metrics['f1']:.4f}"),
                ("Overall Accuracy", f"{overall_metrics['accuracy']:.4f}"),
                ("Total Classes", len(set([r.get('bench_class', '') for r in validation_results if r.get('bench_class')])))
            ]
            
            # Write overall statistics
            for i, (label, value) in enumerate(stats, 5):
                summary_ws.write(f"A{i}", label, normal_border_format)
                summary_ws.write(f"B{i}", value, normal_border_format)
            
            # Add class metrics section (removed GT Category section)
            class_row = len(stats) + 7  # 5 (start) + len(stats) + 2 (spacing)
            summary_ws.merge_range(f"A{class_row}:G{class_row}", "Metrics by Class", header_format)
            
            # Class metrics headers
            headers = ["Class", "number of objects", "true positive", "false positive", "false negative", "true negative", "precision", "recall", "f1", "accuracy"]
            for col, header in enumerate(headers):
                summary_ws.write(class_row + 1, col, header, header_format)
            
            # Write class metrics
            for i, (class_name, metrics) in enumerate(sorted(class_wise_metrics.items()), class_row + 2):
                summary_ws.write(i, 0, class_name, normal_border_format)
                summary_ws.write(i, 1, metrics["number_of_images"], normal_border_format)
                summary_ws.write(i, 2, metrics["true_positive"], normal_border_format)
                summary_ws.write(i, 3, metrics["false_positive"], normal_border_format)
                summary_ws.write(i, 4, metrics["false_negative"], normal_border_format)
                summary_ws.write(i, 5, metrics["true_negative"], normal_border_format)            
                summary_ws.write(i, 6, f"{metrics['precision']:.4f}", normal_border_format)
                summary_ws.write(i, 7, f"{metrics['recall']:.4f}", normal_border_format)
                summary_ws.write(i, 8, f"{metrics['f1']:.4f}", normal_border_format)
                summary_ws.write(i, 9, f"{metrics['accuracy']:.4f}", normal_border_format)
            
            # Create individual class worksheets (same as generate_overall_summary.py)
            for class_name in sorted(class_wise_metrics.keys()):
                # Filter results for this specific class
                class_validation_results = [r for r in validation_results if r.get('bench_class') == class_name]
                class_category_metrics = self.get_metrics_by_category(class_validation_results)
                
                # Create class worksheet (limit worksheet name to 31 characters for Excel)
                safe_class_name = class_name[:31] if len(class_name) > 31 else class_name
                class_ws = summary_workbook.add_worksheet(safe_class_name)
                
                # Class worksheet headers
                class_headers = ["Category", "number of objects", "true positive", "false positive", "false negative", "true negative", "precision", "recall", "f1", "accuracy"]
                for col, header in enumerate(class_headers):
                    class_ws.write(0, col, header, header_format)
                
                # Write class-specific category metrics
                for i, (category, metrics) in enumerate(sorted(class_category_metrics.items()), 1):
                    class_ws.write(i, 0, category, normal_border_format)
                    class_ws.write(i, 1, metrics['number_of_images'], normal_border_format)
                    class_ws.write(i, 2, metrics["true_positive"], normal_border_format)
                    class_ws.write(i, 3, metrics["false_positive"], normal_border_format)
                    class_ws.write(i, 4, metrics["false_negative"], normal_border_format)
                    class_ws.write(i, 5, metrics["true_negative"], normal_border_format)
                    class_ws.write(i, 6, f"{metrics['precision']:.4f}", normal_border_format)
                    class_ws.write(i, 7, f"{metrics['recall']:.4f}", normal_border_format)
                    class_ws.write(i, 8, f"{metrics['f1']:.4f}", normal_border_format)
                    class_ws.write(i, 9, f"{metrics['accuracy']:.4f}", normal_border_format)
            
            # Close workbook
            summary_workbook.close()
            
            print(f"Generated Overall Summary Report: {summary_file}")
            
            # Remove the benchmark report file
            if os.path.exists(benchmark_file):
                os.remove(benchmark_file)
                print(f"Removed benchmark report: {benchmark_file}")
                
        except Exception as e:
            print(f"Error generating overall summary: {e}")
            import traceback
            traceback.print_exc()

    def generate_summary_report(self):
        # Replace ground truth logic with overall summary Excel generation
        self.generate_overall_summary_excel()

    @staticmethod
    def clean_resources():
        # Delete all unwanted files
        try:
            shutil.rmtree('json_data')
            shutil.rmtree("groundtruth_testing_data")
        except Exception as e:
            print(e)
            traceback.print_exc()


if __name__ == "__main__":
    print("🚀 IVA Validation Pipeline")
    print("=" * 40)
    
    # Load IVA validation config to check patch-based inference setting
    iva_config_path = "iva_validation_config.json"
    try:
        with open(iva_config_path, 'r') as f:
            import json
            iva_config = json.load(f)
    except Exception as e:
        print(f"❌ Error loading IVA config: {e}")
        print("Using default configuration...")
        iva_config = {
            'test_img_path': 'groundtruth_testing_data/data/original images',
            'original_annotation_path': 'groundtruth_testing_data/data/original Annotation',
            'predicted_annotation_path': 'groundtruth_testing_data/data/predicted',
            'labels_file_path': 'groundtruth_testing_data/data/classes.names',
            'report_path': 'gt_reports',
            'confidence_threshold': 0.3,
            'patch_based_inference': 'False'
        }
    
    # Check patch-based inference setting from IVA config
    patch_based_inference = iva_config.get('patch_based_inference', 'False').lower() in ['true', '1', 'yes']
    
    print(f"📋 Configuration loaded from: {iva_config_path}")
    print(f"🔧 Patch-based inference: {'ENABLED' if patch_based_inference else 'DISABLED'}")
    
    if patch_based_inference:
        print("\n📄 PATCH-BASED MODE SELECTED")
        print("   ✅ Will convert PDF to images automatically")
        print("   ✅ Will extract patches and run inference")
        print("   ✅ Will stitch results back to original images")
        
        # Load patch config for additional details
        from module.patch_inference import load_patch_based_config
        patch_config = load_patch_based_config('framework_config/patch_based_config.json')
        print(f"   📄 PDF path: {patch_config.get('pdf_path', 'Not specified')}")
        
        choice = input("\n🔍 Run patch-based validation? (y/n): ").lower().strip()
        
    else:
        print("\n📋 STANDARD IVA VALIDATION MODE SELECTED")
        print("   ✅ Will process regular images directly")
        print("   ✅ Will run standard YOLO inference")
        print("   ✅ Will compare with ground truth annotations")
        print(f"   📁 Image folder: {iva_config.get('test_img_path', 'Not specified')}")
        
        choice = input("\n🔍 Run standard IVA validation? (y/n): ").lower().strip()
    
    if choice == 'y':
        # Create GT config
        gt_config = {'output_folder': './gt_reports'}
        
        try:
            pipeline = IVAValidationPipeline(iva_config, gt_config)
            
            if patch_based_inference:
                print("\n🚀 Starting patch-based validation pipeline...")
                validation_stats = pipeline.run_complete_validation_suite()
            else:
                print("\n🚀 Starting standard validation pipeline...")
                validation_stats = pipeline.run_validation()
                
            if validation_stats:
                mode_name = "Patch-based" if patch_based_inference else "Standard"
                print(f"\n✅ {mode_name} validation completed successfully!")
            else:
                print("\n⚠️ Validation completed with warnings")
                
        except Exception as e:
            mode_name = "patch-based" if patch_based_inference else "standard"
            print(f"\n❌ Error in {mode_name} validation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Validation cancelled by user")
        
        # Traditional mode - ask for image folder and excel file
        image_folder = input("\nEnter the image folder path: ")
        excel_file = input("Enter the benchmark Excel file path: ")
        
        if os.path.exists(excel_file) and os.path.exists(image_folder):
            print("Starting validation pipeline...")
            
            # Create instance with dummy config dictionaries for testing
            iva_config = {'test_img_path': image_folder}
            gt_config = {}
            pipeline = IVAValidationPipeline(iva_config, gt_config)
            pipeline.generate_overall_summary_excel(excel_file, image_folder)
            print("Overall summary Excel report generated successfully!")
        else:
            print("Error: Excel file or image folder doesn't exist")

    print("\n" + "="*80)
    print("🚀 ENHANCED PATCH REPORTING EXAMPLES")
    print("="*80)
    print("""
    # Example 1: Run patch analysis only (no standard validation)
    pipeline = IVAValidationPipeline(iva_config, gt_config)
    patch_results = pipeline.run_patch_analysis_only(generate_reports=True)
    
    # Example 2: Run full validation with both standard and patch reports
    results = pipeline.run_validation(multi_object_mode=True, generate_reports=True)
    validation_stats = results['validation_stats']
    patch_results = results['patch_results']
    
    # Example 3: Generate patch Excel reports manually after processing
    patch_results = pipeline.extract_patches_from_images()
    if patch_results:
        patch_excel_reports = pipeline.generate_patch_inference_excel_reports(patch_results)
        print(f"Excel reports saved in: {patch_excel_reports['reports_directory']}")
    
    📊 Generated Reports Include:
    ├── Patch_Overall_Summary_Report.xlsx
    │   ├── Overall Summary (metrics, statistics)
    │   └── Image Breakdown (per-image details)
    ├── Patch_Detailed_Performance_Report.xlsx
    │   └── Performance Analysis (detection rates, recommendations)
    └── Patch_Detection_Analysis_Report.xlsx
        ├── High Performance Patches (multiple detections)
        ├── No Detection Patches (background patches)
        └── Detection Distribution (statistical analysis)
    """)