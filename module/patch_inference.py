'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
""" Patch-based YOLO inference pipeline: PDF -> images -> patches -> detection -> stitched results """
import cv2
import os
import json
import fitz  # PyMuPDF
from ultralytics import YOLO

# Set matplotlib to non-interactive backend for CLI operation
import matplotlib
matplotlib.use('Agg')

from patched_yolo_infer import (
    MakeCropsDetectThem,
    CombineDetections,
    # Removed visualization imports for CLI-only operation
)
##########
class PatchBasedYOLOInference:
    def __init__(self, model_path, patch_size=640, stride=256, conf=0.5, iou=0.7, classes_list=None, 
                 save_patches=False, patches_output_folder="./patch_outputs", save_patch_detections=False,
                 save_combined_visualization=False, class_names_config_path=None):
        self.model_path = model_path
        self.patch_size = patch_size
        self.stride = stride
        self.conf = conf
        self.iou = iou
        self.classes_list = classes_list if classes_list is not None else [0, 1, 2, 3, 4]
        self.save_patches = save_patches
        self.patches_output_folder = patches_output_folder
        self.save_patch_detections = save_patch_detections
        self.save_combined_visualization = save_combined_visualization
        
        # Load class names from config
        self.class_names = {}
        if class_names_config_path and os.path.exists(class_names_config_path):
            try:
                with open(class_names_config_path, 'r') as f:
                    config = json.load(f)
                    self.class_names = config.get('class_names', {})
                print(f"      ✅ Loaded class names: {self.class_names}")
            except Exception as e:
                print(f"      ⚠️ Error loading class names config: {e}")
                self.class_names = {}
        
        # Create output folders if saving is enabled
        if self.save_patches:
            os.makedirs(self.patches_output_folder, exist_ok=True)
            os.makedirs(os.path.join(self.patches_output_folder, "patches"), exist_ok=True)
            if self.save_patch_detections:
                os.makedirs(os.path.join(self.patches_output_folder, "patch_detections"), exist_ok=True)
            if self.save_combined_visualization:
                os.makedirs(os.path.join(self.patches_output_folder, "combined_results"), exist_ok=True)

    def get_class_name(self, class_id):
        """Get class name from class ID, fallback to Class X if not found"""
        class_id_str = str(class_id)
        if class_id_str in self.class_names:
            return self.class_names[class_id_str]
        elif class_id in self.class_names:
            return self.class_names[class_id]
        else:
            return f"Class {class_id}"

    def run_patch_inference(self, img, image_name="image"):
        """
        Run patch-based inference with optional patch saving and visualization
        
        Args:
            img: Input image (numpy array)
            image_name: Name for saving patches and results
            
        Returns:
            result: Combined detection results
        """
        print(f"   🔧 Running patch inference on {image_name}")
        print(f"      - Image shape: {img.shape if img is not None else 'None'}")
        print(f"      - Image dtype: {img.dtype if img is not None else 'None'}")
        print(f"      - Patch size: {self.patch_size}x{self.patch_size}")
        print(f"      - Stride: {self.stride}")
        print(f"      - Overlap: {self.patch_size - self.stride}px")
        print(f"      - Save patches: {self.save_patches}")
        print(f"      - Model path: {self.model_path}")
        print(f"      - Classes: {self.classes_list}")
        
        if img is None:
            print(f"      ❌ Input image is None!")
            return None
        
        # Completely disable all visualization for CLI-only operation
        # Set show_crops=False to prevent any matplotlib usage
        
        try:
            element_crops = MakeCropsDetectThem(
                image=img,
                model_path=self.model_path,
                segment=False,
                show_crops=False,  # Completely disable crops visualization
                shape_x=self.patch_size,
                shape_y=self.patch_size,
                overlap_x=self.patch_size - self.stride,
                overlap_y=self.patch_size - self.stride,
                conf=self.conf,
                iou=self.iou,
                classes_list=self.classes_list,
            )
            print(f"      ✅ MakeCropsDetectThem completed successfully")
            print(f"      🔍 element_crops type: {type(element_crops)}")
            
            # Detailed debugging for why no crops might be created
            if hasattr(element_crops, 'crops'):
                print(f"      🔍 Crops list length: {len(element_crops.crops)}")
                if len(element_crops.crops) == 0:
                    print(f"      🔍 Investigating why no crops were created...")
                    print(f"      🔍 Image size: {img.shape}")
                    print(f"      🔍 Patch size: {self.patch_size}")
                    print(f"      🔍 Calculated patches X: {(img.shape[1] - self.patch_size) // self.stride + 1}")
                    print(f"      🔍 Calculated patches Y: {(img.shape[0] - self.patch_size) // self.stride + 1}")
                    
                    # Check if element_crops has other attributes
                    if hasattr(element_crops, 'patches_info'):
                        print(f"      🔍 patches_info: {element_crops.patches_info}")
                    if hasattr(element_crops, 'get_crops_xy'):
                        try:
                            crops_xy = element_crops.get_crops_xy()
                            print(f"      🔍 get_crops_xy result: {crops_xy}")
                        except Exception as xy_error:
                            print(f"      🔍 get_crops_xy error: {xy_error}")
            else:
                print(f"      ⚠️ element_crops has no 'crops' attribute")
            
        except Exception as e:
            print(f"      ❌ Error in MakeCropsDetectThem: {e}")
            import traceback
            print(f"      📋 MakeCropsDetectThem traceback: {traceback.format_exc()}")
            return None
        
        # Save individual patches if enabled
        if self.save_patches and hasattr(element_crops, 'crops'):
            self.save_individual_patches(element_crops, image_name)
        
        # Combine detections only if crops exist
        try:
            if hasattr(element_crops, 'crops') and element_crops.crops and len(element_crops.crops) > 0:
                result = CombineDetections(element_crops, nms_threshold=0.05)
                print(f"      ✅ CombineDetections completed successfully")
                print(f"      🔍 result type: {type(result)}")
            else:
                print(f"      ⚠️ No crops to combine - skipping CombineDetections")
                result = None
                
        except Exception as e:
            print(f"      ❌ Error in CombineDetections: {e}")
            import traceback
            print(f"      📋 CombineDetections traceback: {traceback.format_exc()}")
            result = None
        
        # Save combined visualization if enabled
        if self.save_combined_visualization and result is not None:
            self.save_final_visualization(img, result, image_name)
        
        print(f"   ✅ Patch inference completed for {image_name}")
        return result
    
    def save_individual_patches(self, element_crops, image_name):
        """Save individual patches and their detections"""
        try:
            patches_folder = os.path.join(self.patches_output_folder, "patches", image_name)
            os.makedirs(patches_folder, exist_ok=True)
            
            # Debug: Print element_crops structure
            print(f"      🔍 Debug - element_crops type: {type(element_crops)}")
            print(f"      🔍 Debug - element_crops attributes: {dir(element_crops)}")
            
            if hasattr(element_crops, 'crops'):
                print(f"      🔍 Debug - crops type: {type(element_crops.crops)}")
                print(f"      🔍 Debug - crops length: {len(element_crops.crops) if element_crops.crops else 0}")
                
                if element_crops.crops:
                    for i, crop_data in enumerate(element_crops.crops):
                        try:
                            print(f"      🔍 Debug - Processing crop {i}, type: {type(crop_data)}")
                            
                            # Handle different possible data structures
                            patch_img = None
                            if isinstance(crop_data, dict):
                                patch_img = crop_data.get('image', None)
                            elif hasattr(crop_data, 'image'):
                                patch_img = crop_data.image
                            elif isinstance(crop_data, (list, tuple)) and len(crop_data) > 0:
                                patch_img = crop_data[0] if hasattr(crop_data[0], 'shape') else None
                            
                            if patch_img is not None:
                                patch_path = os.path.join(patches_folder, f"patch_{i:03d}.jpg")
                                cv2.imwrite(patch_path, patch_img)
                                print(f"      ✅ Saved patch {i}: {patch_path}")
                            else:
                                print(f"      ⚠️ No image data found for crop {i}")
                            
                            # Save patch detections if enabled
                            if self.save_patch_detections:
                                detections = None
                                if isinstance(crop_data, dict) and 'detections' in crop_data:
                                    detections = crop_data['detections']
                                elif hasattr(crop_data, 'detections'):
                                    detections = crop_data.detections
                                
                                if detections is not None and hasattr(detections, 'boxes') and detections.boxes is not None and len(detections.boxes) > 0:
                                    det_folder = os.path.join(self.patches_output_folder, "patch_detections", image_name)
                                    os.makedirs(det_folder, exist_ok=True)
                                    self.save_patch_detections_visualization(patch_img, detections, 
                                                                           os.path.join(det_folder, f"patch_{i:03d}_detections.jpg"))
                        except Exception as crop_error:
                            print(f"      ❌ Error processing crop {i}: {crop_error}")
                            import traceback
                            print(f"      📋 Crop traceback: {traceback.format_exc()}")
                    
                    print(f"      💾 Processed {len(element_crops.crops)} patches in {patches_folder}")
                else:
                    print(f"      ⚠️ No crops found in element_crops.crops")
            else:
                print(f"      ⚠️ element_crops has no 'crops' attribute")
                
        except Exception as e:
            print(f"      ❌ Error saving patches: {e}")
            import traceback
            print(f"      📋 Full traceback: {traceback.format_exc()}")
    
    def save_patch_detections_visualization(self, patch_img, detections, output_path):
        """Draw detections on patch and save"""
        try:
            if patch_img is None or detections is None:
                return
            
            img_with_detections = patch_img.copy()
            
            # Draw bounding boxes on patch
            if hasattr(detections, 'boxes') and detections.boxes is not None:
                boxes = detections.boxes
                for i in range(len(boxes.xyxy)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Draw bounding box
                    cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with class name instead of class ID
                    class_name = self.get_class_name(cls)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(img_with_detections, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite(output_path, img_with_detections)
        except Exception as e:
            print(f"      ⚠️ Error saving patch detection visualization: {e}")
    
    def save_final_visualization(self, original_img, result, image_name):
        """Save final combined detection results"""
        try:
            if result is None or original_img is None:
                return
            
            combined_folder = os.path.join(self.patches_output_folder, "combined_results")
            os.makedirs(combined_folder, exist_ok=True)
            
            img_with_final_detections = original_img.copy()
            
            # Draw final combined detections
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                detection_count = len(boxes.xyxy)
                
                for i in range(detection_count):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Draw bounding box (use different color for final results)
                    cv2.rectangle(img_with_final_detections, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Draw label with class name instead of class ID
                    class_name = self.get_class_name(cls)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(img_with_final_detections, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Add summary text
                summary_text = f"Total Detections: {detection_count}"
                cv2.putText(img_with_final_detections, summary_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                final_path = os.path.join(combined_folder, f"{image_name}_final_detections.jpg")
                cv2.imwrite(final_path, img_with_final_detections)
                print(f"      💾 Saved final visualization to {final_path}")
            
        except Exception as e:
            print(f"      ⚠️ Error saving final visualization: {e}")

    @staticmethod
    def pdf_to_images(pdf_path, output_folder, dpi=300):
        doc = fitz.open(pdf_path)
        os.makedirs(output_folder, exist_ok=True)
        image_paths = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            output_path = os.path.join(output_folder, f"page_{page_num+1}.png")
            pix.save(output_path)
            image_paths.append(output_path)
        return image_paths
    
    @classmethod
    def from_config(cls, config_path="patch_based_config.json"):
        """Create PatchBasedYOLOInference from config file with class names loaded"""
        config = load_patch_based_config(config_path)
        return cls(
            model_path=config.get('model_path'),
            patch_size=config.get('patch_size', 640),
            stride=config.get('stride', 256),
            conf=config.get('conf', 0.5),
            iou=config.get('iou', 0.7),
            classes_list=config.get('classes_list'),
            save_patches=config.get('save_patches', False),
            patches_output_folder=config.get('patches_output_folder', './patch_outputs'),
            save_patch_detections=config.get('save_patch_detections', False),
            save_combined_visualization=config.get('save_combined_visualization', False),
            class_names_config_path=config_path
        )
    
def load_patch_based_config(config_path="patch_based_config.json"):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}
