'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import pandas as pd
import json
import numpy as np
#from openpyxl import Workbook
#from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
#from openpyxl.utils import get_column_letter
#from openpyxl.drawing.image import Image
from io import BytesIO
import PIL.Image
from PIL.Image import Image
import os
import cv2
import xlsxwriter
import traceback
import datetime




image_list = [f for f in os.listdir('groundtruth_testing_data/data/original images/') ]
def get_image_extension( prefix):

  """Finds an image in a list based on a prefix using list comprehension."""
  #print("Img :",image_list)
  #print("Prefix:",prefix)

  results = [image for image in image_list if image.lower().startswith(prefix.lower())]
  return results[0] if results else None
  #return "groundtruth_testing_data/data/original images/"+results[0] if results else None

def create_timestamp_folder():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Format and remove last 3 digits of microseconds
    output_folder = f"groundtruth_testing_data/data/{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

def get_category_for_image(image_name, data_details):
    """
    Get category for an image from data_details DataFrame.
    Returns a default category if image not found or data_details is empty.
    Enhanced with bulletproof error handling.
    """
    try:
        # First check if data_details is None or empty
        if data_details is None or data_details.empty:
            return "unknown_unknown_unknown_unknown"
        
        # Ensure data_details has the required columns
        required_columns = ['image', 'rawtraining', 'product', 'lighting', 'distance']
        for col in required_columns:
            if col not in data_details.columns:
                return "unknown_unknown_unknown_unknown"
        
        # Convert image_name to string to ensure comparison works
        image_name = str(image_name)
        
        # Use boolean indexing more safely
        try:
            matching_mask = data_details["image"].astype(str) == image_name
            matching_rows = data_details[matching_mask]
            
            if len(matching_rows) == 0:
                # Image not found in metadata, return default category
                return "unknown_unknown_unknown_unknown"
            
            # Get the first matching row using iloc for safer access
            first_match = matching_rows.iloc[0]
            
            # Build category string with safe access
            rawtraining = str(first_match.get('rawtraining', 'unknown'))
            product = str(first_match.get('product', 'unknown'))
            lighting = str(first_match.get('lighting', 'unknown'))
            distance = str(first_match.get('distance', 'unknown'))
            
            category = f"{rawtraining}_{product}_{lighting}_{distance}"
            return category
            
        except (IndexError, KeyError, ValueError) as index_error:
            print(f"Warning: Indexing error for image {image_name}: {index_error}")
            return "unknown_unknown_unknown_unknown"
            
    except Exception as e:
        print(f"Warning: Unexpected error getting category for image {image_name}: {e}")
        return "unknown_unknown_unknown_unknown"

def add_category_to_data(gt_data_list, data_details):
    """
    Add category information to ground truth data list.
    Handles cases where data_details is empty or missing entries.
    """
    if data_details.empty or len(gt_data_list) == 0:
        print("Warning: Empty data_details DataFrame or gt_data_list. Using default categories.")
        # Add default category to all items
        modified_list = []
        for item in gt_data_list:
            modified_item = {**item, "category": "unknown_unknown_unknown_unknown"}
            modified_list.append(modified_item)
        return modified_list
    
    modified_list = []
    for item in gt_data_list:
        try:
            category = get_category_for_image(item["image_name"], data_details)
            modified_item = {**item, "category": category}
            modified_list.append(modified_item)
        except Exception as e:
            print(f"Warning: Error processing item {item.get('image_name', 'unknown')}: {e}")
            modified_item = {**item, "category": "unknown_unknown_unknown_unknown"}
            modified_list.append(modified_item)
    
    return modified_list



def calculate_iou(boxA, boxB, image_width=None, image_height=None):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Handles both normalized (0-1) and pixel coordinate formats.
    Auto-detects coordinate format (YOLO vs corner format).
    
    Args:
        boxA: Ground truth box [x_center, y_center, width, height] (YOLO format, normalized)
        boxB: Prediction box - can be YOLO format or corner format [x1, y1, x2, y2]
        image_width: Image width for pixel coordinate normalization  
        image_height: Image height for pixel coordinate normalization
    """
    print(f"DEBUG IoU: boxA (GT)={boxA}, boxB (Pred)={boxB}")
    
    # Validate inputs
    if len(boxA) != 4 or len(boxB) != 4:
        print(f"ERROR: Invalid box format - boxA has {len(boxA)} elements, boxB has {len(boxB)} elements")
        return 0.0
    
    def is_normalized(box):
        """Check if coordinates are normalized (0-1 range)"""
        return all(0 <= val <= 1 for val in box)
    
    def is_corner_format(box):
        """Detect if box is in corner format [x1, y1, x2, y2] vs YOLO format [x_center, y_center, w, h]"""
        # If any coordinate > 1, it's likely pixel coordinates
        if any(val > 1 for val in box):
            return True
        # For normalized coordinates, check if it looks like corner format
        # In corner format: x2 > x1 and y2 > y1, and typically x1, y1 < x2, y2
        if len(box) == 4:
            x1, y1, x2, y2 = box
            # Corner format characteristics:
            # - x2 > x1 and y2 > y1 
            # - All values represent positions, not center + dimensions
            if x2 > x1 and y2 > y1:
                # Additional check: in YOLO format, width and height are typically smaller than center coordinates
                # In corner format, the difference between x2-x1 and y2-y1 should be reasonable
                width_like = x2 - x1
                height_like = y2 - y1
                if width_like > 0 and height_like > 0:
                    return True
        return False
    
    def convert_corner_to_yolo(box, img_width=None, img_height=None):
        """Convert corner format [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]"""
        x1, y1, x2, y2 = box
        
        # If pixel coordinates, normalize first
        if img_width and img_height and (x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1):
            x1, y1, x2, y2 = x1/img_width, y1/img_height, x2/img_width, y2/img_height
        
        # Convert to YOLO format
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2  
        width = x2 - x1
        height = y2 - y1
        
        return [x_center, y_center, width, height]
    
    def convert_yolo_to_corners(box):
        """Convert YOLO format [x_center, y_center, width, height] to corner format [x1, y1, x2, y2]"""
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]
    
    # Process boxB based on its format
    if is_corner_format(boxB):
        print(f"DEBUG: boxB detected as corner format, converting to YOLO format")
        # Use common image dimensions if not provided
        if image_width is None or image_height is None:
            image_width, image_height = 1920, 1080  # Common defaults
            print(f"DEBUG: Using default image dimensions: {image_width}x{image_height}")
        
        boxB = convert_corner_to_yolo(boxB, image_width, image_height)
        print(f"DEBUG: Converted boxB to YOLO format: {boxB}")
    else:
        # boxB is in YOLO format, check if normalization is needed
        if not is_normalized(boxB):
            print(f"DEBUG: boxB is YOLO format but needs normalization")
            if image_width and image_height:
                x_center, y_center, width, height = boxB
                boxB = [x_center/image_width, y_center/image_height, width/image_width, height/image_height]
                print(f"DEBUG: Normalized boxB: {boxB}")
    
    # Ensure boxA is in YOLO format (it should be already)
    if not is_normalized(boxA):
        print(f"WARNING: boxA (ground truth) is not normalized: {boxA}")
        if image_width and image_height:
            if is_corner_format(boxA):
                boxA = convert_corner_to_yolo(boxA, image_width, image_height)
            else:
                x_center, y_center, width, height = boxA
                boxA = [x_center/image_width, y_center/image_height, width/image_width, height/image_height]
            print(f"DEBUG: Normalized boxA: {boxA}")
    
    # Now both boxes should be in normalized YOLO format
    # Convert both to corner format for IoU calculation
    boxA_corners = convert_yolo_to_corners(boxA)
    boxB_corners = convert_yolo_to_corners(boxB)
    
    print(f"DEBUG IoU: boxA_corners={boxA_corners}, boxB_corners={boxB_corners}")
    
    # Calculate intersection
    xA = max(boxA_corners[0], boxB_corners[0])
    yA = max(boxA_corners[1], boxB_corners[1])
    xB = min(boxA_corners[2], boxB_corners[2])
    yB = min(boxA_corners[3], boxB_corners[3])

    print(f"DEBUG IoU: intersection coords: xA={xA}, yA={yA}, xB={xB}, yB={yB}")

    # If no intersection
    if xB <= xA or yB <= yA:
        print("DEBUG IoU: No intersection detected")
        return 0.0

    interArea = (xB - xA) * (yB - yA)

    # Calculate areas
    boxAArea = (boxA_corners[2] - boxA_corners[0]) * (boxA_corners[3] - boxA_corners[1])
    boxBArea = (boxB_corners[2] - boxB_corners[0]) * (boxB_corners[3] - boxB_corners[1])

    print(f"DEBUG IoU: interArea={interArea}, boxAArea={boxAArea}, boxBArea={boxBArea}")

    # Calculate IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    print(f"DEBUG IoU: Final IoU={iou}")
    return iou

def analyze_detections(groundtruth_details, prediction_details, iou_threshold=0.2):
    """
    Analyze detection results and categorize them as correct detections, mispredictions, or missed detections
    Uses the same logic as MC.py - each GT gets matched to at most one prediction (best IoU for same class)
    """
    correct_detection_list = []
    misprediction_list = []
    missed_prediction_list = []
    
    # Track used predictions to handle unused ones as FP (improvement over MC.py)
    used_predictions = set()
    
    # Handle case when there are no ground truth detections but there are predictions
    if len(groundtruth_details) == 0 and len(prediction_details) > 0:
        # All predictions are false positives (mispredictions)
        for pred in prediction_details:
            misprediction_list.append({
                "image_name": pred["image_name"],
                "category": pred["category"],
                "class_name": pred["class_name"],
                "gt_class_name": "NO_GT_OBJECT",  # No ground truth
                "pred_bbox": pred["bounding_box"],
                "confidence_score": pred.get("confidence_score", None),
                "gt_bbox": "N/A"
            })
        
        return correct_detection_list, misprediction_list, missed_prediction_list
    
    # Process each ground truth (same logic as MC.py)
    for gt in groundtruth_details:
        predicted_data = [pred for pred in prediction_details if gt["image_name"] == pred["image_name"]]
        
        if len(predicted_data) == 0:
            # No predictions for this ground truth - it's a missed detection
            missed_prediction_list.append({
                "image_name": gt["image_name"],
                "category": gt["category"],
                "class_name": gt["class_name"],
                "gt_bbox": gt["bounding_box"]
            })
        else:
            best_iou = 0
            best_prediction = None
            best_pred_index = -1
            
            # Find the best matching prediction for this ground truth
            for pred_idx, pred in enumerate(predicted_data):
                if pred["class_name"] == gt["class_name"]:
                    iou = calculate_iou(gt["bounding_box"], pred["bounding_box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_prediction = pred
                        # Track the global index
                        best_pred_index = next(i for i, p in enumerate(prediction_details) if p is pred)
            
            if best_prediction and best_iou >= iou_threshold:
                # This is a correct detection (TP)
                used_predictions.add(best_pred_index)
                correct_detection_list.append({
                    "image_name": gt["image_name"],
                    "category": gt["category"],
                    "class_name": gt["class_name"],
                    "gt_bbox": gt["bounding_box"],
                    "pred_bbox": best_prediction["bounding_box"],
                    "confidence_score": best_prediction.get("confidence_score", None),
                    "iou": best_iou
                })
            elif best_prediction and best_prediction["class_name"] == gt["class_name"]:
                # FIXED: Same class prediction with low IoU should still be TP, not FP
                used_predictions.add(best_pred_index)
                correct_detection_list.append({
                    "image_name": gt["image_name"],
                    "category": gt["category"],
                    "class_name": gt["class_name"],
                    "gt_bbox": gt["bounding_box"],
                    "pred_bbox": best_prediction["bounding_box"],
                    "confidence_score": best_prediction.get("confidence_score", None),
                    "iou": best_iou
                })
            else:
                # No good match found or different class
                if best_prediction:
                    # There was a prediction but it was DIFFERENT class - misprediction
                    print(f"DEBUG: Adding to mispredictions - GT: {gt['class_name']}, Pred: {best_prediction.get('class_name', 'unknown')}, Same class? {best_prediction.get('class_name', 'unknown') == gt['class_name']}")
                    used_predictions.add(best_pred_index)
                    misprediction_list.append({
                        "image_name": gt["image_name"],
                        "category": gt["category"],
                        "class_name": best_prediction.get("class_name", "unknown"),
                        "gt_class_name": gt["class_name"],
                        "pred_bbox": best_prediction["bounding_box"],
                        "confidence_score": best_prediction.get("confidence_score", None),
                        "gt_bbox": gt["bounding_box"]
                    })
                else:
                    # Check if there are predictions with different class names
                    if predicted_data:
                        # FIXED: Only add different-class predictions as mispredictions
                        different_class_predictions = [pred for pred in predicted_data if pred["class_name"] != gt["class_name"]]
                        
                        if different_class_predictions:
                            # There are predictions but none match the ground truth class
                            pred = different_class_predictions[0]  # Take the first DIFFERENT class prediction
                            pred_index = next(i for i, p in enumerate(prediction_details) if p is pred)
                            used_predictions.add(pred_index)
                            print(f"DEBUG: Adding different class to mispredictions - GT: {gt['class_name']}, Pred: {pred.get('class_name', 'unknown')}")
                            misprediction_list.append({
                                "image_name": gt["image_name"],
                                "category": gt["category"],
                                "class_name": pred.get("class_name", "unknown"),
                                "gt_class_name": gt["class_name"],
                                "pred_bbox": pred["bounding_box"],
                                "confidence_score": pred.get("confidence_score", None),
                                "gt_bbox": gt["bounding_box"]
                            })
                        else:
                            # All predictions are same class - this shouldn't happen as they should be handled above
                            # This is a missed detection
                            missed_prediction_list.append({
                                "image_name": gt["image_name"],
                                "category": gt["category"],
                                "class_name": gt["class_name"],
                                "gt_bbox": gt["bounding_box"]
                            })
                    else:
                        # No predictions at all - missed detection
                        missed_prediction_list.append({
                            "image_name": gt["image_name"],
                            "category": gt["category"],
                            "class_name": gt["class_name"],
                            "gt_bbox": gt["bounding_box"]
                        })
    
    # Handle unused predictions as false positives
    for pred_idx, pred in enumerate(prediction_details):
        if pred_idx not in used_predictions:
            # This prediction wasn't matched to any GT - it's a false positive
            gt_image_names = {gt["image_name"] for gt in groundtruth_details}
            if pred["image_name"] in gt_image_names:
                # Image has GT but this prediction didn't match any
                gt_for_image = [gt for gt in groundtruth_details if gt["image_name"] == pred["image_name"]]
                if gt_for_image:
                    # FIXED: Check if this unused prediction is same class as any GT
                    is_same_class_as_any_gt = any(
                        pred["class_name"] == gt["class_name"] 
                        for gt in gt_for_image
                    )
                    
                    if not is_same_class_as_any_gt:
                        # Only add to mispredictions if it's a DIFFERENT class
                        gt = gt_for_image[0]  # Use first GT as reference
                        print(f"DEBUG: Adding unused different class to mispredictions - GT: {gt['class_name']}, Pred: {pred['class_name']}")
                        misprediction_list.append({
                            "image_name": pred["image_name"],
                            "category": pred["category"],
                            "class_name": pred["class_name"],
                            "gt_class_name": gt["class_name"],
                            "pred_bbox": pred["bounding_box"],
                            "confidence_score": pred.get("confidence_score", None),
                            "gt_bbox": gt["bounding_box"]
                        })
                    else:
                        print(f"DEBUG: Ignoring unused same class prediction - Class: {pred['class_name']}")
                    # If it's same class, ignore it (it's a duplicate/extra same-class prediction)
            else:
                # This prediction is for an image with no ground truth - it's a false positive
                misprediction_list.append({
                    "image_name": pred["image_name"],
                    "category": pred["category"],
                    "class_name": pred["class_name"],
                    "gt_class_name": "NO_GT_OBJECT",  # No ground truth for this image
                    "pred_bbox": pred["bounding_box"],
                    "confidence_score": pred.get("confidence_score", None),
                    "gt_bbox": "N/A"
                })
    
    return correct_detection_list, misprediction_list, missed_prediction_list


def calculate_metrics_by_category(correct_detections, mispredictions, missed_detections):
    """
    Calculate metrics (precision, recall, F1) by category
    """
    # Get all unique categories
    all_categories = set()
    for item in correct_detections + mispredictions + missed_detections:
        all_categories.add(item["category"])
    
    metrics = {}
    for category in sorted(all_categories):
        tp = len([d for d in correct_detections if d["category"] == category])
        fp = len([d for d in mispredictions if d["category"] == category])
        fn = len([d for d in missed_detections if d["category"] == category])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[category] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return metrics

def calculate_metrics_for_overall_class(correct_detections, mispredictions, missed_detections):
    """
    Calculate metrics (precision, recall, F1) by class
    """
    # Get all unique classes
    all_classes = set()
    for item in correct_detections + mispredictions + missed_detections:
        all_classes.add(item["class_name"])
    
    metrics = {}
    for class_name in sorted(all_classes):
        tp = len([d for d in correct_detections if d["class_name"] == class_name])
        # Note: For mispredictions, we might not have the correct class_name
        fp = len([d for d in mispredictions if d["class_name"] == class_name])
        fn = len([d for d in missed_detections if d["class_name"] == class_name])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[class_name] = {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn, 
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return metrics

def get_image_excel(image_path):
    img = PIL.Image.open(image_path)
    # Resize the image to a thumbnail size (adjust as needed)
    thumbnail_size = (80, 80)
    img.thumbnail(thumbnail_size)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')  # or JPEG
    img_byte_arr.seek(0)
    img_excel = Image(img_byte_arr)
    return img_excel

def get_scale(image_path):
    img = PIL.Image.open(image_path)
    original_width,original_height = img.size
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
    return x_scale,y_scale
      

def draw_bounding_boxes_fixed(image_path, gt_detections, pred_detections, correct_detections, missed_detections, mispredictions, output_folder):
    """
    Fixed version with coordinate validation that draws all bounding boxes for an image at once to avoid double drawing
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    height, width, _ = img.shape
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    def validate_and_convert_bbox(bbox, source_type, img_width, img_height):
        """Validate and safely convert bounding box coordinates"""
        if not isinstance(bbox, list) or len(bbox) != 4:
            print(f"⚠️  Invalid bbox from {source_type}: {bbox}")
            return None
        
        x_center, y_center, box_width, box_height = bbox
        
        # Check if coordinates are normalized
        normalized = all(0 <= coord <= 1 for coord in bbox)
        
        if not normalized:
            print(f"⚠️  {source_type} bbox not normalized: {bbox}")
            # Try to normalize if they seem like pixel coordinates
            max_coord = max(bbox)
            if max_coord > 10:  # Likely pixel coordinates
                print(f"   Converting from pixel coordinates...")
                x_center = x_center / img_width
                y_center = y_center / img_height
                box_width = box_width / img_width
                box_height = box_height / img_height
                print(f"   Normalized to: [{x_center:.3f}, {y_center:.3f}, {box_width:.3f}, {box_height:.3f}]")
        
        # Convert to pixel coordinates
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        box_width_px = box_width * img_width
        box_height_px = box_height * img_height
        
        # Calculate corners
        x1 = int(x_center_px - box_width_px / 2)
        y1 = int(y_center_px - box_height_px / 2)
        x2 = int(x_center_px + box_width_px / 2)
        y2 = int(y_center_px + box_height_px / 2)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # Validate box size
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️  Invalid box size for {source_type}: ({x1},{y1}) to ({x2},{y2})")
            return None
        
        return [x1, y1, x2, y2]
    
    print(f"🔧 Processing image: {image_name} ({width}x{height})")
    boxes_drawn = 0
    
    # SIMPLIFIED: Only draw GT (green) and best prediction (blue) - 2 boxes total
    gt_for_image = [gt for gt in gt_detections if gt.get("image_name", "").replace(".jpg", "").replace(".png", "") == image_name]
    print(f"📊 Found {len(gt_for_image)} GT detections")
    
    # Draw only the first GT box (green)
    if gt_for_image:
        gt = gt_for_image[0]  # Take first GT only
        if 'bounding_box' in gt:
            coords = validate_and_convert_bbox(gt['bounding_box'], "GT", width, height)
            if coords:
                x1, y1, x2, y2 = coords
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green for GT
                cv2.putText(img, f"GT: {gt.get('class_name', 'unknown')}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                boxes_drawn += 1
    
    # Draw only the best prediction (blue/red) - but SKIP for missed detections
    best_pred = None
    
    # Check if this image has missed detections (no prediction boxes should be drawn)
    missed_for_image = [det for det in missed_detections if det.get("image_name", "").replace(".jpg", "").replace(".png", "") == image_name]
    
    # Only draw prediction boxes if this is NOT a missed detection
    if not missed_for_image:
        # First check for correct detections
        correct_for_image = [det for det in correct_detections if det.get("image_name", "").replace(".jpg", "").replace(".png", "") == image_name]
        if correct_for_image:
            best_pred = correct_for_image[0]  # Take first correct detection
            color = (255, 0, 0)  # Blue for correct
            label = "PRED"
        else:
            # If no correct detections, show misprediction
            mispred_for_image = [det for det in mispredictions if det.get("image_name", "").replace(".jpg", "").replace(".png", "") == image_name]
            if mispred_for_image:
                best_pred = mispred_for_image[0]  # Take first misprediction
                color = (0, 0, 255)  # Red for misprediction
                label = "FP"
        
        # Draw the best prediction (only if not a missed detection)
        if best_pred and 'pred_bbox' in best_pred:
            coords = validate_and_convert_bbox(best_pred['pred_bbox'], "PRED", width, height)
            if coords:
                x1, y1, x2, y2 = coords
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, f"{label}: {best_pred.get('class_name', 'unknown')}", 
                           (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                boxes_drawn += 1
    else:
        print(f"DEBUG: Skipping prediction box for missed detection image: {image_name}")
    
    print(f"✅ Successfully drew {boxes_drawn} validated bounding boxes")
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save the image with all bounding boxes
    image_filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_filename)
    cv2.imwrite(output_path, img)
    print(f"DEBUG: Saved image with bounding boxes to {output_path}")
    
    return output_path

def draw_bounding_boxes(image_path, gt,detection, output_folder):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return
        height, width, _ = img.shape
        class_name = detection['class_name']
        
        def convert_yolo_to_pixel_coords(yolo_box, img_width, img_height):
            """Convert YOLO format [x_center, y_center, width, height] to pixel coordinates [x1, y1, x2, y2]"""
            x_center, y_center, box_width, box_height = yolo_box
            
            # Convert normalized coordinates to pixel coordinates
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            box_width_px = box_width * img_width
            box_height_px = box_height * img_height
            
            # Calculate corner coordinates
            x1 = int(x_center_px - box_width_px / 2)
            y1 = int(y_center_px - box_height_px / 2)
            x2 = int(x_center_px + box_width_px / 2)
            y2 = int(y_center_px + box_height_px / 2)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            return [x1, y1, x2, y2]
        
        if "pred_bbox" in detection:
            det_box = detection['pred_bbox']
            # Convert YOLO format to pixel coordinates
            x1, y1, x2, y2 = convert_yolo_to_pixel_coords(det_box, width, height)
            # Draw prediction bounding box in red
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 10)  # Red rectangle
            
        if 'gt_bbox' in detection: 
            gt_box = detection['gt_bbox']
            # Convert YOLO format to pixel coordinates
            x1, y1, x2, y2 = convert_yolo_to_pixel_coords(gt_box, width, height)
            # Draw ground truth bounding box in green
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 10)  # Green rectangle
            
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Save the image with bounding boxes
        image_filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, img)
        return output_path

def calculate_metrics_by_class(groundtruth_details,correct_detections, mispredictions, missed_detections):
    unique_class = list({gt["class_name"] for gt in groundtruth_details})
    class_wise_metrics = {}
    for class_name in unique_class:
        correct_detections_class = [cd for cd in correct_detections if cd["class_name"] == class_name]
        mispredictions_class = [cd for cd in mispredictions if cd["gt_class_name"] == class_name]
        missed_detections_class = [cd for cd in missed_detections if cd["class_name"] == class_name]
        class_wise_metrics[class_name] = calculate_metrics_by_category(correct_detections_class, mispredictions_class, missed_detections_class)
    return class_wise_metrics

def export_to_excel(groundtruth_details, prediction_details, correct_detections, mispredictions, missed_detections, category_metrics, class_metrics,class_wise_metrics):
    """
    Export all results to an Excel file using XlsxWriter
    """    
    # Ensure output directory exists
    os.makedirs("groundtruth_testing_data/data/images_with_bounding_boxes/", exist_ok=True)
    summary_excel_file="Overall_summary.xlsx"
    timestamp_folder = create_timestamp_folder()
    
    # FIXED: Process all images for bounding boxes ONCE to avoid double detection
    processed_images = {}
    print("DEBUG: Pre-processing all images with bounding boxes to avoid duplicates...")
    
    # Get all unique image names from all detection types
    all_image_names = set()
    for detection in correct_detections + mispredictions + missed_detections:
        all_image_names.add(detection.get("image_name", ""))
    
    # Process each unique image only once
    for image_name in all_image_names:
        if not image_name:
            continue
            
        # Find the actual image file
        actual_image_name = get_image_extension(image_name)
        if actual_image_name is None:
            print(f"Warning: Could not find image for {image_name}")
            continue
            
        image_path = f"groundtruth_testing_data/data/original images/{actual_image_name}"
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        # Draw all bounding boxes for this image at once
        output_path = draw_bounding_boxes_fixed(
            image_path, 
            groundtruth_details,
            prediction_details,
            correct_detections, 
            missed_detections, 
            mispredictions, 
            "groundtruth_testing_data/data/images_with_bounding_boxes"
        )
        
        if output_path:
            processed_images[image_name] = output_path
            print(f"DEBUG: Processed bounding boxes for {image_name}")
    
    print(f"DEBUG: Pre-processed {len(processed_images)} unique images with bounding boxes")
    
    # Create a new workbook
    output_path = f"{timestamp_folder}/{summary_excel_file}"
    summary_workbook = xlsxwriter.Workbook(output_path)

    # Initialize all workbook variables to None to prevent UnboundLocalError
    correct_detection_workbook = None
    missed_detections_workbook = None
    mispredictions_workbook = None

    
    
    try:
        # Define formats
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
        # brown_format = summary_workbook.add_format({
        #     'border': 1,
        #     'bg_color': '#a82a07'
        # })
        # red_format = summary_workbook.add_format({
        #     'border': 1,
        #     'bg_color': '#d41002'
        # })
        # orange_format = summary_workbook.add_format({
        #     'border': 1,
        #     'bg_color': '#e88738'
        # })
        # dark_green_format = summary_workbook.add_format({
        #     'border': 1,
        #     'bg_color': '#18910a'
        # })
        # yellow_format = summary_workbook.add_format({
        #     'border': 1,
        #     'bg_color': '#eaff05'
        # })
        # light_green = summary_workbook.add_format({
        #     'border': 1,
        #     'bg_color': '#82d669'
        # }) 
        
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
          # Calculate overall statistics
        stats = [
            ("Total Ground Truth Images", len(groundtruth_details)),
            ("Total Predictions", len(prediction_details)),
            ("Correct Detections (True Positives)", len(correct_detections)),
            ("Mispredictions (False Positives)", len(mispredictions)),
            ("Missed Detections (False Negatives)", len(missed_detections)),
        ]
        
        # Write overall statistics
        for i, (label, value) in enumerate(stats, 5):
            summary_ws.write(f"A{i}", label, normal_border_format)
            summary_ws.write(f"B{i}", value, normal_border_format)
        
        # Calculate overall metrics
        tp = len(correct_detections)
        fp = len(mispredictions)
        fn = len(missed_detections)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        total_class = len({gt["class_name"] for gt in groundtruth_details})
        metric_stats = [
            ("Overall Precision", f"{precision:.4f}"),
            ("Overall Recall", f"{recall:.4f}"),
            ("Overall F1 Score", f"{f1:.4f}"),
            ("Total Classes", total_class),
        ]
        
        # Write overall metrics
        last_row = len(stats) + 4
        for i, (label, value) in enumerate(metric_stats, last_row + 1):
            summary_ws.write(f"A{i}", label, normal_border_format)
            summary_ws.write(f"B{i}", value, normal_border_format)
        
        # Add category metrics section
        category_row = last_row + len(metric_stats) + 2
        summary_ws.merge_range(f"A{category_row}:G{category_row}", "Metrics by GT Category", header_format)
        
        # Category metrics headers
        headers = ["Category","Product","Lighting","Distance", "True Positives", "False Positives", "False Negatives", "Precision", "Recall", "F1"]
        for col, header in enumerate(headers):
            summary_ws.write(category_row + 1, col, header, header_format)
        
        # Write category metrics
        for i, (category, metrics) in enumerate(sorted(category_metrics.items()), category_row + 2):
            # accuracy_percent = (metrics["true_positives"] / (metrics["true_positives"] + metrics["false_positives"] + metrics["false_negatives"])) *100
            # if accuracy_percent >= 80:
            #     color_format = dark_green_format
            # elif accuracy_percent >= 70:
            #     color_format = light_green
            # elif accuracy_percent >= 50:
            #     color_format = yellow_format
            # elif accuracy_percent >= 30:
            #     color_format = orange_format
            # elif accuracy_percent >= 20:
            #     color_format = red_format
            # elif accuracy_percent < 20:
            #     color_format = brown_format
            summary_ws.write(i, 0, category, normal_border_format)
            category_split = category.split("_")
            summary_ws.write(i, 1, category_split[1], normal_border_format)
            summary_ws.write(i, 2, category_split[2], normal_border_format)
            summary_ws.write(i, 3, category_split[3], normal_border_format)
            summary_ws.write(i, 4, metrics["true_positives"], normal_border_format)
            summary_ws.write(i, 5, metrics["false_positives"], normal_border_format)
            summary_ws.write(i, 6, metrics["false_negatives"], normal_border_format)
            summary_ws.write(i, 7, f"{metrics['precision']:.4f}", normal_border_format)
            summary_ws.write(i, 8, f"{metrics['recall']:.4f}", normal_border_format)
            summary_ws.write(i, 9, f"{metrics['f1']:.4f}", normal_border_format)
        
        # Add class metrics section
        class_row = category_row + len(category_metrics) + 3
        summary_ws.merge_range(f"A{class_row}:G{class_row}", "Metrics by Class", header_format)
        
        # Class metrics headers
        headers = ["Class", "True Positives", "False Positives", "False Negatives", "Precision", "Recall", "F1"]
        for col, header in enumerate(headers):
            summary_ws.write(class_row + 1, col, header, header_format)
        
        # Write class metrics
        for i, (class_name, metrics) in enumerate(sorted(class_metrics.items()), class_row + 2):
            # accuracy_percent = (metrics["true_positives"] / (metrics["true_positives"] + metrics["false_positives"] + metrics["false_negatives"]))*100
            # if accuracy_percent >= 80:
            #     color_format = dark_green_format
            # elif accuracy_percent >= 70:
            #     color_format = light_green
            # elif accuracy_percent >= 50:
            #     color_format = yellow_format
            # elif accuracy_percent >= 30:
            #     color_format = orange_format
            # elif accuracy_percent >= 20:
            #     color_format = red_format
            # elif accuracy_percent < 20:
            #     color_format = brown_format
            summary_ws.write(i, 0, class_name, normal_border_format)
            summary_ws.write(i, 1, metrics["true_positives"], normal_border_format)
            summary_ws.write(i, 2, metrics["false_positives"], normal_border_format)
            summary_ws.write(i, 3, metrics["false_negatives"], normal_border_format)
            summary_ws.write(i, 4, f"{metrics['precision']:.4f}", normal_border_format)
            summary_ws.write(i, 5, f"{metrics['recall']:.4f}", normal_border_format)
            summary_ws.write(i, 6, f"{metrics['f1']:.4f}", normal_border_format)

        for class_name in class_wise_metrics:
            # Create class worksheet
            class_ws = summary_workbook.add_worksheet(class_name)
             # Class metrics headers
            headers = ["Category","Product","Lighting","Distance","num_of_images",  "True Positives", "False Positives", "False Negatives", "Precision", "Recall", "F1"]
            for col, header in enumerate(headers):
                class_ws.write(0, col, header, header_format)
            class_metrics = class_wise_metrics[class_name]
            i = 0
            for category,metrics in  class_metrics.items():
                i+=1
                #num_images = len([gt for gt in groundtruth_details if gt["category"] == category])
                num_images = len(set([gt["image_name"] for gt in groundtruth_details if gt["category"] == category and gt["class_name"] == class_name]))
                class_ws.write(i, 0, category, normal_border_format)
                category_split = category.split("_")
                class_ws.write(i, 1,  category_split[1], normal_border_format)
                class_ws.write(i, 2,  category_split[2], normal_border_format)
                class_ws.write(i, 3,  category_split[3], normal_border_format)
                class_ws.write(i, 4, num_images, normal_border_format)
                class_ws.write(i, 5, metrics["true_positives"], normal_border_format)
                class_ws.write(i, 6, metrics["false_positives"], normal_border_format)
                class_ws.write(i, 7, metrics["false_negatives"], normal_border_format)
                class_ws.write(i, 8, f"{metrics['precision']:.4f}", normal_border_format)
                class_ws.write(i, 9, f"{metrics['recall']:.4f}", normal_border_format)
                class_ws.write(i, 10, f"{metrics['f1']:.4f}", normal_border_format)
          

        # Create worksheets for detections
        # 1. Correct Detections        if correct_detections:
            correct_detection_excel_file="Correct_Detection_analysis.xlsx"
            # Create a new workbook
            output_path = f"{timestamp_folder}/{correct_detection_excel_file}"
            correct_detection_workbook = xlsxwriter.Workbook(output_path)
            correct_ws = correct_detection_workbook.add_worksheet("Correct Detections")
            headers = ["Image Name","Image       ","Category","Product","Lighting","Distance", "GroundTruth Class","Predicted Class", "Confidence", "IoU",  "Ground Truth Box", "Predicted Box"]
            
            for col, header in enumerate(headers):
                correct_ws.write(0, col, header, header_format)
            
            for i, detection in enumerate(correct_detections, 1):
                image_name = get_image_extension( detection["image_name"])
                if image_name is None:
                    print(f"Warning: Could not find image for {detection['image_name']}")
                    image_name = detection["image_name"]  # Fallback to original name
                    image_path = None
                else:
                    image_path = "groundtruth_testing_data/data/original images/"+ image_name
                
                correct_ws.write(i, 0, image_name + "\n\n\n\n", normal_border_format)
                
                # Get GT details for data fields (not for drawing)
                image_gt_details = [gt for gt in groundtruth_details if gt["image_name"] == detection["image_name"]]
                image_gt_details = image_gt_details[0] if image_gt_details else {}
                
                # FIXED: Use pre-processed image with bounding boxes to avoid double detection
                if detection["image_name"] in processed_images:
                    image_path = processed_images[detection["image_name"]]
                    x_scale, y_scale = get_scale(image_path)
                    correct_ws.insert_image(i,1,image_path,{"x_scale":x_scale,"y_scale":y_scale,"align":"center","valign":"vcenter"})
                    print(f"DEBUG: Used pre-processed image for correct detection: {image_name}")
                else:
                    correct_ws.write(i, 1, "Image not found\n\n\n\n", normal_border_format)
                    print(f"DEBUG: No pre-processed image found for: {detection['image_name']}")
                
                correct_ws.write(i, 2, detection["category"]+ "\n\n\n\n", normal_border_format)
                category = detection["category"]
                category_split = category.split("_")
                correct_ws.write(i, 3, category_split[1], normal_border_format)
                correct_ws.write(i, 4, category_split[2], normal_border_format)
                correct_ws.write(i, 5, category_split[3], normal_border_format)
                correct_ws.write(i, 6, image_gt_details["class_name"]+ "\n\n\n\n", normal_border_format)
                correct_ws.write(i, 7, detection["class_name"]+ "\n\n\n\n", normal_border_format)
                correct_ws.write(i, 8, str(detection["confidence_score"])+ "\n\n\n\n", normal_border_format)
                correct_ws.write(i, 9, f"{detection['iou']:.4f}"+ "\n\n\n\n", normal_border_format)
                correct_ws.write(i, 10, str(detection["gt_bbox"])+ "\n\n\n\n", normal_border_format)
                correct_ws.write(i, 11, str(detection["pred_bbox"])+ "\n\n\n\n", normal_border_format)
                
        
        # 2. Mispredictions (False Positives)
        if mispredictions:
            mispredictions_excel_file="Misprediction_analysis.xlsx"
            # Create a new workbook            
            output_path = f"{timestamp_folder}/{mispredictions_excel_file}"
            mispredictions_workbook = xlsxwriter.Workbook(output_path)
            mispred_ws = mispredictions_workbook.add_worksheet("Mispredictions")
            headers = ["Image Name","Image      ", "Category","Product","Lighting","Distance","Ground Truth Class","Predicted Class", "Confidence","Ground Truth BBox", "Predicted BBox"]
            
            for col, header in enumerate(headers):
                mispred_ws.write(0, col, header, header_format)
            
            for i, detection in enumerate(mispredictions, 1):
                image_name = get_image_extension( detection["image_name"])
                if image_name is None:
                    print(f"Warning: Could not find image for {detection['image_name']}")
                    image_name = detection["image_name"]  # Fallback to original name
                    image_path = None
                else:
                    image_path = "groundtruth_testing_data/data/original images/"+ image_name
                
                mispred_ws.write(i, 0, image_name + "\n\n\n\n", normal_border_format)
                
                # Get GT details for data fields (not for drawing)
                image_gt_details = [gt for gt in groundtruth_details if gt["image_name"] == detection["image_name"]]
                image_gt_details = image_gt_details[0] if image_gt_details else {}
                
                # FIXED: Use pre-processed image with bounding boxes to avoid double detection
                if detection["image_name"] in processed_images:
                    image_path = processed_images[detection["image_name"]]
                    x_scale, y_scale = get_scale(image_path)
                    mispred_ws.insert_image(i,1,image_path,{"x_scale":x_scale,"y_scale":y_scale,"align":"center","valign":"vcenter"})
                    print(f"DEBUG: Used pre-processed image for misprediction: {image_name}")
                else:
                    mispred_ws.write(i, 1, "Image not found\n\n\n\n", normal_border_format)
                    print(f"DEBUG: No pre-processed image found for: {detection['image_name']}")
                mispred_ws.write(i, 2, detection["category"]+ "\n\n\n\n", normal_border_format)
                category = detection["category"]
                category_split = category.split("_")
                mispred_ws.write(i, 3, category_split[1], normal_border_format)
                mispred_ws.write(i, 4, category_split[2], normal_border_format)
                mispred_ws.write(i, 5, category_split[3], normal_border_format)
                mispred_ws.write(i, 6, detection.get("gt_class_name", "unknown")+ "\n\n\n\n", normal_border_format)    
                mispred_ws.write(i, 7, detection["class_name"]+ "\n\n\n\n", normal_border_format)
                mispred_ws.write(i, 8, str(detection.get("confidence_score", "N/A"))+ "\n\n\n\n", normal_border_format)
                
                # Handle different gt_bbox formats (list vs "N/A")
                gt_bbox_value = detection.get("gt_bbox", "N/A")
                if isinstance(gt_bbox_value, list):
                    gt_bbox_str = str(gt_bbox_value)
                else:
                    gt_bbox_str = str(gt_bbox_value)  # For "N/A" case
                    
                mispred_ws.write(i, 9, gt_bbox_str + "\n\n\n\n", normal_border_format)
                mispred_ws.write(i, 10, str(detection["pred_bbox"])+ "\n\n\n\n", normal_border_format)
                
                
        
        # 3. Missed Detections (False Negatives)
        if missed_detections:
            missed_detections_excel_file="Misdetection_analysis.xlsx"
            # Create a new workbook
            output_path = f"{timestamp_folder}/{missed_detections_excel_file}"
            missed_detections_workbook = xlsxwriter.Workbook(output_path)            
            missed_ws = missed_detections_workbook.add_worksheet("Missed Detections")
            headers = ["Image Name", "Image      ","Category","Product","Lighting","Distance", "Class", "Ground Truth Box"]
            
            for col, header in enumerate(headers):
                missed_ws.write(0, col, header, header_format)
            
            for i, detection in enumerate(missed_detections, 1):
                image_name = get_image_extension( detection["image_name"])
                if image_name is None:
                    print(f"Warning: Could not find image for {detection['image_name']}")
                    image_name = detection["image_name"]  # Fallback to original name
                    image_path = None
                else:
                    image_path = "groundtruth_testing_data/data/original images/"+ image_name
                
                missed_ws.write(i, 0,image_name + "\n\n\n\n", normal_border_format)
                
                # FIXED: Use pre-processed image with bounding boxes to avoid double detection
                if detection["image_name"] in processed_images:
                    image_path = processed_images[detection["image_name"]]
                    x_scale, y_scale = get_scale(image_path)
                    missed_ws.insert_image(i,1,image_path,{"x_scale":x_scale,"y_scale":y_scale,"align":"center","valign":"vcenter"})
                    print(f"DEBUG: Used pre-processed image for missed detection: {image_name}")
                else:
                    missed_ws.write(i, 1, "Image not found\n\n\n\n", normal_border_format)
                    print(f"DEBUG: No pre-processed image found for: {detection['image_name']}")
                
                missed_ws.write(i, 2, detection["category"] + "\n\n\n\n", normal_border_format)
                category = detection["category"]
                category_split = category.split("_")
                missed_ws.write(i, 3, category_split[1], normal_border_format)
                missed_ws.write(i, 4, category_split[2], normal_border_format)
                missed_ws.write(i, 5, category_split[3], normal_border_format)
                missed_ws.write(i, 6, detection["class_name"] + "\n\n\n\n", normal_border_format)
                missed_ws.write(i, 7, str(detection["gt_bbox"]) + "\n\n\n\n", normal_border_format)# Auto-adjust column widths
        if missed_detections_workbook is not None:
            for worksheet in missed_detections_workbook.worksheets():
                worksheet.autofit()
        if mispredictions_workbook is not None:
            for worksheet in mispredictions_workbook.worksheets():
                worksheet.autofit()
        if correct_detection_workbook is not None:
            for worksheet in correct_detection_workbook.worksheets():
                worksheet.autofit()
        for worksheet in summary_workbook.worksheets():
            worksheet.autofit()
    
    except Exception as e:
        print(e)
        traceback.print_exc()
    
    finally:
        # Close the workbooks safely
        try:
            summary_workbook.close()
        except:
            pass
        try:
            if correct_detection_workbook is not None:
                correct_detection_workbook.close()
        except:
            pass
        try:
            if missed_detections_workbook is not None:
                missed_detections_workbook.close()
        except:
            pass
        try:
            if mispredictions_workbook is not None:
                mispredictions_workbook.close()
        except:
            pass
    
    return timestamp_folder

# Run the analysis
def calculate_metrics(groundtruth_details, prediction_details, data_df, iou_threshold):
    """
    Calculate comprehensive metrics and generate Excel reports.
    Handles edge cases with empty data gracefully.
    """
    try:
        # Validate inputs
        if not groundtruth_details:
            print("Error: No ground truth details provided!")
            raise ValueError("groundtruth_details cannot be empty")
        
        if not isinstance(groundtruth_details, list):
            print("Error: groundtruth_details must be a list!")
            raise TypeError("groundtruth_details must be a list")
        
        if not isinstance(prediction_details, list):
            print("Warning: prediction_details is not a list, converting to empty list")
            prediction_details = []
        
        if data_df is None:
            print("Warning: data_df is None, creating empty DataFrame")
            data_df = pd.DataFrame({'image': [], 'rawtraining': [], 'product': [], 'lighting': [], 'distance': []})
        
        print(f"Input validation completed:")
        print(f"  - Ground truth entries: {len(groundtruth_details)}")
        print(f"  - Prediction entries: {len(prediction_details)}")
        print(f"  - Metadata entries: {len(data_df) if not data_df.empty else 0}")
        
        # Add category information to data
        print("Adding category information to data...")
        groundtruth_details = add_category_to_data(groundtruth_details, data_df) 
        prediction_details = add_category_to_data(prediction_details, data_df) 
        
        print("Analyzing detections...")
        correct_detections, mispredictions, missed_detections = analyze_detections(
            groundtruth_details, prediction_details, iou_threshold
        )
        
        print(f"Detection analysis completed:")
        print(f"  - Correct detections: {len(correct_detections)}")
        print(f"  - Mispredictions: {len(mispredictions)}")
        print(f"  - Missed detections: {len(missed_detections)}")

        print("Calculating metrics by category...")
        category_metrics = calculate_metrics_by_category(correct_detections, mispredictions, missed_detections)

        print("Calculating class-wise metrics...")
        class_wise_metrics = calculate_metrics_by_class(groundtruth_details, correct_detections, mispredictions, missed_detections)
        
        print("Calculating overall class metrics...")
        class_metrics = calculate_metrics_for_overall_class(correct_detections, mispredictions, missed_detections)

        print("Exporting results to Excel...")
        report_folder = export_to_excel(
            groundtruth_details, prediction_details, correct_detections, 
            mispredictions, missed_detections, category_metrics, class_metrics, class_wise_metrics
        )
        print(f"Results exported to: {report_folder}")
        
        return report_folder
        
    except Exception as e:
        print(f"Error in calculate_metrics: {e}")
        traceback.print_exc()
        raise