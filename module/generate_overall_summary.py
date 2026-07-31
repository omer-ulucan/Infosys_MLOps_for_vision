'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import xlsxwriter
import traceback
import os

def export_to_summary_excel(report_data, report_path, header_list, row_limit=60):
    excel_path = os.path.join(report_path, "Overall_summary.xlsx")
    summary_workbook = xlsxwriter.Workbook(excel_path)
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
        
        overall_metrics = report_data['overall_metrics']
        class_wise_metrics = report_data['class_wise_metrics']
        category_wise_metrics = report_data['category_wise_metrics']
        class_category_wise_metrics = report_data['class_category_wise_metrics']
        
        # Calculate overall statistics
        stats = [
            ("Total Ground Truth Images", overall_metrics['data size']),
            ("Total Predictions", overall_metrics['data size']),
            ("Correct Detections (True Positives)", overall_metrics['true positive']),
            ("Mispredictions (False Positives)", overall_metrics['false positive']),
            ("Missed Detections (False Negatives)", overall_metrics['false negative']),
            ("True Negatives", overall_metrics['true negative'])
        ]
        
        # Write overall statistics
        for i, (label, value) in enumerate(stats, 5):
            summary_ws.write(f"A{i}", label, normal_border_format)
            summary_ws.write(f"B{i}", value, normal_border_format)
        
        metric_stats = [
            ("Overall Precision", f"{overall_metrics['precision']:.2f}"),
            ("Overall Recall", f"{overall_metrics['recall']:.4f}"),
            ("Overall F1 Score", f"{overall_metrics['f1']:.4f}"),
            ("Overall Accuracy", f"{overall_metrics['accuracy']:.4f}"),
            
            ("Total Classes", len(report_data['class_wise_metrics'])),
        ]
        
        # Write overall metrics
        # last_row = len(stats) + 4
        last_row = len(stats) + 3

        for i, (label, value) in enumerate(metric_stats, last_row + 1):
            summary_ws.write(f"A{i}", label, normal_border_format)
            summary_ws.write(f"B{i}", value, normal_border_format)
        
        # Add category metrics section
        category_row = last_row + len(metric_stats) + 2
        summary_ws.merge_range(f"A{category_row}:G{category_row}", "Metrics by GT Category", header_format)
        
        # Category metrics headers
        headers = ["Category"] + header_list + list(next(iter(category_wise_metrics.values())).keys())[1:]
        # headers = ["Category", "Product", "Lighting", "Distance", "True Positives", "False Positives","False Negatives", "True Negatives", "Precision", "Recall", "F1","Accuracy"]
        for col, header in enumerate(headers):
            summary_ws.write(category_row + 1, col, header, header_format)
    

    # Write category metrics
        for i, (category, metrics) in enumerate(sorted(category_wise_metrics.items()), category_row + 2):
            summary_ws.write(i, 0, category, normal_border_format)
            category_split = category.split("_")
            
            # Write each part of the split category to the worksheet
            for j, part in enumerate(category_split):
                summary_ws.write(i, j + 1, part, normal_border_format)
            
            # Write the metrics to the worksheet
            summary_ws.write(i, len(category_split) + 1, metrics["true positive"], normal_border_format)
            summary_ws.write(i, len(category_split) + 2, metrics["false positive"], normal_border_format)
            summary_ws.write(i, len(category_split) + 3, metrics["false negative"], normal_border_format)
            summary_ws.write(i, len(category_split) + 4, metrics["true negative"], normal_border_format)
            summary_ws.write(i, len(category_split) + 5, f"{metrics['precision']:.4f}", normal_border_format)
            summary_ws.write(i, len(category_split) + 6, f"{metrics['recall']:.4f}", normal_border_format)
            summary_ws.write(i, len(category_split) + 7, f"{metrics['f1']:.4f}", normal_border_format)
            summary_ws.write(i, len(category_split) + 8,f"{metrics['accuracy']:.4f}",normal_border_format)
        # Add class metrics section
        class_row = category_row + len(category_wise_metrics) + 4

        summary_ws.merge_range(f"A{class_row}:G{class_row}", "Metrics by Class", header_format)
        
        # Class metrics headers
        headers = ["Class"] + list(next(iter(category_wise_metrics.values())).keys())[1:]
        # headers = ["Class", "True Positives", "False Positives", "False Negative", "True Negative", "Precision", "Recall", "F1","Accuracy"]
        for col, header in enumerate(headers):
            summary_ws.write(class_row + 1, col, header, header_format)
        
        # Write class metrics
        for i, (class_name, metrics) in enumerate(sorted(class_wise_metrics.items()), class_row + 2):
            
            summary_ws.write(i, 0, class_name, normal_border_format)
            summary_ws.write(i, 1, metrics["true positive"], normal_border_format)
            summary_ws.write(i, 2, metrics["false positive"], normal_border_format)
            summary_ws.write(i, 3, metrics["false negative"], normal_border_format)
            summary_ws.write(i, 4, metrics["true negative"], normal_border_format)            
            summary_ws.write(i, 5, f"{metrics['precision']:.2f}", normal_border_format)
            summary_ws.write(i, 6, f"{metrics['recall']:.4f}", normal_border_format)
            summary_ws.write(i, 7, f"{metrics['f1']:.4f}", normal_border_format)
            summary_ws.write(i,8,f"{metrics['accuracy']:.4f}",normal_border_format)

        for class_name,category_metrics in class_category_wise_metrics.items():
            # Create class worksheet
            class_ws = summary_workbook.add_worksheet(class_name)
            # Class metrics headers - matching the exact structure from existing Excel files
            headers = ["Category", "Product", "Lighting", "Distance", "num_of_images", "True Positives", "False Positives", "False Negatives", "Precision", "Recall", "F1"]
            for col, header in enumerate(headers):
                class_ws.write(0, col, header, header_format)
            i = 0
            for category,metrics in  category_metrics.items():
                i+=1
                class_ws.write(i, 0, category, normal_border_format)
                category_split = category.split("_")
            
                # Write each part of the split category to the worksheet
                for p, value in enumerate(category_split):
                    class_ws.write(i, p + 1, value, normal_border_format)
                
                # Write the metrics to the worksheet - exact 11 column structure
                class_ws.write(i, 4, metrics['data size'], normal_border_format)  # num_of_images
                class_ws.write(i, 5, metrics["true positive"], normal_border_format)  # True Positives
                class_ws.write(i, 6, metrics["false positive"], normal_border_format)  # False Positives
                class_ws.write(i, 7, metrics["false negative"], normal_border_format)  # False Negatives
                class_ws.write(i, 8, f"{metrics['precision']:.4f}", normal_border_format)  # Precision
                class_ws.write(i, 9, f"{metrics['recall']:.4f}", normal_border_format)  # Recall
                class_ws.write(i, 10, f"{metrics['f1']:.4f}", normal_border_format)  # F1
        

    except Exception as e:
        print(e)
        traceback.print_exc()

    finally:
        # Close the workbook
        summary_workbook.close()
        