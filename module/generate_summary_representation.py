'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json



def load_and_prepare(filepath, header_list):
    df_raw = pd.read_excel(filepath, sheet_name="Summary", header=None)

    # Find the row where "Metrics by GT Category" starts
    start_row = df_raw[df_raw.iloc[:, 0] == "Category"].index[0] + 2

    # Find where it ends (usually where "Metrics by Class" starts)
    end_row = df_raw[df_raw.iloc[:, 0] == "Class"].index[0] - 2

    df_gt = df_raw.iloc[start_row:end_row].copy()
    
    # Debug: Print actual DataFrame shape and expected columns
    print(f"DataFrame shape: {df_gt.shape}")
    actual_columns = df_gt.shape[1]
    
    # Build expected column names
    expected_columns = ["Category"] + header_list + ["True Positive", "False Positive","False Negative","True Negative", "Precision", "Recall", "F1","Accuracy"]
    print(f"Expected columns ({len(expected_columns)}): {expected_columns}")
    print(f"Actual columns: {actual_columns}")
    
    # Adjust column assignment based on actual DataFrame structure
    if actual_columns == len(expected_columns):
        df_gt.columns = expected_columns
    elif actual_columns < len(expected_columns):
        # If we have fewer columns, truncate the expected columns
        df_gt.columns = expected_columns[:actual_columns]
        print(f"Warning: DataFrame has fewer columns than expected. Using: {df_gt.columns.tolist()}")
    else:
        # If we have more columns, pad with generic names
        extra_cols = [f"Extra_Col_{i}" for i in range(len(expected_columns), actual_columns)]
        df_gt.columns = expected_columns + extra_cols
        print(f"Warning: DataFrame has more columns than expected. Using: {df_gt.columns.tolist()}")
    
    df_gt = df_gt.reset_index(drop=True)
    
    # Try to find the distance column by different possible names
    distance_col = None
    for col in df_gt.columns:
        if col.lower() in ['distance', 'dist'] or 'distance' in col.lower():
            distance_col = col
            break
    
    if distance_col:
        df_gt = df_gt.dropna(subset=[distance_col])
        print(f"Filtered by distance column: {distance_col}")
    else:
        print("Warning: No distance column found, skipping distance filter")
        # Just drop rows where all values are NaN
        df_gt = df_gt.dropna(how='all')

    # Apply numeric conversion only to columns that exist and might be numeric
    numeric_cols = []
    for col in ["Precision", "Recall", "F1", "Accuracy"]:
        if col in df_gt.columns:
            numeric_cols.append(col)
    
    if numeric_cols:
        df_gt[numeric_cols] = df_gt[numeric_cols].apply(pd.to_numeric, errors="coerce")
        print(f"Applied numeric conversion to: {numeric_cols}")
    
    # Calculate F1 score if it's missing but we have Precision and Recall
    if 'F1' not in df_gt.columns and 'Precision' in df_gt.columns and 'Recall' in df_gt.columns:
        # Calculate F1 = 2 * (precision * recall) / (precision + recall)
        # Handle division by zero
        df_gt['F1'] = 2 * (df_gt['Precision'] * df_gt['Recall']) / (df_gt['Precision'] + df_gt['Recall'])
        df_gt['F1'] = df_gt['F1'].fillna(0)  # Replace NaN values with 0
        print("Calculated F1 score from Precision and Recall")
        
    # Calculate Accuracy if it's missing and we have the confusion matrix values
    if 'Accuracy' not in df_gt.columns and all(col in df_gt.columns for col in ['True Positive', 'False Positive', 'False Negative', 'True Negative']):
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        tp = pd.to_numeric(df_gt['True Positive'], errors='coerce').fillna(0)
        tn = pd.to_numeric(df_gt['True Negative'], errors='coerce').fillna(0)
        fp = pd.to_numeric(df_gt['False Positive'], errors='coerce').fillna(0)
        fn = pd.to_numeric(df_gt['False Negative'], errors='coerce').fillna(0)
        
        total = tp + tn + fp + fn
        df_gt['Accuracy'] = (tp + tn) / total
        df_gt['Accuracy'] = df_gt['Accuracy'].fillna(0)  # Replace NaN values with 0
        print("Calculated Accuracy from confusion matrix values")
    
    print(f"Final DataFrame shape: {df_gt.shape}")
    print(f"Final columns: {df_gt.columns.tolist()}")
    
    return df_gt


def plot_heatmap(df, metric, output_dir):
    # Check if required columns exist
    required_cols = ["Lighting", "Distance", metric]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Cannot create {metric} heatmap. Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Check if we have data
    if df.empty:
        print(f"Warning: DataFrame is empty, cannot create {metric} heatmap")
        return
        
    try:
        pivot = df.pivot_table(index="Lighting", columns="Distance", values=metric, aggfunc="mean")
        
        # Check if pivot table has data
        if pivot.empty:
            print(f"Warning: Pivot table is empty for {metric} heatmap")
            return
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
        plt.title(f"{metric} Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_heatmap.png"))
        plt.close()
        print(f"Successfully created {metric} heatmap")
        
    except Exception as e:
        print(f"Error creating {metric} heatmap: {e}")
        plt.close()  # Make sure to close the figure even if there's an error

def plot_f1_bar(df, output_dir):
    # Check if required columns exist
    required_cols = ["Product", "F1"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Cannot create F1 bar chart. Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Check if we have data
    if df.empty:
        print(f"Warning: DataFrame is empty, cannot create F1 bar chart")
        return
        
    try:
        avg_f1 = df.groupby("Product")["F1"].mean().reset_index()
        
        # Check if we have data after grouping
        if avg_f1.empty:
            print(f"Warning: No data after grouping by Product for F1 bar chart")
            return
            
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=avg_f1, x="Product", y="F1")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),  # Format the label
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),  # Distance from the top of the bar.
                        textcoords='offset points')
        plt.title(f"Average F1 Score by Product")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"f1_bar.png"))
        plt.close()
        print(f"Successfully created F1 bar chart")
        
    except Exception as e:
        print(f"Error creating F1 bar chart: {e}")
        plt.close()  # Make sure to close the figure even if there's an error

def plot_precision_bar(df, output_dir):
    # Check if required columns exist
    required_cols = ["Product", "Precision"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Cannot create Precision bar chart. Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Check if we have data
    if df.empty:
        print(f"Warning: DataFrame is empty, cannot create Precision bar chart")
        return
        
    try:
        avg_precision = df.groupby("Product")["Precision"].mean().reset_index()
        
        # Check if we have data after grouping
        if avg_precision.empty:
            print(f"Warning: No data after grouping by Product for Precision bar chart")
            return
            
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=avg_precision, x="Product", y="Precision")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),  # Format the label
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),  # Distance from the top of the bar.
                        textcoords='offset points')
        plt.title(f"Average Precision by Product")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"precision_bar.png"))
        plt.close()
        print(f"Successfully created Precision bar chart")
        
    except Exception as e:
        print(f"Error creating Precision bar chart: {e}")
        plt.close()  # Make sure to close the figure even if there's an error

def plot_recall_bar(df, output_dir):
    # Check if required columns exist
    required_cols = ["Product", "Recall"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Cannot create Recall bar chart. Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Check if we have data
    if df.empty:
        print(f"Warning: DataFrame is empty, cannot create Recall bar chart")
        return
        
    try:
        avg_recall = df.groupby("Product")["Recall"].mean().reset_index()
        
        # Check if we have data after grouping
        if avg_recall.empty:
            print(f"Warning: No data after grouping by Product for Recall bar chart")
            return
            
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=avg_recall, x="Product", y="Recall")
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.2f'),  # Format the label
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9),  # Distance from the top of the bar.
                        textcoords='offset points')
        plt.title(f"Average Recall by Product")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"recall_bar.png"))
        plt.close()
        print(f"Successfully created Recall bar chart")
        
    except Exception as e:
        print(f"Error creating Recall bar chart: {e}")
        plt.close()  # Make sure to close the figure even if there's an error


def generate_summary_graph(output_report_path, val_config):



    # output_dir = "OD_Merged_Comparison_Charts"
    output_dir = os.path.join(output_report_path, "Merged_Comparison_Charts")
    os.makedirs(output_dir, exist_ok=True)

    summary_stats = []

    # for dataset_name, filepath in summary_files.items():
    header_list = val_config['report_generator']['gt_category_headers']
    
    # Try both possible filenames (Overall_summary.xlsx and overall_summary.xlsx)
    possible_filenames = ["Overall_summary.xlsx", "overall_summary.xlsx"]
    file_path = None
    
    for filename in possible_filenames:
        potential_path = os.path.join(output_report_path, filename)
        if os.path.exists(potential_path):
            file_path = potential_path
            print(f"Found summary file: {filename}")
            break
    
    # If not found, try to find any Excel file containing "summary"
    if file_path is None and os.path.exists(output_report_path):
        print("Primary summary files not found, searching for any files containing 'summary'...")
        for file in os.listdir(output_report_path):
            if file.lower().endswith('.xlsx') and 'summary' in file.lower():
                file_path = os.path.join(output_report_path, file)
                print(f"Found alternative summary file: {file}")
                break
    
    if file_path is None:
        print(f"Available files in {output_report_path}:")
        if os.path.exists(output_report_path):
            for file in os.listdir(output_report_path):
                print(f"  - {file}")
        else:
            print("  Directory does not exist")
        raise FileNotFoundError(f"Could not find summary Excel file in {output_report_path}. Looked for: {possible_filenames}")
    
    try:
        df_gt = load_and_prepare(file_path, header_list)
        print(f"Successfully loaded data with shape: {df_gt.shape}")
        print(f"Columns: {df_gt.columns.tolist()}")
        
        # Generate individual metric plots
        available_metrics = []
        for metric in ["Precision", "Recall", "F1", "Accuracy"]:
            if metric in df_gt.columns:
                available_metrics.append(metric)
        
        print(f"Available metrics for heatmaps: {available_metrics}")
        
        # Generate heatmaps for available metrics (prioritize Precision and F1)
        priority_metrics = ["Precision", "F1", "Recall", "Accuracy"]
        for metric in priority_metrics:
            if metric in available_metrics:
                plot_heatmap(df_gt, metric, output_dir)

        # Generate F1 bar chart (or fallback to Precision if F1 not available)
        if "F1" in df_gt.columns:
            plot_f1_bar(df_gt, output_dir)
        elif "Precision" in df_gt.columns:
            plot_precision_bar(df_gt, output_dir)  # Create a new function for Precision bar chart
        elif "Recall" in df_gt.columns:
            plot_recall_bar(df_gt, output_dir)  # Create a new function for Recall bar chart
        
        print("Individual charts completed")
        
    except Exception as load_error:
        print(f"Error loading or processing data: {load_error}")
        print("Skipping chart generation due to data loading issues")
        return
    
    # Generate overall comparison plots (only if we have valid data)
    try:
        # Summary stats for overall comparison
        df_overall = pd.read_excel(file_path, sheet_name="Summary", header=None)
        
        # Check if we can find the "Statistic" section
        statistic_rows = df_overall[df_overall[0] == "Statistic"].index
        if len(statistic_rows) == 0:
            print("Warning: Could not find 'Statistic' section in Excel file. Skipping overall comparison plots.")
            return
            
        summary_section = statistic_rows[0] + 1
        overall_metrics = df_overall.iloc[summary_section:summary_section + 10, [0, 1]].dropna()
        
        if overall_metrics.empty:
            print("Warning: No overall metrics found. Skipping overall comparison plots.")
            return
            
        overall_metrics.columns = ["Metric", output_dir]
        summary_stats = [overall_metrics.set_index("Metric")]

        # =============================
        # OVERALL COMPARISON PLOTS
        # =============================
        df_combined = pd.concat(summary_stats, axis=1).dropna()
        
        # Check if we have the expected metrics
        expected_metrics = ["Overall Precision", "Overall Recall", "Overall F1 Score", 
                           "Correct Detections (True Positives)", "Mispredictions (False Positives)", 
                           "Missed Detections (False Negatives)", "True Negatives"]
        
        available_metrics = [metric for metric in expected_metrics if metric in df_combined.index]
        
        if not available_metrics:
            print("Warning: None of the expected overall metrics found. Skipping overall comparison plots.")
            print(f"Available metrics: {df_combined.index.tolist()}")
            return
            
        df_combined = df_combined.loc[available_metrics]
        df_combined = df_combined.astype(float).reset_index().melt(id_vars="Metric", var_name="Dataset", value_name="Value")

        # Plot 1: Metric-wise Comparison (only for Overall metrics)
        overall_data = df_combined[df_combined["Metric"].str.contains("Overall")]
        if not overall_data.empty:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=overall_data, x="Metric", y="Value", hue="Dataset")
            for p in ax.patches:
                if p.get_height() > 0:  # Only annotate if height is positive
                    ax.annotate(format(p.get_height(), '.2f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center',
                                xytext=(0, 9),
                                textcoords='offset points')
            plt.title("Overall Precision, Recall, F1 by Dataset")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "overall_precision_recall_f1_comparison.png"))
            plt.close()
            print("Successfully created overall metrics comparison chart")

        # Plot 2: Error Distribution
        error_data = df_combined[df_combined["Metric"].str.contains("Positives") | df_combined["Metric"].str.contains("Negatives")]
        if not error_data.empty:
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=error_data, x="Metric", y="Value", hue="Dataset")
            for p in ax.patches:
                if p.get_height() > 0:  # Only annotate if height is positive
                    ax.annotate(format(p.get_height(), '.2f'),
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center',
                                xytext=(0, 9),
                                textcoords='offset points')
            plt.title("TP, FP, FN, TN Counts by Dataset")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "error_distribution_comparison.png"))
            plt.close()
            print("Successfully created error distribution chart")

        print(f"All charts saved to: {os.path.abspath(output_dir)}")
        
    except Exception as chart_error:
        print(f"Error generating overall comparison charts: {chart_error}")
        print("Individual charts may have been created successfully")