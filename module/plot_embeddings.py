'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import pickle
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import umap.umap_ as umap
import umap.plot

def read_pkl_file(file_path):

    try:
        with open(file_path, 'rb') as f:  # 'rb' for read binary
            loaded_object = pickle.load(f)
        return loaded_object
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading pickle file '{file_path}': {e}")
        return None

def get_pkl_file_paths(folder_path):
    try:
        # Using glob for a more concise way.
        pkl_files = glob.glob(os.path.join(folder_path, "*.pkl"))
        return pkl_files

    except Exception as e:
        print(f"Error getting .pkl file paths: {e}")
        return []

def plot_pca_from_dict_list(data_list, n_components=2):
    """
    Plots PCA from a list of dictionaries where:
    - 'originalClass' is the class label
    - 'embeddings' is the feature vector

    Args:
        data_list (list of dict): List of dictionaries with 'originalClass' and 'embeddings'.
        n_components (int): Number of principal components to keep (2 or 3).
    """

    embeddings = []
    classes = []

    for item in data_list:
        embeddings.append(item['embeddings'])
        classes.append(item['originalClass'])

    embeddings = np.array(embeddings)
    classes = np.array(classes)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(embeddings)
    return principal_components,classes

import plotly.graph_objects as go
import pandas as pd

def plot_plotly_3d_pca(principal_components, classes, save_path=None):
    """
    Plots an interactive 3D PCA scatter plot using plotly with legend.

    Args:
        principal_components (numpy.ndarray): The principal components from PCA.
        classes (list or numpy.ndarray): The original class names.
        save_path (str, optional): Path to save the HTML file.
    """

    unique_classes = list(set(classes))
    class_codes = pd.Categorical(classes).codes

    data = go.Scatter3d(
        x=principal_components[:, 0],
        y=principal_components[:, 1],
        z=principal_components[:, 2],
        mode='markers',
        marker=dict(
            color=class_codes,
            colorscale='viridis',
            colorbar=dict(title='Class Codes') # Add colorbar
        ),
        text=classes,  # Hover text
        hovertemplate="Class: %{text}<extra></extra>" # remove extra info from hover
    )

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        title='Interactive 3D PCA',
        scene=dict(
            xaxis=dict(title='Principal Component 1'),
            yaxis=dict(title='Principal Component 2'),
            zaxis=dict(title='Principal Component 3')
        )
    )

    fig = go.Figure(data=[data], layout=layout)

    # Add legend (using colorbar as a proxy)
    # fig.update_layout(
    #     coloraxis_colorbar=dict(
    #         tickvals=list(range(len(unique_classes))),
    #         ticktext=unique_classes,
    #         title='Classes'
    #     )
    # )
    # Add legend (using colorbar as a proxy)
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickvals=list(range(len(unique_classes))),
            ticktext=unique_classes,
            title='Classes'
        )
    )

    if save_path:
        fig.write_html(save_path)
       # print(f"Plot saved to: {save_path}")
    else:
        fig.show()

# Example usage:
# plot_plotly_3d_pca(principal_components, classes, "pca_3d.html")

# Example Usage (replace with your data):
# Assuming 'principal_components' and 'classes' are available
# plot_interactive_3d_pca(principal_components, classes)

import os

def remove_image_extension(filename):
    image_extensions = ['.png', '.jpg', '.jpeg']  # Add more if needed.

    for ext in image_extensions:
        if filename.lower().endswith(ext): #make case insensitive
            return filename[:-len(ext)] #remove the extension
    return filename #return the original filename if no extension is found.



def get_template_data(template_folder):
    data=[]
    pkl_paths = get_pkl_file_paths(template_folder)
    for path in pkl_paths:
        data+=read_pkl_file(path)
    return data

def plot_embeddings_in_3d(report_path,file_name,predicted_embedding,template_data):
    image_file_name = remove_image_extension(file_name)
    output_folder_path = os.path.join(report_path,"3d_plot")
    os.makedirs(output_folder_path,exist_ok=True)
    output_path=os.path.join(output_folder_path )    
    # plot_pca_from_dict_list(data, n_components=2)
    pred_data = {}
    pred_data['embeddings']=predicted_embedding
    pred_data['originalClass'] = image_file_name
    template_data.append(pred_data)
    pca_plot_data = plot_pca_from_dict_list(template_data, n_components=3)
    pca = pca_plot_data[0]
    classes = pca_plot_data[1]
    plot_path = os.path.join(output_path,f"{image_file_name}.html")
    plot_plotly_3d_pca(pca, classes,plot_path )
    return plot_path


# def plot_embeddings_in_3d(report_path, file_name, predicted_embedding, template_data):
#     # Remove image extension
#     image_file_name = remove_image_extension(file_name)
    
#     # Create output folder
#     output_folder_path = os.path.join(report_path, "3d_plot")
#     os.makedirs(output_folder_path, exist_ok=True)
    
#     # Append new data
#     pred_data = {
#         'embeddings': predicted_embedding,  # list is fine here
#         'originalClass': image_file_name
#     }
#     template_data.append(pred_data)
    
#     # Convert to DataFrame
#     df = pd.DataFrame(template_data)
    
#     # Extract embeddings and apply PCA
#     embeddings_matrix = pd.DataFrame(df['embeddings'].tolist())
#     pca = PCA(n_components=3)
#     reduced_embeddings = pca.fit_transform(embeddings_matrix)
    
#     # Create new DataFrame for plotting
#     df_plot = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z'])
#     df_plot['originalClass'] = df['originalClass']
#     if 'product' in df.columns:
#         df_plot['product'] = df['product']
    
#     # Plot 3D scatter
#     fig = px.scatter_3d(
#         df_plot,
#         x='x', y='y', z='z',
#         color='product' if 'product' in df_plot.columns else 'originalClass',
#         hover_data=['originalClass'],
#         title="3D PCA Embeddings"
#     )
    
#     # Save HTML
#     plot_path = os.path.join(output_folder_path, f"{image_file_name}.html")
#     fig.write_html(plot_path)
    
#     return plot_path



def plot_embeddings_in_3d(report_path, file_name, predicted_embedding, template_data,
                          metric='cosine',
                          n_neighbors=15,
                          min_dist=0.0,
                          random_state=42,
                          n_jobs=None):
    """
    Reduce embeddings to 3D with UMAP and save an interactive Plotly 3D scatter as HTML.


    Args:
        report_path (str): Base output folder.
        file_name (str): Name of the current file (may include extension).
        predicted_embedding (list/array): New embedding to append.
        template_data (list[dict]): Existing rows with keys like 'embeddings', 'originalClass', optional 'product'.
        metric (str): UMAP distance metric ('cosine' often best for deep embeddings).
        n_neighbors (int): UMAP neighborhood size (10–50 typical).
        min_dist (float): UMAP min_dist (0.0–0.5; smaller = tighter clusters).
        random_state (int or None): Seed for reproducibility; set None for faster parallel runs.
        n_jobs (int or None): Parallel jobs for UMAP. Use -1 for all cores. Only effective if random_state=None.


    Returns:
        str: Path to the saved Plotly HTML file.
    """


    # Local helper: remove image extension (if your project already defines this, you can delete this local one)
    def remove_image_extension(name: str) -> str:
        return os.path.splitext(name)[0]


    # Strip extension for output file naming
    image_file_name = remove_image_extension(file_name)


    # Create output folder
    output_folder_path = os.path.join(report_path, "3d_plot")
    os.makedirs(output_folder_path, exist_ok=True)


    # Append new row (embedding + class label)
    pred_data = {
        'embeddings': predicted_embedding,
        'originalClass': image_file_name
    }
    template_data.append(pred_data)


    # Convert to DataFrame
    df = pd.DataFrame(template_data)


    # Build the embeddings matrix (rows = samples, cols = features)
    embeddings_matrix = pd.DataFrame(df['embeddings'].tolist())


    # UMAP reducer: 3D for visualization
    # Note: If you set random_state (seed), UMAP will run single-threaded internally.
    # For parallel speed, set random_state=None and n_jobs=-1 (or desired core count).
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs if random_state is None else 1
    )
    reduced_embeddings = reducer.fit_transform(embeddings_matrix)


    # Prepare plotting DataFrame
    df_plot = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z'])
    df_plot['originalClass'] = df['originalClass'].astype(str)
    color_col = 'originalClass'  # default color
    if 'product' in df.columns:
        df_plot['product'] = df['product'].astype(str)
        color_col = 'product'


    # Plotly 3D scatter (interactive)
    fig = px.scatter_3d(
        df_plot,
        x='x', y='y', z='z',
        color=color_col,
        hover_data=['originalClass'],
        title="3D UMAP Embeddings"
    )


    # Save HTML
    plot_path = os.path.join(output_folder_path, f"{image_file_name}.html")
    fig.write_html(plot_path)


    return plot_path