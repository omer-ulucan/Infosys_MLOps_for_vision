'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import numpy as np
import pickle
from PIL import Image
import os
from datetime import datetime

import ssl

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler


class Google_vit_model:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, ViTModel
        if torch.cuda.is_available() == False:
            devices = torch.cuda.device_count()
            print(f"devices available : {devices} ")
            current_device = torch.cuda.current_device()
            print(f"device : {current_device} ")
            self.device = torch.device(current_device)
        else:
            self.device = "cpu"
        print(f"Device used : {self.device}")
        #self.device = torch.device("cpu")
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        
        #os.environ['HF_HUB_OFFLINE']='1'
        self.processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k",use_fast = True)
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(self.device)
 
    def extract_features(self, image,aggregation="mean"):
        
        start_time = datetime.now()
        processed_image = self.processor(image, return_tensors="pt",device = self.device).to(self.device)
        query_features = self.model(**processed_image).last_hidden_state
        query_features_np = query_features.detach().cpu().numpy().squeeze()
        #query_features_np = query_features.detach().numpy().squeeze()
        if aggregation == "max":
            aggregated_features = np.max(query_features_np, axis=0)
        elif aggregation == "mean":
            aggregated_features = np.mean(query_features_np, axis=0)
        aggregated_features = np.array(aggregated_features).astype(np.float32).reshape(1, -1)
        end_time = datetime.now()
        duartion = end_time - start_time
        print("timetaken for extracting features:",duartion )
        return aggregated_features.flatten()
 
    
    def save_template_embeddings(self, features, file_name, output_path, file_type="pickle"):
        self.final_output_path = output_path + "/"+file_name
        combined_features = []

        # Check if the pickle file already exists
        if os.path.exists(self.final_output_path+".pkl"):
            try:
                with open(self.final_output_path+".pkl", 'rb') as f:
                    existing_features = pickle.load(f)
                combined_features.extend(existing_features)
            except Exception as e:
                print(f"Error loading existing features: {e}")
 
        # Add new features to the combined list
        combined_features.extend(features)
        try:
            if file_type == "pickle":
                with open(self.final_output_path+".pkl", 'wb') as f:
                    pickle.dump(combined_features, f)
            else:
                np.save(self.final_output_path.replace('.pkl', '.npy'), combined_features)
            print(f"Combining pickle files {file_name}.pkl")
        except Exception as e:
            print(f"Error saving features: {e}")

    def load_template_embeddings(path):
        try:
            templateEmbeddings=[]
            picklefound = False         

            for path_val, sub_dirs_val, files_val in os.walk(path):
                for name in files_val:
                    if name.endswith('pkl'):  
                        with open(os.path.join(path_val+ "/"+name), 'rb') as file:
                            data = pickle.load(file)
                            templateEmbeddings.append({'name':name,'embeddings':data})
                            picklefound = True
                            print(f"Features loaded from pickle file {name}")
            if not picklefound:
                print("No template files found")
           
            return templateEmbeddings   
        except Exception as e:
            print(f"Error loading template embeddings: {e}")
    
    def get_template_images_by_knn(self,folder_path, num_images=20):
        absFolderPath =  folder_path

        image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_paths:
            print("No images found in the specified folder.")
            return []
        features = []
        for img_path in image_paths:
            try:
                img_data = Image.open(img_path).convert("RGB")
                img_feature = self.extract_features(img_data,"mean")
                print(type(img_feature))
                features.append(img_feature)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                image_paths.remove(img_path) #remove the bad image path
                continue
 
        if not features: #if no features were extracted
            print("No features could be extracted from the images.")
            return []
       
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
 
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=num_images, random_state=42, n_init='auto') #n_init avoids future warning
        kmeans.fit(scaled_features)
 
        # Select representative images from each cluster
        representative_images = []
        for cluster_label in range(num_images):
            cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]
            if len(cluster_indices) > 0:
                # Find the image closest to the cluster center
                cluster_center = kmeans.cluster_centers_[cluster_label]
                distances = np.linalg.norm(scaled_features[cluster_indices] - cluster_center, axis=1)
                closest_index = cluster_indices[np.argmin(distances)]
                representative_images.append(image_paths[closest_index])
            else:
                print(f"Cluster {cluster_label} is empty")
 
        return representative_images
 
