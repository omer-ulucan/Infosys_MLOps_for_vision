'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def draw_and_save_boxplot(valDatasetMetadataWithSimilarity, output_dir, filename1='similarity_score_boxplot.png', filename2='all_products_boxplot.png'):
    similarity_scores = {}
    all_scores = []

    for item in valDatasetMetadataWithSimilarity:
        product_name = item['actualClass']
        for prediction in item['predictions']:
            if product_name not in similarity_scores:
                similarity_scores[product_name] = []
            similarity_scores[product_name].append(prediction['similarity_score'])
            all_scores.append(prediction['similarity_score'])

    # Convert to DataFrame
    similarity_scores_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in similarity_scores.items()]))
    all_scores_df = pd.DataFrame({'All Products': all_scores})

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
    file_path1 = os.path.join(output_dir, filename1)
    file_path2 = os.path.join(output_dir, filename2)

    # Boxplot for individual products
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=similarity_scores_df)
    plt.title('Boxplot of Similarity Scores for Multiple Products')
    plt.xlabel('Products')
    plt.ylabel('Similarity Scores')

    # Adding data labels for count of products within each boxplot range (1st and 3rd quartile)
    for i, column in enumerate(similarity_scores_df.columns):
        column_data = similarity_scores_df[column].dropna()
        lower_limit = column_data.quantile(0.25)
        upper_limit = column_data.quantile(0.75)
        count = column_data[(column_data >= lower_limit) & (column_data <= upper_limit)].count()
        plt.text(i, column_data.median(), f'n={count}', ha='center', va='center', fontsize=10, color='black', weight='bold')

    plt.savefig(file_path1)
    plt.close()

    # Boxplot for all products combined
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=all_scores_df)
    plt.title('Boxplot of Similarity Scores for All Products')
    plt.xlabel('All Products')
    plt.ylabel('Similarity Scores')

    # Adding data labels for count of products within the boxplot range (1st and 3rd quartile)
    lower_limit_all = all_scores_df['All Products'].quantile(0.25)
    upper_limit_all = all_scores_df['All Products'].quantile(0.75)
    count_all = all_scores_df[(all_scores_df['All Products'] >= lower_limit_all) & (all_scores_df['All Products'] <= upper_limit_all)].count()
    plt.text(0, all_scores_df['All Products'].median(), f'n={count_all[0]}', ha='center', va='center', fontsize=10, color='black', weight='bold')

    plt.savefig(file_path2)
    plt.close()
