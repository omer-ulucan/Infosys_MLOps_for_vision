'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def remove_double_count(pixel_annot):
    neigh = NearestNeighbors(n_neighbors=2)  # edit with respected n_neighbours

    print("Original annotations count = ", len(pixel_annot))
    # Storing labels,conf,bbox into the respective index of respective lists
    bboxes = []
    conf = []
    label = []
    for annot in pixel_annot:
        label.append(annot[-1])
        bboxes.append(annot[:4])
        conf.append(annot[-2])

    # Creating dataframe on bbox,labels,conf
    df = pd.DataFrame(bboxes, columns=['x', 'y', 'x2', 'y2'])
    df['conf'] = conf
    df['labels'] = label
    print(df.shape)
    if df.shape[0] >= 2:
        # Applying n-neigbors by taking n=2 and getting distances between nearest neighbors
        nbrs = neigh.fit(df[['x', 'y']])
        distances, indices = nbrs.kneighbors(df[['x', 'y']])

        # storing distances in distance list by extracting from 2D np array
        distances_list = distances.tolist()
        lis = lambda a: a[1]
        distances_list = [lis(i) for i in distances_list]

        # Creating a dictionary where key is distance and values are nested lists containing indices whose distnace
        # is equal to key
        indices_dict = {}
        for i in range(len(distances_list)):
            if distances_list[i] not in indices_dict:
                indices_dict[distances_list[i]] = [list(indices[i])]
            else:
                indices_dict[distances_list[i]].append(list(indices[i]))

        # removing duplicate list of indices from value
        # For Example here duplicate indicates [3,4] and [4,3] bothe indices represents same points
        for ind in indices_dict.values():
            if len(ind) > 1:
                rev_val = ind[0][-1::-1]
                if rev_val in ind:
                    ind = ind.remove(rev_val)

        # taking distance between points less than 45
        less_dist = [i for i in indices_dict.keys() if i < 45]

        # getting the indices of lesser distance points
        indices_less_dist = [indices_dict[i][0] for i in less_dist]

        # creating 2 lists , common_indices contains nested lists having atleat one common index. and
        # indices_less_dist_updated remaining list of indices
        common_indices = []
        for i in range(len(indices_less_dist) - 1):
            for j in range(i + 1, len(indices_less_dist)):
                a = indices_less_dist[i]
                b = indices_less_dist[j]
                if any(set(a) & set(b)):
                    common_indices.append([a, b])
        c_i = [j for i in common_indices for j in i]
        # print(c_i)
        indices_less_dist_updated = [i for i in indices_less_dist if i not in c_i]
        # print(common_indices[0:])
        # print(indices_less_dist_updated)

        # Extracting the index of point which have less confidence than the other points and storing in a list to_remove
        to_remove = []
        for i in indices_less_dist_updated:
            ser = df.iloc[i]['conf']

            to_remove.append(ser.idxmin())
        # print(to_remove)

        # Checking in the common_indices,
        labl = []
        if len(common_indices) != 0:
            for i in common_indices:
                for j in i:
                    labl.extend(df.iloc[j]['labels'].values.tolist())

            # if all the labels are not same of the common indices we will extract minimum conf index
            # from each nested list and store in ro_remove list
            if len(set(labl)) > 1:
                for i in common_indices:
                    for j in i:
                        ser = df.iloc[j]['conf']

                        to_remove.append(ser.idxmin())
            # if all the indices have same label here except the max conf index , remaining index will add into
            # to_remove list
            else:

                result_list = []
                for inner_list in common_indices:
                    single_list = []
                    for inner_l in inner_list:
                        single_list.extend(inner_l)
                    result_list.append(list(set(single_list)))

                    # new_list = []
                    # new_list.extend(inner_list)
                    print(result_list)

                for i in result_list:
                    max_c = df.iloc[i]['conf'].idxmax()
                    # print(max_c)
                    # print([j for j in i if j!=max_c])
                    to_remove.extend([j for j in i if j != max_c])

        print(to_remove)
    else:
        to_remove = []
    # Creating a new data frame by removing the to_remove indices

    df_new = df.copy()
    if len(to_remove) > 0:
        df_new.drop(df.index[to_remove], inplace=True)
    df_new.reset_index(drop=True).shape
    return df_new.values.tolist()
