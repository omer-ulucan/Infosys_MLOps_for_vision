'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from src.bounding_box import BoundingBox
from src.utils.enumerators import (BBFormat, CoordinatesType,
                                   MethodAveragePrecision)


def calculate_ap_every_point(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def calculate_ap_11_point_interp(rec, prec, recall_vals=11):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, recall_vals)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / len(recallValues)
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]


def get_pascalvoc_metrics(gt_boxes,
                          det_boxes,
                          data_df,
                          iou_threshold=0.3,
                          method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                          generate_table=False):
    """Get the metrics used by the VOC Pascal 2012 challenge.
    Args:
        boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
        bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP
        (dget_pascalvoc_metricsns:
        A dictioanry contains information and metrics of each class.
        The key represents the class and the values are:
        dict['class']: class representing the current dictionary;
        dict['precision']: array with the precision values;
        dict['recall']: array with the recall values;
        dict['AP']: average precision;
        dict['interpolated precision']: interpolated precision values;
        dict['interpolated recall']: interpolated recall values;
        dict['total positives']: total number of ground truth positives;
        dict['total TP']: total number of True Positive detections;
        dict['total FP']: total number of False Positive detections;"""
    # ret = {}
    # Get classes of all bounding boxes separating them by classes
    gt_classes_only = []
    classes_bbs = {}
    for bb in gt_boxes:
        c = bb.get_class_id()
        gt_classes_only.append(c)
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['gt'].append(bb)
    gt_classes_only = list(set(gt_classes_only))
    for bb in det_boxes:
        c = bb.get_class_id()
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['det'].append(bb)
    # Precision x Recall is obtained individually by each class
    class_res = []

    for c, v in classes_bbs.items():
        ret = {}
        print(f"\n\nc, v: {c, v}\n")
        # Report results only in the classes that are in the GT
        if c not in gt_classes_only:
            continue

        test_dict = {"image": [], "num_obj": []}
        gt_per_image = Counter([bb.get_image_name() for bb in gt_boxes if bb.get_class_id() == c])
        # print(f"gt_per_image: {gt_per_image}")
        for key, val in gt_per_image.items():
            test_dict["image"].append(key)
            test_dict["num_obj"].append(val)
        # print(f'test_dict:{test_dict}')

        test_df = pd.DataFrame(test_dict)
        merged_df = pd.merge(data_df, test_df, on="image")
        report_columns = list(merged_df.columns.values)
        report_columns = [e for e in report_columns if e not in ('image', 'num_obj')]
        cat_df = merged_df.groupby(report_columns).sum(
            'num_obj').reset_index()
        print('cat_df', cat_df)
        # cat_df.to_excel(c + 'cat_df.xlsx')
        # merged_df.to_excel(c + 'merged_df.xlsx')
        for index, row in cat_df.iterrows():
            npos = row['num_obj']
            print(f"npos: {npos}")

            final_val = []
            name_val = []
            for col in report_columns:
                val = (merged_df[col] == row[col])
                final_val.append(val)
                name_val.append(row[col])

            ###########################################################################################
            # MADE DYNAMIC by FOR LOOP
            # WITH gt_categories.json file
            for col_val in final_val:
                cat_merged_df = merged_df.loc[col_val]

            images_cat_df = cat_merged_df[['image', 'num_obj']].reset_index()
            print('images_cat_df', images_cat_df)
            ###########################################################################################
            # create dictionary with amount of expected detections for each image
            detected_gt_per_image = Counter([bb.get_image_name() for bb in gt_boxes])
            detected_gt_per_image = {}
            for r in images_cat_df.index:
                # print(f'row: {r}')
                image = images_cat_df['image'][r]
                objects = images_cat_df['num_obj'][r]
                # for key, val in detected_gt_per_image.items():
                #   if (key == images_cat_df['image']):
                detected_gt_per_image[image] = np.zeros(objects)
            print(f"detected_gt_per_image: {detected_gt_per_image}")
            # print(f"v-det: {v['det']}")
            # sort detections by decreasing confidence
            dect = []
            dects = [a for a in sorted(v['det'], key=lambda bb: bb.get_confidence(), reverse=True)]
            print(f"initial len(dects):{len(dects)}")
            for det in dects:
                img_det = det.get_image_name()
                print(f"in the img_det: {img_det}")
                if img_det in detected_gt_per_image:
                    dect.append(det)
                    print("found the image in the list,adding to dect")

                    # print(f"after len(dects):{len(dect)}")
            print(f"outside len(dect):{len(dect)}")

            TP = np.zeros(len(dect))
            FP = np.zeros(len(dect))
            print(f'FP & TP: {FP, TP}')

            # print(f'Evaluating class: {c}')
            # dict_table = {
            #     'image': [],
            #     'confidence': [],
            #     'TP': [],
            #     'FP': [],
            #     'acc TP': [],
            #     'acc FP': [],
            #     'precision': [],
            #     'recall': []
            # }

            dict_table = {
                'image': [],
                'class_name': [],
                'confidence': [],
                'TP': [],
                'FP': [],
                'precision': [],
                'recall': []
            }
            # Loop through detections
            for idx_det, det in enumerate(dect):

                img_det = det.get_image_name()

                if generate_table:
                    dict_table['image'].append(img_det)
                    dict_table['confidence'].append(f'{100 * det.get_confidence():.2f}%')

                # Find ground truth image
                gt = [gt for gt in classes_bbs[c]['gt'] if gt.get_image_name() == img_det]
                # gt = [gt for gt in classes_bbs[c]['gt']]
                # Get the maximum iou among all detectins in the image
                iouMax = sys.float_info.min
                # Given the detection det, find ground-truth with the highest iou
                for j, g in enumerate(gt):
                    # print('Ground truth gt => %s' %
                    #       str(g.get_absolute_bounding_box(format=BBFormat.XYX2Y2)))
                    iou = BoundingBox.iou(det, g)
                    # print(f"\n\ndet: {det}\ng: {g}\n\n")
                    # print(f"\n\ndet_bbox: {det.get_absolute_bounding_box(format=BBFormat.XYX2Y2)}\ng_bbox: {g.get_absolute_bounding_box(format=BBFormat.XYX2Y2)}\n\n")

                    if iou > iouMax:
                        iouMax = iou
                        id_match_gt = j
                # Assign detection as TP or FP
                if iouMax >= iou_threshold:
                    # gt was not matched with any detection
                    if detected_gt_per_image[img_det][id_match_gt] == 0:
                        TP[idx_det] = 1  # detection is set as true positive
                        detected_gt_per_image[img_det][
                            id_match_gt] = 1  # set flag to identify gt as already 'matched'
                        # print("TP")
                        if generate_table:
                            dict_table['class_name'].append(c)
                            dict_table['TP'].append(1)
                            dict_table['FP'].append(0)
                    else:
                        FP[idx_det] = 1  # detection is set as false positive
                        if generate_table:
                            dict_table['class_name'].append(c)
                            dict_table['FP'].append(1)
                            dict_table['TP'].append(0)
                        # print("FP")
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= iou_threshold.
                else:
                    FP[idx_det] = 1  # detection is set as false positive
                    if generate_table:
                        dict_table['class_name'].append(c)
                        dict_table['FP'].append(1)
                        dict_table['TP'].append(0)
                    # print("FP")
            # compute precision, recall and average precision
            # print(f"FP: {FP}")
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            print(f"acc_FP, acc_TP: {acc_FP, acc_TP}")
            # print(f"npos: {npos}")
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            rec[rec>1]=1
            prec[prec>1]=1
            if generate_table:
                # print(f"\n\ndict table: {dict_table}")
                # dict_table['acc TP'] = list(acc_TP)
                # dict_table['acc FP'] = list(acc_FP)
                dict_table['precision'] = list(prec)
                dict_table['recall'] = list(rec)
                table = pd.DataFrame(dict_table)
            else:
                table = None
            # Depending on the method, call the right implementation
            if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
                # print(c)
                # print(rec, prec)
                # print(type(rec))
                # print(type(prec))
                [ap, mpre, mrec, ii] = calculate_ap_every_point(rec, prec)
            elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
                [ap, mpre, mrec, _] = calculate_ap_11_point_interp(rec, prec)
            else:
                Exception('method not defined')
            # add class result in the dictionary to be returned
            ret['_'.join(name_val)] = {
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),
                'method': method,
                'iou': iou_threshold,
                'table': table
            }

            # print(f"ret: {ret[distance + '_' + illumination + '_' +  occlusion  + '_' + num_objects_image]}")

        class_res.append({c: ret})

    # For mAP, only the classes in the gt set should be considered
    # mAP = sum([v['AP'] for k, v in ret.items() if k in gt_classes_only]) / len(gt_classes_only)
    # return {'per_class': ret, 'mAP': mAP}

    return class_res


def plot_precision_recall_curve(results, iteration_id,
                                mAP=None,
                                showInterpolatedPrecision=False,
                                savePath=None,
                                showGraphic=True):
    result = None
    plt.close()
    # Each result represents a class
    for classId, result in results.items():
        if result is None:
            raise IOError(f'Error: Class {classId} could not be found.')

        prec_ = []
        rec_ = []
        ap_ = []

        for index, (v1, v2, v3, v4, v5, v6) in enumerate(
                zip(result["categories"], result["num_images_" + iteration_id], result["ap_" + iteration_id],
                    result["precision_" + iteration_id], result["recall_" + iteration_id],
                    result["f1-score_" + iteration_id])):
            if v2 != 0:
                prec_.append(v4)
                rec_.append(v5)
                ap_.append(v3)

        precision = np.asarray(prec_, dtype=np.float32)
        recall = np.asarray(rec_, dtype=np.float32)
        ap = np.asarray(ap_, dtype=np.float32)

        print(precision, recall, ap)

        _, mpre, mrec, i = calculate_ap_every_point(recall, precision)
        print(f"mpre, mrec : {mpre}, {mrec}")

        if showInterpolatedPrecision:
            # if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
            plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')

        plt.plot(recall, precision, label=f'{classId}')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    if mAP:
        map_str = "{0:.2f}%".format(mAP * 100)
        plt.title(f'Precision x Recall curve, mAP={map_str}')
    else:
        plt.title('Precision x Recall curve')
    plt.legend(shadow=True)
    plt.grid()
    if savePath is not None:
        plt.savefig(os.path.join(savePath, 'all_classes.png'))
    if showGraphic is True:
        plt.show()
        # plt.waitforbuttonpress()
        plt.pause(0.05)
    return results


def plot_precision_recall_curves(results, iteration_id,
                                 showAP=False,
                                 showInterpolatedPrecision=False,
                                 savePath=None,
                                 showGraphic=True):
    result = None
    # Each result represents a class
    for classId, result in results.items():
        prec_ = []
        rec_ = []
        ap_ = []
        if result is None:
            raise IOError(f'Error: Class {classId} could not be found.')

        for index, (v1, v2, v3, v4, v5, v6) in enumerate(
                zip(result["categories"], result["num_images_" + iteration_id], result["ap_" + iteration_id],
                    result["precision_" + iteration_id], result["recall_" + iteration_id],
                    result["f1-score_" + iteration_id])):
            if v2 != 0:
                prec_.append(v4)
                rec_.append(v5)
                ap_.append(v3)

        precision = np.asarray(prec_, dtype=np.float32)
        recall = np.asarray(rec_, dtype=np.float32)
        try:
            ap = sum(ap_) / result["num_cat_types"]
        except ZeroDivisionError as e:
            ap = None
            print(e)

        # print(precision, recall, ap)
        _, mpre, mrec, i = calculate_ap_every_point(recall, precision)
        print(classId)
        print(f"mpre, mrec : {mpre}, {mrec}")
        plt.close()
        print(precision, recall)
        plt.plot(recall, precision, label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        # if showAP:
        if ap is not None:
            ap_str = "{0:.2f}%".format(ap * 100)
        else:
            ap_str = 0
        # ap_str = "{0:.4f}%".format(average_precision * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(classId), ap_str))
        # else:
        # plt.title('Precision x Recall curve \nClass: %s' % str(classId))
        plt.legend(shadow=True)
        plt.grid()
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        if savePath is not None:
            plt.savefig(os.path.join(savePath, classId + '.png'))
        if showGraphic is True:
            plt.show()
            # plt.waitforbuttonpress()
            plt.pause(0.05)
    return results
