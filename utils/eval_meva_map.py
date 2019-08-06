from sklearn.metrics import average_precision_score
import numpy as np
import os
import json

def evaluate_map(full_gt_labels,full_pred_scores,label_list):
    pass
    full_pred_scores = np.asarray(full_pred_scores)
    ap_list = []

    #   Note: the map is only evaluated on 12 positive events.
    for i in range(0, len(label_list)):
        label = label_list[i]

        gt_label = [];
        pred_scores = full_pred_scores[:, i]

        for now_label in full_gt_labels:
            if now_label == i:
                gt_label.append(1)
            else:
                gt_label.append(0)

        gt_label = np.asarray(gt_label)
        ap = average_precision_score(gt_label, pred_scores)
        print(label, ap)

        ap_list.append(ap)
    map = np.nanmean(ap_list)
    print("mAP:", map)

    return map, ap_list

if __name__ =="__main__":

    video_lst_file = '/data2/Meva/mevaTrainTestList/trainE_list.txt'
    pred_file = '/data2/Meva/result/trainA/vidtest/test.json'

    label_id_file = '/data2/Meva/proposals/lists/label.lst'
    fr = open(label_id_file, 'r')
    fr_lines = fr.readlines()
    class_names = []
    for fr_line in fr_lines:
        class_names.append(fr_line[:-1])
    fr.close()

    input_files = []
    with open(video_lst_file, 'r') as f:
        for row in f:
            input_files.append(row.replace("\n", ""))

    labels_np = []
    predicted_np = []
    with open(pred_file, 'r') as f:
        pred_results = json.load(f)
        pred_results = pred_results['results']
    for k, val in pred_results.items():
        score = val
        labels_np.append(class_names.index(k.split('/')[1]))
        predicted_np.append(score)

    map_split = evaluate_map(np.asarray(labels_np), np.asarray(predicted_np), class_names)
