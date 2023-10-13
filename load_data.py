import pandas as pd
import numpy as np
import os


def load_data_r(file_path):
    df = pd.read_csv(file_path)
    label_cols = ['R1-1-1', 'R2-1-2', 'R3-2-1', 'R4-2-2', 'R5-3-1', 'R6-3-2',
                  'R7-3-3', 'R8-4-1', 'R9-4-2', 'R10-5-1', 'R11-5-2', 'R12-5-3']

    texts = [t.strip() for t in df['text']]
    labels = np.array(df[label_cols])

    return texts, labels


def load_data_i(file_path, indicators):
    df = pd.read_csv(file_path)
    r_label_cols = ['R1-1-1', 'R2-1-2', 'R3-2-1', 'R4-2-2', 'R5-3-1', 'R6-3-2',
                    'R7-3-3', 'R8-4-1', 'R9-4-2', 'R10-5-1', 'R11-5-2', 'R12-5-3']
    i_label_cols = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12']

    r_labels = (np.array(df[r_label_cols])).transpose()
    i_labels = (np.array(df[i_label_cols])).transpose()
    texts = np.array([t.strip() for t in df['text']])

    text_input, target_input, labels, target_idx = [], [], [], []
    for i in range(12):
        related_mask = r_labels[i] == 1
        related_num = np.sum(r_labels[i])

        text_input += list(texts[related_mask])
        target_input += [indicators[i]] * related_num
        labels += list(i_labels[i][related_mask])
        target_idx += [i] * related_num

    return text_input, target_input, labels, target_idx

