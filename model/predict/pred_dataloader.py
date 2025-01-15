import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pickle

import pred_features
# from pred_feat_select import *
from pred_data_file import *
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" predict数据集"""
def predict_dataloader(seq_list, seq_names, peptide_type, use_corred_features):
    # 从 features.py 生成特征
    sequence, seq_name, pred_sequence, pred_feature = pred_features.predict_seq_features(seq_list, seq_names, peptide_type) # predict_seq_features

    scaler_filename = f'../feature_result/scaler_state_{peptide_type}.pkl'
    with open(scaler_filename, 'rb') as file:
        loaded_scaler = pickle.load(file)
    scaled_pred_features = loaded_scaler.transform(pred_feature)

    # 根据 use_corred_features 参数来决定是否进行高度相关性筛选
    if use_corred_features:
        filename = f'../feature_result/correlated_indices_{peptide_type}.pkl'
        with open(filename, 'rb') as file:
            correlated_indices = pickle.load(file)
        # 移除高度相关的特征
        pred_corred = np.delete(scaled_pred_features, list(correlated_indices), axis=1)
    else:
        pred_corred = scaled_pred_features

    # 定义需要应用 PCA 转换的肽类型列表
    pca_peptide_types = ['AGP', 'DDP', 'HeP', 'NuP', 'UmP']
    # 检查当前肽类型是否需要应用 PCA 转换
    if peptide_type in pca_peptide_types:
        pca_filename = f'../feature_result/select_indices_{peptide_type}.pkl'
        if os.path.exists(pca_filename):
            with open(pca_filename, 'rb') as file:
                pca = pickle.load(file)
            pred_features_selected = pca.transform(pred_corred)
        else:
            print(f"No PCA model found for {peptide_type}, using original features.")
            pred_features_selected = pred_corred
    else:
        # 对于不在上述列表中的肽类型，直接使用原始特征
        indices_filename = f'../feature_result/select_indices_{peptide_type}.pkl'
        saved_indices = load_feature_indices(indices_filename)
        pred_features_selected = pred_corred[:, saved_indices]

    # 转换为 PyTorch 张量并移到 GPU
    pred_feature_tensors = torch.tensor(pred_features_selected, dtype=torch.float32).to(device)
    pred_seqs_tensors = torch.tensor(pred_sequence, dtype=torch.long).to(device)

    # 创建一个 TensorDataset 并移到 GPU
    pred_dataset = TensorDataset(pred_seqs_tensors, pred_feature_tensors)
    # 创建 DataLoader
    pred_loader = DataLoader(pred_dataset, shuffle=False)

    return sequence, seq_name, pred_loader
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    excel_path = './data/train_test/test_balanced_Hemolytic.xlsx'
    peptide_type = 'ADP'  # ['ACP', 'AHP', 'AIP', 'antigen', 'cpp', 'Defense', 'Drug_Delivery', 'Hemolytic', 'Neuro']
    # 设置特征索引文件名为对应的peptide_type
    indices_filename = f'{peptide_type}.pkl'

    if peptide_type == 'ACP':
        feature_method = 'PCA'
        num_features = 150
        use_corred_features = False
        use_saved_indices = False
    elif peptide_type == 'AHP':
        feature_method = 'RF'
        num_features = 150
        use_corred_features = True
        use_saved_indices = False
    else:
        raise ValueError(
            "Invalid peptide_type. Choose from ['ACP', 'ADP', 'AHP', 'AIP', 'antigen', 'cpp', 'Defense', 'Drug_Delivery', 'Hemolytic', 'Neuro', 'Bitter', 'Umami']")


    # 对新的数据集进行预测
    seq_list = ['ADDDCTALLOMCAAMMNN']  # 这里需要替换为实际的序列数据
    seq_names = 'seq1'
    sequence, seq_name, pred_loader = predict_dataloader(seq_list, seq_names, peptide_type, use_corred_features)

