import os
import pickle
import pandas as pd

# def load_excel(filename):
#     df = pd.read_excel(filename)
#     # Mandatory field, assumed to be present in every excel file
#     if 'Sequence' not in df.columns:
#         raise ValueError("The mandatory 'Sequence' column is missing in the Excel file.")
#
#     df['Sequence'] = df['Sequence'].astype(str)
#
#     sequences = df['Sequence'].tolist()
#
#     # Optional fields
#     labels = df['Label'].tolist() if 'Label' in df.columns else None
#     names = df['Name'].tolist() if 'Name' in df.columns else None
#     seq_lengths = df['Sequence Length'].tolist() if 'Sequence Length' in df.columns else None
#
#     return sequences, labels, names, seq_lengths

def load_excel(filename):
    df = pd.read_excel(filename)

    # Convert column names to lowercase for case-insensitive matching
    df.columns = [c.lower() for c in df.columns]

    # Mandatory field, assumed to be present in every excel file
    if 'sequence' not in df.columns:
        raise ValueError("The mandatory 'sequence' column is missing in the Excel file.")

    df['sequence'] = df['sequence'].astype(str)
    sequences = df['sequence'].tolist()

    # Optional fields: only include them if they are present in the DataFrame
    labels = df['label'].tolist() if 'label' in df.columns else None
    names = df['name'].tolist() if 'name' in df.columns else None
    seq_lengths = df['sequence length'].tolist() if 'sequence length' in df.columns else None

    return sequences, labels, names, seq_lengths

def save_feature_indices(indices, filename):
    """保存特征索引至文件"""
    # 检查目录是否存在，如果不存在则创建
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(filename, 'wb') as f:
        pickle.dump(indices, f)

def load_feature_indices(filename):
    """从文件加载特征索引"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    load_excel(r'../data/train_test/test_balanced_Bitter.xlsx')