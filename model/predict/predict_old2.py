import os
import pandas as pd
import torch
from Bio import SeqIO
import matplotlib.pyplot as plt
from pred_model import BiLSTM, BiLSTMAttention, BiLSTMPredictor  # 导入模型
from pred_dataloader import predict_dataloader  # 导入数据加载函数
import argparse  # 导入argparse
import numpy as np

# 解析命令行参数
parser = argparse.ArgumentParser(description='Run peptide predictions and plot results.')
parser.add_argument('--infile', type=str, required=True, help='Input file path containing sequences.')
parser.add_argument('--result_path', type=str, default='predict_result/test_result', help='Directory to save prediction results.')
parser.add_argument('--plt_path', type=str, default='predict_result/test_result/test_plt', help='Directory to save plots.')

args = parser.parse_args()

# 设置随机种子和设备
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# 确保输出文件夹存在
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
if not os.path.exists(args.plt_path):
    os.makedirs(args.plt_path)

# 读取输入文件（此处假设是CSV文件，其中包含一列序列）
seq_df = pd.read_csv(args.infile)
seq_names = seq_df['Name'].tolist()
seq_list = seq_df['Sequence'].tolist()  # 假设序列列名为 'Sequence'


# 定义 peptide_type 和模型参数
peptide_types = ['ACP', 'ADP', 'AHP', 'AIP',  'CPP', 'AGP', 'DDP', 'DeP', 'HeP', 'NuP', 'UmP', 'BiP']
model_params_dict = {
    'ACP': {'model_type': 'BiLSTM', 'use_corred_features': True},
    'ADP': {'model_type': 'BiLSTMPredictor', 'use_corred_features': True},
    'AHP': {'model_type': 'BiLSTMPredictor', 'use_corred_features': True},
    'AIP': {'model_type': 'BiLSTMPredictor', 'use_corred_features': True},
    'CPP': {'model_type': 'BiLSTMPredictor', 'use_corred_features': True},
    'AGP': {'model_type': 'BiLSTMAttention', 'use_corred_features': True},
    'DDP': {'model_type': 'BiLSTMAttention', 'use_corred_features': True},
    'DeP': {'model_type': 'BiLSTMPredictor', 'use_corred_features': True},
    'HeP': {'model_type': 'BiLSTMAttention', 'use_corred_features': True},
    'NuP': {'model_type': 'BiLSTMPredictor', 'use_corred_features': True},
    'UmP': {'model_type': 'BiLSTM', 'use_corred_features': True},
    'BiP': {'model_type': 'BiLSTM', 'use_corred_features': True}
}

# 循环处理所有肽类型
for peptide_type, settings in model_params_dict.items():
    # 初始化一个DataFrame来保存当前peptide_type的预测结果
    current_df = pd.DataFrame()

    model_type = settings['model_type']
    use_corred_features = settings['use_corred_features']

    # Create dataloader
    sequences, seq_name, pred_loader = predict_dataloader(seq_list, seq_names, peptide_type, use_corred_features)
    batch_seq, batch_feature = next(iter(pred_loader))
    # 获取序列长度
    input_seq_len = batch_seq.shape[1]
    print("input_seq_len: ", input_seq_len)
    # 获取特征维度
    input_dim = batch_feature.shape[1]
    print("input_dim: ", input_dim)

    model_path = f'../best_model/best_model_{peptide_type}.pth'

    # 加载模型并进行预测
    with torch.no_grad():
        if model_type == 'BiLSTM':
            model = BiLSTM(vocab_size=22, embedding_dim=200, hidden_dim=128, input_dim=input_dim, output_dim=1).to(
                device)
        elif model_type == 'BiLSTMAttention':
            model = BiLSTMAttention(vocab_size=22, embedding_dim=200, hidden_dim=128, input_seq_len=input_seq_len,
                                    input_dim=input_dim, output_dim=1).to(device)
        elif model_type == 'BiLSTMPredictor':
            model = BiLSTMPredictor(vocab_size=22, embedding_dim=200, hidden_dim=128, input_dim=input_dim,
                                    aggregate_dim=128, predictor_d_h=128).to(device)
        else:
            continue

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model device:", next(model.parameters()).device)

    all_predictions = []
    # Prediction
    for batch_seq, batch_feature in pred_loader:
        batch_seq = batch_seq.to(device).long()
        batch_feature = batch_feature.to(device)
        # print("Data device:", batch_seq.device, batch_feature.device)
        with torch.no_grad():
            predictions = model(batch_seq, batch_feature).squeeze()

        all_predictions.append(predictions.cpu().detach().numpy())

    torch.cuda.empty_cache()
    # 保留前10个字符
    filtered_sequences = [seq[:10] for seq in sequences]
    filtered_seq_names = [seq_name[i] for i in range(len(sequences))]
    filtered_predictions = [all_predictions[i] for i in range(len(sequences))]

    current_df['symbol'] = filtered_seq_names
    current_df['Sequences'] = filtered_sequences
    current_df[f'{peptide_type}'] = filtered_predictions
    # 创建索引列，从1开始
    current_df.insert(0, 'Index', range(1, len(current_df) + 1))
    current_df.to_csv(f'{args.result_path}/Predicted_{peptide_type}.csv', index=False)
    print(f"Predictions for {peptide_type} saved to {args.result_path}/Predicted_{peptide_type}.csv'")

# Combine predictions into a single DataFrame
combined_df = pd.DataFrame()
# Load predictions for each peptide type and combine them
for peptide_type in peptide_types:
    # Load predictions for the current peptide type
    predicted_file = f'{args.result_path}/Predicted_{peptide_type}.csv'
    current_df = pd.read_csv(predicted_file)
    # Append predictions to the combined DataFrame
    if combined_df.empty:
        combined_df = current_df[[f'{peptide_type}']].rename(columns={f'{peptide_type}': peptide_type})
    else:
        combined_df = pd.concat(
            [combined_df, current_df[[f'{peptide_type}']].rename(columns={f'{peptide_type}': peptide_type})],
            axis=1)
    # Load predictions for AMP peptide type
predicted_amp_file = f'{args.result_path}/Predicted_AMP.csv'
predicted_amp_df = pd.read_csv(predicted_amp_file)
# Rename the AMP prediction column to 'AMP'
predicted_amp_df = predicted_amp_df.rename(columns={'AMP': 'AMP'})
# Merge AMP predictions into the combined DataFrame
combined_df['AMP'] = predicted_amp_df['AMP']
# Add Index column
combined_df.insert(0, 'Index', range(1, len(combined_df) + 1))
# Save combined predictions to CSV
combined_df.to_csv(f'{args.result_path}/Combined_predict.csv', index=False)
print(f"Combined predictions saved to {args.result_path}/Combined_predict.csv'")

# Combine predictions into a single DataFrame
combined_df = pd.DataFrame()
# Load predictions for each peptide type and combine them
for peptide_type in peptide_types:
    # Load predictions for the current peptide type
    predicted_file = f'{args.result_path}/Predicted_{peptide_type}.csv'
    current_df = pd.read_csv(predicted_file)
    # Append predictions to the combined DataFrame
    if combined_df.empty:
        combined_df = current_df[[f'{peptide_type}']].rename(columns={f'{peptide_type}': peptide_type})
    else:
        combined_df = pd.concat(
            [combined_df, current_df[[f'{peptide_type}']].rename(columns={f'{peptide_type}': peptide_type})],
            axis=1)
    # Load predictions for AMP peptide type
predicted_amp_file = f'{args.result_path}/Predicted_AMP.csv'
predicted_amp_df = pd.read_csv(predicted_amp_file)
# Rename the AMP prediction column to 'AMP'
predicted_amp_df = predicted_amp_df.rename(columns={'AMP': 'AMP'})
# Merge AMP predictions into the combined DataFrame
combined_df['AMP'] = predicted_amp_df['AMP']
# Add Index column
combined_df.insert(0, 'Index', range(1, len(combined_df) + 1))
# Save combined predictions to CSV
combined_df.to_csv(f'{args.result_path}/Combined_predict.csv', index=False)
print(f"Combined predictions saved to {args.result_path}/Combined_predict.csv'")

# Plotting
# 定义每种peptide_type的颜色
color_map = {
    'ACP': 'blue', 'ADP': 'orange', 'AHP': 'green', 'AIP': 'brown',
    'CPP': 'magenta', 'AGP': 'purple', 'DDP': 'olive', 'DeP': 'pink',
    'HeP': 'cyan', 'NuP': 'Gold', 'UmP': 'navy', 'BiP': 'red'
}
# 定义每种peptide_type的阈值线
threshold_map = {
    'ADP': 0.7, 'AIP': 0.6, 'AGP': 0.95, 'DDP': 0.95,
    'HeP': 0.9, 'UmP': 0.97
}
for peptide_type in peptide_types:
    df = pd.read_csv(f'{args.result_path}/Predicted_{peptide_type}.csv')
    
    # """横坐标一个点代表一个序列段"""
    # plt.figure(figsize=(20, 5))
    # sequence_indices = range(len(df))
    # peptide_color = color_map[peptide_type]
    # for column in df.columns:
    #     if f'{peptide_type}' in column:
    #         plt.plot(sequence_indices, df[column], label=column, linewidth=10, color=peptide_color)
    # # 去除顶部和右侧的边框
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # # 设置坐标轴和刻度的线条宽度
    # ax.spines['left'].set_linewidth(5)
    # ax.spines['bottom'].set_linewidth(5)
    # # 设置刻度的宽度和大小
    # ax.tick_params(axis='both', which='major', width=5, length=8, labelsize=30)
    # ax.tick_params(axis='both', which='minor', width=5, length=6, labelsize=30)
    # # 添加阈值虚线
    # threshold = threshold_map.get(peptide_type, 0.5)
    # plt.axhline(y=threshold, color='grey', linewidth=5, linestyle='--')
    # plt.title(f'Threshold ({threshold})', fontsize=25, fontweight='bold', loc='left')
    # plt.xlabel('Position(aa)', fontsize=25, fontweight='bold')
    # plt.ylabel(f'{peptide_type}', fontsize=30, fontweight='bold', color=peptide_color)
    # #plt.legend(fontsize=20)
    # # 设置横坐标显示的个数和位置
    # num_xticks = min(10, len(df))  # 设置最多显示的刻度数量
    # xticks_positions = range(0, len(df), max(1, len(df) // num_xticks))
    # plt.xticks(xticks_positions, [f'{x}' for x in xticks_positions])

    # plt.tight_layout()
    # plt.savefig(f'{args.plt_path}/{peptide_type}_predictions.png', format='png', bbox_inches='tight')
    # plt.show()
    
    # plt.close()  # Close the plot to release memory
    """平铺序列段"""
    # 将所有序列连接成一个长的序列
    long_sequence = ''.join(df['Sequences'])
    # 初始化一个列表来保存每个氨基酸的位置和预测值
    positions = []
    predictions = []
    # 计算位置并记录预测值
    current_position = 0
    for index, row in df.iterrows():
        sequence_length = len(row['Sequences'])
        positions.extend(range(current_position, current_position + sequence_length))
        predictions.extend([row[f'{peptide_type}']] * sequence_length)
        current_position += sequence_length
    peptide_color = color_map[peptide_type]
    # 创建绘图
    plt.figure(figsize=(20, 5))
    # 绘制每个位置的预测值，并加粗线条
    plt.plot(positions, predictions, linewidth=10, label=f'{peptide_type}', color=peptide_color)
    # 去除顶部和右侧的边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 设置坐标轴和刻度的线条宽度
    ax.spines['left'].set_linewidth(5)
    ax.spines['bottom'].set_linewidth(5)
    # 设置刻度的宽度和大小
    ax.tick_params(axis='both', which='major', width=5, length=8, labelsize=30)
    ax.tick_params(axis='both', which='minor', width=5, length=6, labelsize=30)
    
    # 添加阈值虚线
    threshold = threshold_map.get(peptide_type, 0.5)
    plt.axhline(y=threshold, color='grey', linewidth=5, linestyle='--')
    # 设置标签和标题
    plt.xlabel('Sequences', fontsize=25, fontweight='bold')
    plt.ylabel(f'{peptide_type}', fontsize=25, fontweight='bold',color=peptide_color)
    plt.title(f'Threshold ({threshold})', fontsize=30, fontweight='bold', loc='left')
    # 添加图例
    #plt.legend(fontsize=20)
    # 调整布局以适应标签
    plt.tight_layout()
    plt.savefig(f'{args.plt_path}/{peptide_type}_predictions.png', format='png', bbox_inches='tight')
    plt.show()
    plt.close()
