import pandas as pd
import numpy as np
import random
from preprocess import data_loader, data_loader_kmer
from configuration import config as cf
from util import util_metric
from train.model_operation import load_model
from model_value import Basic, LSTM
from train.model_operation import save_model, adjust_model
from train.visualization import dimension_reduction, penultimate_feature_visulization
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from train.main import load_config
from util.util_file import load_fasta,load_excel
import torch.utils.data as Data
import torch
import torch.nn.functional as F
import os,sys,re
import pickle
import gc
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import warnings
warnings.filterwarnings("ignore")


model_name_path_dict = \
    {'50':r'result/Firsttime_train/50_amps.pt',
     '100':r'result/Firsttime_train/100_amps.pt',
     '150':r'result/Firsttime_train/150_amps.pt',
     '200':r'result/Firsttime_train/200_amps.pt',
     '250':r'result/Firsttime_train/250_amps.pt'
     }

color_list = ['g', 'r', 'm', 'c', 'k']
length_limited = 300 #
length_list = model_name_path_dict.keys()


def fasta_converter(sequence_list, file_name, sequence_name_list=None, labels_list=None):
    '''Use the indices of the database and the column 'Sequence' to create a fasta file'''
    ofile = open(os.path.join(file_name), "w")
    for i in range(len(sequence_list)):
        try:
            if sequence_name_list and labels_list:
                ofile.write(">" + str(i) + "|" + str(sequence_name_list[i]) + "|" +str( labels_list[i]) + "\n" + sequence_list[i] + "\n")
            elif labels_list:
                ofile.write(">" + str(i) + "|" +str( labels_list[i]) + "\n" + sequence_list[i] + "\n")
            else:
                ofile.write(">" +str(i) + "\n" + sequence_list[i] + "\n")
        except:
            print()
    ofile.close()


def file_main(path):
    """"""
    config = load_config()
    residue2idx = pickle.load(open('data/residue2idx.pkl', 'rb'))
    config.vocab_size = len(residue2idx)
    config.token2index = residue2idx
    # 载入数据
    if 'xlsx' in path:
        seq, label, name = load_excel(path)
    else:
        seq, label, name = load_fasta(path)
    df = pd.DataFrame.from_dict({'sequence':seq, 'label':label, 'name':name})
    df['length'] = df['sequence'].apply(lambda x: len(x))
    df = df.loc[(10<=df['length'])&(df['length']<=300), ]

    total_df = pd.DataFrame()
    for num, i in enumerate(length_list):
        last = int(i)
        if last == 50:
            tmp_data = df.loc[df['length']<=last,]
        else:
            first = int(list(length_list)[num-1])
            tmp_data = df.loc[df['length']<=last,]
            tmp_data = tmp_data.loc[first<tmp_data['length'],]

        tmp_seq = tmp_data['sequence'].tolist()
        tmp_label = tmp_data['label'].tolist()

        token_list, max_len = data_loader.transform_token2index(tmp_seq, config)
        config.max_len = int(i)+2
        data = data_loader.make_data_with_unified_length(token_list, tmp_label, config)

        data = data_loader.construct_dataset(data, config)

        data_loader_value = Data.DataLoader(data,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            drop_last=False)
        print('model_name', config.model_name)
        model = Basic.BERT(config)
        if config.cuda: model.cuda()

        model = load_model(model, model_name_path_dict[i])
        # device = torch.device("cuda" if config.cuda else "cpu")
        device = torch.device("cpu")

        label_pred = torch.empty([0], device=device)
        label_real = torch.empty([0], device=device)
        pred_prob = torch.empty([0], device=device)

        for batch in data_loader_value:
            input, label = batch
            logits, output = model(input)

            pred_prob_all = F.softmax(logits, dim=1)
            # Prediction probability [batch_size, class_num]
            pred_prob_positive = pred_prob_all[:, 1]
            # Probability of predicting positive classes [batch_size]
            pred_prob_sort = torch.max(pred_prob_all, 1)
            # The maximum probability of prediction in each sample [batch_size]
            pred_class = pred_prob_sort[1]
            label_pred = torch.cat([label_pred, pred_class.float()])
            label_real = torch.cat([label_real, label.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])
        tmp_data['label_pred'] = label_pred.tolist()
        tmp_data['possibilities'] = pred_prob.tolist()
        total_df = pd.concat([total_df, tmp_data])
        # 释放GPU资源
        del model
        del label_pred
        del label_real
        del pred_prob
        # 释放其他变量
        del data_loader_value
        del data
        if config.cuda:
            torch.cuda.empty_cache()
        # 手动触发垃圾回收
        gc.collect()
    total_df.to_csv('{}*predict_df.csv'.format(path), index=False)

def website_fasta_to_dataframe(fasta_string):
    """"""
    sequences = []
    headers = []

    # Split the input string into lines
    lines = fasta_string.strip().split('\n')

    # Process each line in the FASTA string
    current_header = None
    current_sequence = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            # If the line starts with '>', it's a header line
            # Save the previous sequence and header
            if current_header and current_sequence:
                headers.append(current_header)
                sequences.append(''.join(current_sequence))

            # Start a new sequence and header
            current_header = line[1:].strip()
            current_sequence = []
        else:
            # Append the sequence line to the current sequence
            current_sequence.append(line.strip())

    # Save the last sequence and header
    if current_header and current_sequence:
        headers.append(current_header)
        sequences.append(''.join(current_sequence))

    # Create the DataFrame
    df = pd.DataFrame({'name': headers, 'sequence': sequences})

    return df



def website_main(str_value):
    # 设置随机数种子
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    df = website_fasta_to_dataframe(str_value)
    df['length'] = df['sequence'].apply(lambda x: len(x))
    df['label'] = 1
    process_df = df.loc[(10 <= df['length']) & (df['length'] <= 300),]

    config = load_config()
    residue2idx = pickle.load(open('data/residue2idx.pkl', 'rb'))
    config.vocab_size = len(residue2idx)
    config.token2index = residue2idx

    total_df = pd.DataFrame()
    for num, i in enumerate(length_list):
        last = int(i)
        if last == 50:
            tmp_data = df.loc[df['length'] <= last,]
        else:
            first = int(list(length_list)[num - 1])
            tmp_data = df.loc[df['length'] <= last,]
            tmp_data = tmp_data.loc[first < tmp_data['length'],]
        if len(tmp_data) == 0:
            continue
        tmp_seq = tmp_data['sequence'].tolist()
        tmp_label = tmp_data['label'].tolist()

        token_list, max_len = data_loader.transform_token2index(tmp_seq, config)
        config.max_len = int(i) + 2
        data = data_loader.make_data_with_unified_length(token_list, tmp_label, config)

        data = data_loader.construct_dataset(data, config)
        data_loader_value = Data.DataLoader(data,
                                            batch_size=config.batch_size,
                                            shuffle=False,
                                            drop_last=False)

        model = Basic.BERT(config)
        device = torch.device("cuda" if config.cuda else "cpu")
        if config.cuda:
            model = model.cuda()

        # 保证模型加载到正确的设备，并设置为评估模式
        model = load_model(model, model_name_path_dict[i])
        model.to(device)
        model.eval()

        label_pred = torch.empty([0], device=device)
        label_real = torch.empty([0], device=device)
        pred_prob = torch.empty([0], device=device)

        for batch in data_loader_value:
            input, label = batch
            input = input.to(device)
            label = label.to(device)
            logits, output = model(input)

            pred_prob_all = F.softmax(logits, dim=1)
            pred_prob_positive = pred_prob_all[:, 1]
            pred_prob_sort = torch.max(pred_prob_all, 1)
            pred_class = pred_prob_sort[1]
            label_pred = torch.cat([label_pred, pred_class.float()])
            label_real = torch.cat([label_real, label.float()])
            pred_prob = torch.cat([pred_prob, pred_prob_positive])

        tmp_data['label_pred'] = label_pred.tolist()
        tmp_data['possibilities'] = pred_prob.tolist()
        total_df = pd.concat([total_df, tmp_data])

        # 释放GPU资源
        del model
        del label_pred
        del label_real
        del pred_prob
        # 释放其他变量
        del data_loader_value
        del data
        if config.cuda:
            torch.cuda.empty_cache()
        # 手动触发垃圾回收
        gc.collect()

    return total_df['possibilities'].tolist()


def valid_fasta(text):
    """
    return bool
    """
    aa_U = "AFCUDNEQGHLIKOMPRSTVWYBZJX"
    aa_L = aa_U.lower()
    nu_U = "ATCG"
    nu_L = nu_U.lower()
    m_nu = re.search("[^%s%s]"%(nu_U, nu_L), text)
    m_aa = re.search("[^%s%s]"%(aa_U, aa_L), text)
    char_seq = {i:0 for i in text}
    if not m_aa:
        return True
    return False

def save_fasta(file, cont):
    """
    save file
    """
    f_ = open(file, "w")
    new_str = ">seq\n"+cont
    f_.write(new_str)
    f_.close()


def read_fasta(file):
    """
    read content of fasta
    """
    f_ = open(file,"r")
    c_ = f_.read()
    f_.close()
    seqs_ = re.sub("\s","",c_).upper()
    if valid_fasta(seqs_):
        new_str = ">seq\n"+seqs_
        return True, new_str
    return False, None

"""网站可视化"""
def read_csv(file_path):
    df = pd.read_csv(file_path, usecols=['Name', 'Sequence'])
    return df

def process_and_predict(df, window_size, stride, output_file):
    segments_data = []
    for index, row in df.iterrows():
        gene_id = row['Name']
        full_sequence = row['Sequence']
        n = len(full_sequence)

        segment_number = 0  # Reset segment counter for each new sequence
        i = 0
        # Process main segments
        while i + window_size <= n:
            segment_number += 1
            segment_id = f"{gene_id}_{segment_number}"
            segment = full_sequence[i:i + window_size]
            segments_data.append(predict_segment(segment_id, segment))
            i += stride

        # Process smaller segments at the end if any
        while i + 10 <= n:
            segment_number += 1
            segment_id = f"{gene_id}_{segment_number}"
            segment = full_sequence[i:i + 10]
            segments_data.append(predict_segment(segment_id, segment))
            i += 10

        # Last piece if smaller than 10
        if i < n:
            segment_number += 1
            segment_id = f"{gene_id}_{segment_number}"
            segment = full_sequence[i:]
            segments_data.append(predict_segment(segment_id, segment))

    # Convert to DataFrame and save
    result_df = pd.DataFrame(segments_data)
    result_df.insert(0, 'Index', range(1, len(result_df) + 1))
    result_df.to_csv(output_file, index=False)
    print(f"Segmented predictions have been saved to '{output_file}'.")

def predict_segment(segment_id, segment):
    fasta_formatted_segment = f">{segment_id}\n{segment}"
    predictions = website_main(fasta_formatted_segment)  # Assuming this returns a list of predictions
    formatted_predictions = [str(pred)[:5] for pred in predictions]

    return {
        "Symbol": segment_id,
        "Sequences": segment,
        "AMP": ",".join(formatted_predictions)
    }


if __name__ == '__main__':
    # 确保输出文件夹存在
    # if not os.path.exists(args.result_path):
    #     os.makedirs(args.result_path)
    # if not os.path.exists(args.plt_path):
    #     os.makedirs(args.plt_path)

    config = load_config()
    in_file = config.infile
    results = config.results
    plt_results = config.plt_results
    input_file = in_file  # Update this path if needed
    seq_df = pd.read_csv(input_file)
    output_filename=results+'/'+'Predicted_AMP.csv'
    process_and_predict(seq_df, window_size=30, stride=10, output_file=output_filename)

    # Plotting
    #df = pd.read_csv(f'{args.result_path}/Predicted_AMP.csv.csv')
    df = pd.read_csv(output_filename)
    """横坐标一个点代表一个序列段"""
    plt.figure(figsize=(20, 5))
    sequence_indices = range(len(df))
    for column in df.columns:
        if f'AMP' in column:
            plt.plot(sequence_indices, df[column], color='black', label=column, linewidth=10)

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
    #添加在y=0.5处的水平虚线
    plt.axhline(y=0.5, color='grey', linewidth=5, linestyle='--')

    plt.title('threshold(0.5)', fontsize=25, fontweight='bold', loc='left')
    plt.xlabel('Position(aa)', fontsize=25, fontweight='bold')
    plt.ylabel('AMP', fontsize=30, fontweight='bold')
    #plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(plt_results+'/AMP_predictions.png', format='png', bbox_inches='tight')
    plt.show()
#     plt.close()  # Close the plot to release memory

    """平铺序列段"""
    # 将所有序列连接成一个长的序列
    # long_sequence = ''.join(df['Sequences'])
    # # 初始化一个列表来保存每个氨基酸的位置和预测值
    # positions = []
    # predictions = []
    # # 计算位置并记录预测值
    # current_position = 0
    # for index, row in df.iterrows():
    #     sequence_length = len(row['Sequences'])
    #     positions.extend(range(current_position, current_position + sequence_length))
    #     predictions.extend([row['AMP']] * sequence_length)
    #     current_position += sequence_length
    # # 创建绘图
    # plt.figure(figsize=(20, 5))
    # # 绘制每个位置的预测值，并加粗线条
    # plt.plot(positions, predictions, color='black', linewidth=10, label='AMP')
    # # 去除顶部和右侧的边框
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # # 设置坐标轴和刻度的线条宽度
    # ax.spines['left'].set_linewidth(5)
    # ax.spines['bottom'].set_linewidth(5)
    #
    # # 设置刻度的宽度和大小
    # ax.tick_params(axis='both', which='major', width=5, length=8, labelsize=30)
    # ax.tick_params(axis='both', which='minor', width=5, length=6, labelsize=30)
    #
    # # 设置加粗字体
    # bold_font = FontProperties(weight='bold')
    # for label in ax.get_xticklabels():
    #     label.set_fontproperties(bold_font)
    # for label in ax.get_yticklabels():
    #     label.set_fontproperties(bold_font)
    #
    # # 添加在y=0.5处的水平虚线
    # plt.axhline(y=0.5, color='grey',linewidth=5, linestyle='--')
    # # 设置标签和标题
    # plt.xlabel('Position(aa)', fontsize=25, fontweight='bold')
    # plt.ylabel('AMP', fontsize=30, fontweight='bold')
    # # plt.title('threshold(0.5)', fontsize=25, fontweight='bold', loc='left')
    # # 添加图例
    # #plt.legend(fontsize=20)
    # # 调整布局以适应标签
    # plt.tight_layout()
    # plt.savefig('AMP_predictions.png', format='png', bbox_inches='tight')
    # plt.show()
