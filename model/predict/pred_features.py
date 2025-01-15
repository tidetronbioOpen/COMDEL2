
import numpy as np
from multiprocessing import Pool
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amino_dict = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'U': 4,
    'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
    'L': 10, 'M': 11, 'O': 11, 'N': 12, 'P': 13,
    'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18,
    'W': 19, 'X': 20, 'B': 20, 'Z': 20, 'Y': 21
}

"""index encoding"""
def index_encoding(sequences):
    aa_num = amino_dict
    fea_list = []

    for seq_temp in sequences:
        fea = []
        for char in seq_temp:
            tem_vec = aa_num.get(char, 0)  # Using the get method of the dictionary, return 0 if the character does not exist
            fea.append(tem_vec)
        fea_list.append(fea)

    return fea_list

def get_max_length(peptide_type):
    """返回根据 peptide_type 定义的最大序列长度"""
    max_length_dict = {
        'ACP': 69, 'ADP': 30, 'AHP': 30, 'AIP': 25, 'AGP': 25,
        'CPP': 32, 'DeP': 33, 'DDP': 50, 'HeP': 36,
        'NuP': 200, 'BiP': 19, 'UmP':15
    }
    return max_length_dict.get(peptide_type, 15)

def get_min_lenth(peptide_type):
    """返回根据 peptide_type 定义的最小序列长度"""
    min_length_dict = {
        'ACP': 2, 'ADP': 4, 'AHP': 7, 'AIP': 11, 'antigen': 8,
        'cpp': 4, 'Defense': 11, 'Drug_Delivery': 4, 'Hemolytic': 4,
        'Neuro': 11, 'Bitter': 2, 'Umami': 2
    }
    return min_length_dict.get(peptide_type, 4)

"""pad sequences"""
def pad_encoding(sequences, peptide_type):
    aa_num = amino_dict
    max_seq_length = get_max_length(peptide_type)
    padded_seqs = []

    for seq_temp in sequences:
        fea = []
        for char in seq_temp:
            tem_vec = aa_num.get(char, 0)
            fea.append(tem_vec)
        fea += [0] * (max_seq_length - len(seq_temp))  # Pad the sequence with zeros
        padded_seqs.append(fea)

    return np.array(padded_seqs)

"""氨基酸组成(AAC)"""
def compute_AAC(fea_list):
    #AAC = np.zeros((len(fea_list), 21), device=device)
    AAC = torch.zeros((len(fea_list), 21), device=device)
    for row, seq in enumerate(fea_list):
        for aa in seq:
            # col = aa[0] - 1
            col = aa - 1
            AAC[row][col] += 1 / len(seq)

    return AAC

"""二肽组成DPC)"""
def compute_DPC(fea_list, comb_length=22):
    comb = []  # {list:231} [[1,1],[1,2],[1,3]....]
    for i in range(1, comb_length):
    # for i in range(comb_length):
        for j in range(i, comb_length):
            comb.append([i, j])
    comb_index = {}
    for i in range(len(comb)):
        comb_index[tuple(comb[i])] = i

    #DPC = [[0] * len(comb) for _ in range(len(fea_list))]
    DPC = torch.zeros((len(fea_list), len(comb)), device=device)
    for row in range(len(fea_list)):
        seq = fea_list[row]
        for i in range(len(seq) - 1):
            a = sorted([seq[i], seq[i + 1]])
            index = comb_index[tuple(a)]
            DPC[row][index] += 1 / (len(seq) - 1)

    return DPC


"""k间隔基酸组对(CKSAAGP)"""
def CKSAAGP(sequences, gap=5):
    def generateGroupPairs(groupKey):
        gPair = {}
        for key1 in groupKey:
            for key2 in groupKey:
                gPair[key1 + '.' + key2] = 0
        return gPair

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    num_seqs = len(sequences)
    num_features = len(gPairIndex) * (gap + 1)

    #encodings = np.zeros((num_seqs, num_features))
    encodings = torch.zeros((num_seqs, num_features), device=device)

    for row, seq in enumerate(sequences):
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum_count = 0
            for p1 in range(len(seq)):
                p2 = p1 + g + 1
                if p2 < len(seq) and seq[p1] in AA and seq[p2] in AA:
                    gPair[index[seq[p1]] + '.' + index[seq[p2]]] += 1
                    sum_count += 1

            if sum_count != 0:
                for idx, gp in enumerate(gPairIndex):
                    encodings[row, g * len(gPairIndex) + idx] = gPair[gp] / sum_count

    return encodings


"""k-mer"""
def kmer_feature(sequences):
    def TransDict_from_list(groups):
        tar_list = ['0', '1', '2', '3', '4', '5', '6']
        result = {}
        index = 0
        for group in groups:
            g_members = sorted(group)
            for c in g_members:
                result[c] = str(tar_list[index])
            index += 1
        return result

    def translate_sequence(seq, TranslationDict):
        from_list = []
        to_list = []
        for k, v in TranslationDict.items():
            from_list.append(k)
            to_list.append(v)
        return seq.translate(str.maketrans("".join(from_list), "".join(to_list)))

    def get_3_protein_trids():
        nucle_com = []
        chars = ['0', '1', '2', '3', '4', '5', '6']
        base = len(chars)
        end = len(chars) ** 3
        for i in range(end):
            n = i
            ch0 = chars[n % base]
            n //= base
            ch1 = chars[n % base]
            n //= base
            ch2 = chars[n % base]
            nucle_com.append(ch0 + ch1 + ch2)
        return nucle_com

    def get_4_nucleotide_composition(tris, seq):
        seq_len = len(seq)
        k = len(tris[0])

        # 添加检查以确保 seq_len - k + 1 为非负数
        if seq_len < k:
            return torch.zeros(len(tris), device=device)

        tri_feature = torch.zeros(len(tris), device=device)
        note_feature = torch.zeros((len(tris), seq_len - k + 1), device=device)

        for x in range(seq_len - k + 1):
            kmer = seq[x:x + k]
            if kmer in tris:
                ind = tris.index(kmer)
                note_feature[ind][x] += 1

        u, s, v = torch.svd(note_feature)
        for i in range(len(s)):
            tri_feature += u[:, i] * s[i] / seq_len

        return tri_feature

    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    protein_tris = get_3_protein_trids()

    kmer = []
    for protein_seq in sequences:
        protein_seq = translate_sequence(protein_seq, group_dict)
        protein_tri_fea = get_4_nucleotide_composition(protein_tris, protein_seq)
        kmer.append(protein_tri_fea)

    kmer_tensor = torch.stack(kmer)
    return kmer_tensor


"""
aa_features
"""

"""AAindex features"""
def AAindex_embedding(sequences, file, peptide_type):
    aaindex = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            tokens = line.strip().split('\t')
            aa = tokens[0]
            features = list(map(float, tokens[1:]))
            aaindex[aa] = torch.tensor(features, device=device)

    max_len = get_max_length(peptide_type)
    feature_dim = len(next(iter(aaindex.values())))  # 获取特征的维度

    all_feature_matrix = []
    for seq in sequences:
        feature_matrix = torch.zeros((max_len, feature_dim), device=device)
        for i, aa in enumerate(seq):
            if aa in aaindex:
                feature_matrix[i] = aaindex[aa]
            else:
                feature_matrix[i] = torch.zeros(feature_dim, device=device)  # Use a zero-vector if the AA is not found
        all_feature_matrix.append(feature_matrix)

    all_embeddings = torch.stack(all_feature_matrix)
    reshaped_feats = all_embeddings.view(all_embeddings.size(0), -1)
    return reshaped_feats


"""PC6 features"""
def pc6_embedding(sequences, file, peptide_type):
    pc6_feat = {}
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            tokens = line.strip().split('\t')
            aa = tokens[0]
            features = list(map(float, tokens[1:]))
            pc6_feat[aa] = torch.tensor(features, device=device)

    max_len = get_max_length(peptide_type)
    feature_dim = len(next(iter(pc6_feat.values())))  # 获取特征的维度

    all_pc6_matrix = []
    for seq in sequences:
        pc6_matrix = torch.zeros((max_len, feature_dim), device=device)
        for i, aa in enumerate(seq):
            if aa in pc6_feat:
                pc6_matrix[i] = pc6_feat[aa]
        all_pc6_matrix.append(pc6_matrix)

    all_embeddings = torch.stack(all_pc6_matrix)
    reshaped_feats = all_embeddings.view(all_embeddings.size(0), -1)
    return reshaped_feats

"""BLOSUM62"""


def BLOSUM62_embedding(sequences, peptide_type):
    # 读取BLOSUM62矩阵并创建字典
    with open('../feature_data/blosum62.txt') as f:
        text = f.read().split('\n')
    text = list(filter(None, text))  # 移除空行
    cha = list(filter(None, text[0].split(' ')))  # 提取字符

    index = []
    for i in range(1, len(text)):
        temp = list(map(float, filter(None, text[i].split(' '))))
        index.append(temp)
    index = np.array(index)
    BLOSUM62_dict = {cha[j]: torch.tensor(index[:, j], device=device) for j in range(len(cha))}

    max_len = get_max_length(peptide_type)
    all_embeddings = []

    for each_seq in sequences:
        temp_embeddings = []
        for each_char in each_seq:
            temp_embeddings.append(BLOSUM62_dict[each_char])

        temp_embeddings = torch.stack(temp_embeddings, dim=0).to(device)

        if max_len > len(each_seq):
            zero_padding = torch.zeros((max_len - len(each_seq), 23), device=device)
            data_pad = torch.cat((temp_embeddings, zero_padding), dim=0)
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]

        all_embeddings.append(data_pad)

    all_embeddings = torch.stack(all_embeddings, dim=0)
    reshaped_feats = all_embeddings.view(all_embeddings.size(0), -1)

    return reshaped_feats

"""PAAC"""


def PAAC_embedding(sequences, peptide_type):
    with open('../feature_data/PAAC.txt') as f:
        text = f.read().split('\n')
    text = list(filter(None, text))  # Remove empty strings

    cha = list(filter(None, text[0].split('\t')))[1:]
    index = []
    for line in text[1:]:
        temp = list(map(float, filter(None, line.split('\t')[1:])))
        index.append(temp)
    index = torch.tensor(index, device=device)

    AAI_dict = {cha[j]: index[:, j] for j in range(len(cha))}
    AAI_dict['X'] = torch.zeros(3, device=device)

    max_len = get_max_length(peptide_type)
    all_embeddings = []
    for each_seq in sequences:
        temp_embeddings = [AAI_dict[each_char] for each_char in each_seq]
        temp_embeddings = torch.stack(temp_embeddings, dim=0)

        if max_len > len(each_seq):
            zero_padding = torch.zeros((max_len - len(each_seq), 3), device=device)
            data_pad = torch.cat((temp_embeddings, zero_padding), dim=0)
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]

        all_embeddings.append(data_pad)

    all_embeddings = torch.stack(all_embeddings)
    reshaped_feats = all_embeddings.view(all_embeddings.size(0), -1)
    return reshaped_feats

"""predict pad sequences"""

def predict_seq_features(seq_list, seq_names, peptide_type):
    window_sizes = {
        'ACP': 20, 'AIP': 10, 'AHP': 20, 'ADP': 20, 'antigen': 10, 'cpp': 20,
        'Drug_Delivery': 20, 'Hemolytic': 20, 'Neuro': 20, 'Defense': 20,
        'Bitter': 10, 'Umami': 10
    }
    strides = {
        'ACP': 10, 'AIP': 10, 'AHP': 10, 'ADP': 10, 'antigen': 10, 'cpp': 10,
        'Drug_Delivery': 10, 'Hemolytic': 10, 'Neuro': 10, 'Defense': 10,
        'Bitter': 10, 'Umami': 10
    }

    window_size = window_sizes.get(peptide_type, 10)  # Default window size if not listed
    stride = strides.get(peptide_type, 10)  # Default stride if not listed

    new_seq_list = []
    new_name_list = []
    # Apply sliding window to each sequence and generate fragment names
    for idx, seq in enumerate(seq_list):
        seq_name = seq_names[idx]
        length = len(seq)
        segment_number = 0
        i = 0

        # Process sequence with sliding window until the last possible full segment
        while i <= length - window_size:
            new_seq_list.append(seq[i:i + window_size])
            segment_number += 1
            new_name_list.append(f"{seq_name}_{segment_number}")
            i += stride

        # Handle the last segment based on the remaining sequence length
        remaining = length - i
        if remaining > 0:
            if remaining <= 10:
                # If remaining <= 10, use it directly
                new_seq_list.append(seq[i:])
                segment_number += 1
                new_name_list.append(f"{seq_name}_{segment_number}")
            else:
                # If remaining > 10 and less than 20, process with window_size=10 and stride=10
                while i <= length - 10:
                    new_seq_list.append(seq[i:i + 10])
                    segment_number += 1
                    new_name_list.append(f"{seq_name}_{segment_number}")
                    i += 10
                # Add the last segment if there's any sequence left
                if i < length:
                    new_seq_list.append(seq[i:])
                    segment_number += 1
                    new_name_list.append(f"{seq_name}_{segment_number}")

    seq_name = new_name_list
    sequences = new_seq_list
    # 序列编码
    excel_index = index_encoding(sequences)
    # pad seq
    pad_seqs = pad_encoding(sequences, peptide_type)
    # # 计算seq特征
    AAC_ndarray = compute_AAC(excel_index)  # 形状：(seq_num, 21)
    DPC_ndarray = compute_DPC(excel_index)  # 形状：(seq_num, 231)
    CKSAAGP_ndarray = CKSAAGP(sequences, gap=5)  # 形状：(seq_num, 150)
    kmer_ndarray = kmer_feature(sequences)  # 形状：(seq_num, 343)

    # 计算aa特征
    AAinedx_ndarray = AAindex_embedding(sequences,
                                        '../feature_data/aaindex_feature.txt', peptide_type)  # {ndarray:(3,6(len_max),531)}  [[[4.35,0.61,1.18..],[4.65,3.44..]],[[...],[..]]]
    pc6_ndarray = pc6_embedding(sequences, '../feature_data/PC6.txt', peptide_type)  # {ndarray:(3,6(len_max),6)}
    BLOSUM62_ndarray = BLOSUM62_embedding(sequences, peptide_type)  # {ndarray:(3,6(len_max),23)}
    PAAC_ndarray = PAAC_embedding(sequences, peptide_type)  # {ndarray:(3,6(len_max),3)}

    # 将数据转换为PyTorch张量并移动到GPU上
    pad_seqs = torch.tensor(pad_seqs, dtype=torch.float32, device=device)
    # 计算seq特征

    # 再次检查数据是否在GPU上
    print("data to GPU:")
    print("pad_seqs device:", pad_seqs.device)
    print("AAC_ndarray device:", AAC_ndarray.device)

    # 根据 feature_type 参数来选择合并哪些特征
    if peptide_type == 'ACP':
        merged_array = np.c_[AAinedx_ndarray.cpu().numpy(), pc6_ndarray.cpu().numpy(), BLOSUM62_ndarray.cpu().numpy()]
    elif peptide_type == 'ADP':
        merged_array = np.c_[kmer_ndarray.cpu().numpy(), pc6_ndarray.cpu().numpy(), BLOSUM62_ndarray.cpu().numpy()]
    elif peptide_type == 'AHP':
        merged_array = np.c_[AAC_ndarray.cpu().numpy(), AAinedx_ndarray.cpu().numpy(), BLOSUM62_ndarray.cpu().numpy()]
    elif peptide_type == 'AIP':
        merged_array = np.c_[DPC_ndarray.cpu().numpy(), CKSAAGP_ndarray.cpu().numpy(), AAinedx_ndarray.cpu().numpy(), pc6_ndarray.cpu().numpy()]
    elif peptide_type == 'AGP':
        merged_array = np.c_[AAC_ndarray.cpu().numpy(), DPC_ndarray.cpu().numpy(), kmer_ndarray.cpu().numpy(), pc6_ndarray.cpu().numpy(), PAAC_ndarray.cpu().numpy()]
    elif peptide_type == 'BiP':
        merged_array = np.c_[AAC_ndarray.cpu().numpy(), DPC_ndarray.cpu().numpy(), AAinedx_ndarray.cpu().numpy()]
    elif peptide_type == 'CPP':
        merged_array = np.c_[AAC_ndarray.cpu().numpy(), CKSAAGP_ndarray.cpu().numpy(), BLOSUM62_ndarray.cpu().numpy(), PAAC_ndarray.cpu().numpy()]
    elif peptide_type == 'DeP':
        merged_array = np.c_[AAC_ndarray.cpu().numpy(), DPC_ndarray.cpu().numpy(), AAinedx_ndarray.cpu().numpy(), pc6_ndarray.cpu().numpy(), BLOSUM62_ndarray.cpu().numpy(), PAAC_ndarray.cpu().numpy()]
    elif peptide_type == 'DDP':
        merged_array = np.c_[DPC_ndarray.cpu().numpy(), kmer_ndarray.cpu().numpy(), pc6_ndarray.cpu().numpy(), BLOSUM62_ndarray.cpu().numpy(), PAAC_ndarray.cpu().numpy()]
    elif peptide_type == 'HeP':
        merged_array = np.c_[AAC_ndarray.cpu().numpy(), CKSAAGP_ndarray.cpu().numpy(), kmer_ndarray.cpu().numpy(), PAAC_ndarray.cpu().numpy()]
    elif peptide_type == 'NuP':
        merged_array = np.c_[AAinedx_ndarray.cpu().numpy(), pc6_ndarray.cpu().numpy()]
    elif peptide_type == 'UmP':
        merged_array = np.c_[AAC_ndarray.cpu().numpy(), kmer_ndarray.cpu().numpy(), pc6_ndarray.cpu().numpy()]
    else:
        raise ValueError("Invalid feature_type. Choose from ['ACP', 'ADP', 'AHP', 'AIP', 'antigen', 'Bitter', 'cpp', 'Defense', 'Drug_Delivery', 'Hemolytic', 'Neuro', 'Umami']")

    return sequences, seq_name, pad_seqs.cpu(), merged_array

if __name__ == '__main__':
    seq_list = ['ACCMLYSDAHHNMMLLYYEEASLLHGHHAGG', 'VFLVLLFLGALGLCLAGRRRSVQWCAVSQPEATKCFQWQRNMRKVRGPPV', 'GRRRSVQWCAVSQPEATKCFQWQRNMRKVRGPPVSCIKRDSPIQCIQAIA']
    seq_names = ['SampleSeq1', 'SampleSeq2', 'SampleSeq3']
    excel_path = './data/train_test/train_balanced_AIP.xlsx'
    peptide_type = 'ACP'  # ['ACP', 'ADP', 'AHP', 'AIP',  'CPP', 'AGP', 'DDP', 'DeP', 'HeP', 'NuP', 'UmP', 'BiP']    # normalized_features, labels, names, pad_seqs = generate_seq_features(excel_path, peptide_type)
    sequences, seq_name, pad_seqs, normalized_features = predict_seq_features(seq_list, seq_names, peptide_type)
    print("Features calculated:", normalized_features.shape)
