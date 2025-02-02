

import pickle
import torch
import torch.utils.data as Data

from configuration import config
from util import util_file
from preprocess import data_augmentation


def transform_token2index(sequences, config):
    token2index = config.token2index
    token2index['O'] =6
    for i, seq in enumerate(sequences):
        seq = seq.replace('l', 'L')
        sequences[i] = list(seq)

    token_list = list()
    max_len = 0
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        seq_id_v = len(list(set(seq))-token2index.keys())
        if seq_id_v >0:
            print()
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)

    print('-' * 20, '[transform_token2index]: check sequences_residue and token_list head', '-' * 20)
    # print('sequences_residue', sequences[0:5])
    # print('token_list', token_list[0:5])
    return token_list, max_len


def make_data_with_unified_length(token_list, labels, config):
    max_len = config.max_len = config.max_len + 2  # add [CLS] and [SEP]
    token2index = config.token2index

    data = []
    for i in range(len(labels)):
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]
        n_pad = max_len - len(token_list[i])
        token_list[i].extend([0] * n_pad)
        data.append([token_list[i], labels[i]])

    print('-' * 20, '[make_data_with_unified_length]: check token_list head', '-' * 20)
    # print('max_len + 2', max_len)
    # print('token_list + [pad]', token_list[0:5])
    return data


# 构造迭代器
def construct_dataset(data, config):
    cuda = torch.cuda.is_available()
    batch_size = config.batch_size

    # print('-' * 20, '[construct_dataset]: check data dimension', '-' * 20)
    # print('len(data)', len(data))
    # print('len(data[0])', len(data[0]))
    # print('len(data[0][0])', len(data[0][0]))
    # print('data[0][1]', data[0][1])
    # print('len(data[1][0])', len(data[1][0]))
    # print('data[1][1]', data[1][1])

    input_ids, labels = zip(*data)

    if cuda:
        input_ids, labels = torch.cuda.LongTensor(input_ids), torch.cuda.LongTensor(labels)
    else:
        input_ids, labels = torch.LongTensor(input_ids), torch.LongTensor(labels)

    print('-' * 20, '[construct_dataset]: check data device', '-' * 20)
    print('input_ids.device:', input_ids.device)
    print('labels.device:', labels.device)

    print('-' * 20, '[construct_dataset]: check data shape', '-' * 20)
    print('input_ids:', input_ids.shape)  # [num_sequences, seq_len]
    print('labels:', labels.shape)  # [num_sequences, seq_len]

    data_set = MyDataSet(input_ids, labels)

    print('len(data_loader)', len(data_set))
    return data_set


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def load_data_two_pathways(config):
    path_data_train = config.path_train_data
    path_data_test = config.path_test_data

    # data augmentation
    # sequences_train, labels_train = data_augmentation.augmentation(path_data_train, config, append = False)
    sequences_train, labels_train = util_file.load_tsv_format_data(path_data_train)
    sequences_test, labels_test = util_file.load_tsv_format_data(path_data_test)
    # sequences_train: ['MNH', 'APD', ...]
    # labels_train: [1, 0, ...]

    token_list_train, max_len_train = transform_token2index(sequences_train, config)
    token_list_test, max_len_test = transform_token2index(sequences_test, config)
    # token_list_train: [[1, 5, 8], [2, 7, 9], ...]

    config.max_len = max(max_len_train, max_len_test)
    config.max_len_train = max_len_train
    config.max_len_test = max_len_test

    data_train = make_data_with_unified_length(token_list_train, labels_train, config)
    data_test = make_data_with_unified_length(token_list_test, labels_test, config)
    # data_train: [[[1, 5, 8], 0], [[2, 7, 9], 1], ...]

    data_loader_train = construct_dataset(data_train, config)
    data_loader_test = construct_dataset(data_test, config)
    data_loader_train = Data.DataLoader(data_loader_train,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=False)
    data_loader_test = Data.DataLoader(data_loader_test,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=False)
    return data_loader_train, data_loader_test


def load_data_one_pathways(config):
    """"""
    path_data_train = config.path_train_data

    sequences_0, labels_0, names_0 = util_file.load_fasta(r'../data/AMPs/M_model_train_nonAMP_sequence.fasta', 0, )
    sequences, labels, names = util_file.load_fasta(r'../data/AMPs/M_model_train_AMP_sequence.fasta', 1, )
    sequences.extend(sequences_0)
    labels.extend(labels_0)
    names.extend(names_0)
    # no_amp_sequences, no_amp = util_file.load_fasta(r'E:\CT_Project\ACPred-LAF\data\AMPs\M_model_train_nonAMP_sequence.fasta', 0)
    #
    # amp_sequences.extend(no_amp_sequences)
    # amp.extend(no_amp)

    token_list_total, max_len = transform_token2index(sequences, config)
    config.max_len = max_len
    config.max_len_train = max_len
    config.max_len_test = max_len

    data_total = make_data_with_unified_length(token_list_total, labels, config)

    data_loader = construct_dataset(data_total, config)
    data_loader_train, data_loader_test = torch.utils.data.random_split(data_loader, [0.8,0.2])

    data_loader_train = Data.DataLoader(data_loader_train,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=False)
    data_loader_test = Data.DataLoader(data_loader_test,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=False)
    return data_loader_train, data_loader_test


def load_data_one_pathways1(config):
    """"""
    train_sequences, train_labels, train_names = util_file.load_excel('')
    test_sequences, test_labels, test_names = util_file.load_excel('')

    token_list_train, max_len_train = transform_token2index(train_sequences, config)
    token_list_test, max_len_test = transform_token2index(test_sequences, config)
    config.max_len = max(max_len_train, max_len_test)
    config.max_len_train = max_len_train
    config.max_len_test = max_len_test

    data_train = make_data_with_unified_length(token_list_train, train_labels, config)
    data_test = make_data_with_unified_length(token_list_test, test_labels, config)

    data_loader_train = construct_dataset(data_train, config)
    data_loader_test = construct_dataset(data_test, config)

    data_loader_train = Data.DataLoader(data_loader_train,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=False)
    data_loader_test = Data.DataLoader(data_loader_test,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  drop_last=False)
    return data_loader_train, data_loader_test

if __name__ == '__main__':
    '''
    check loading tsv data
    '''
    config = config.get_train_config()

    token2index = pickle.load(open('../data/residue2idx.pkl', 'rb'))
    config.token2index = token2index
    print('token2index', token2index)

    config.path_train_data = '../data.tsv'
    sequences, labels = util_file.load_tsv_format_data(config.path_train_data)
    token_list, max_len = transform_token2index(sequences, config)
    data = make_data_with_unified_length(token_list, labels, config)
    data_loader = construct_dataset(data, config)

    print('-' * 20, '[data_loader]: check data batch', '-' * 20)
    for i, batch in enumerate(data_loader):
        input, label = batch
        print('batch[{}], input:{}, label:{}'.format(i, input.shape, label.shape))
