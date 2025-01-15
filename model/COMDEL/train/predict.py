
from train.main import periodic_test, load_config
from model import Basic
import torch
from util import util_file
from preprocess.data_loader import transform_token2index, make_data_with_unified_length, construct_dataset
from train.model_operation import load_model
config = load_config()
config.max_len = 257
config.vocab_size = 28

import pickle
residue2idx = pickle.load(open('../data/residue2idx.pkl', 'rb'))
config.vocab_size = len(residue2idx)
config.token2index = residue2idx
model = Basic.BERT(config)
model = load_model(model, r'../result/ACPred-LAF_train_amps/ACC[0.8202], ACPred-LAF_train_amps.pt')

import torch.utils.data as Data
device = torch.device("cpu")

path = r'G:\DataChrome\Lactobacillus_acidipiscis_gca_001436945.ASM143694v1.pep.all.fa'
seq, id = util_file.load_fasta(path,1)
token_list_total, max_len = transform_token2index(seq, config)
config.max_len = max_len
data_total = make_data_with_unified_length(token_list_total, id, config)
data_loader = construct_dataset(data_total, config)
test_iter = Data.DataLoader(data_loader,
                            batch_size=config.batch_size,
                            drop_last=False)

model =  model.to(device)
for batch in test_iter:
    input, label = batch
    input = input.to(device)
    logits, output = model(input)
    print()









