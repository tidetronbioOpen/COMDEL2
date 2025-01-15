
import pandas as pd


def load_tsv_format_data(filename, skip_head=True):
    sequences = []
    labels = []

    with open(filename, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            list = line.split('\t')
            sequences.append(list[2])
            labels.append(int(list[1]))

    return sequences, labels

def load_excel(filename, limited_length=1000, sample=300, label=1):
    """"""
    df = pd.read_excel(filename)
    df['sequence'] = df['sequence'].astype(str)
    df['length'] = df['sequence'].apply(lambda x: len(x))
    df = df.loc[df['length']<=limited_length,]
    # df = df.sample(sample)
    return df['sequence'].tolist(), df['label'].tolist(), df['id'].tolist()

def load_fasta(filename, default_ids=1, limited_length=1000):
    """载入单种类型的标签，需要手动打上标签"""
    from Bio import SeqIO
    sequences, ids, labels  = [], [], []
    for i in SeqIO.parse(filename, "fasta"):
        seq = str(i.seq)
        description = i.description
        if len(seq) <= limited_length:
            sequences.append(seq)
            ids.append(i.id)
            if '\t' in description:
                labels.append(int(description.split('\t')[-1]))
            elif '|' in description:
                labels.append(int(description.split('|')[-1]))
            else:
                labels.append(default_ids)
    return sequences, labels, ids

if __name__ == '__main__':
    load_fasta(r'/home/adminpro/Project/CT_AMP/CT_Probio\train.fa')