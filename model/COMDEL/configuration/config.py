
import argparse


def get_train_config():
    parse = argparse.ArgumentParser(description='model')

    # preoject setting
    parse.add_argument('-learn-name', type=str, default='50', help='learn name')
    parse.add_argument('-save-best', type=bool, default=True, help='if save parameters of the current best model ')
    parse.add_argument('-threshold', type=float, default=0.80, help='save threshold')

    # model parameters
    parse.add_argument('-max-len', type=int, default=256, help='max length of input sequences')
    parse.add_argument('-num-layer', type=int, default=3, help='number of encoder blocks')
    parse.add_argument('-num-head', type=int, default=8, help='number of head in multi-head attention')
    parse.add_argument('-dim-embedding', type=int, default=64, help='residue embedding dimension') # 残基嵌入维度
    parse.add_argument('-dim-feedforward', type=int, default=64, help='hidden layer dimension in feedforward layer') # 前馈层中的隐藏层维度
    parse.add_argument('-dim-k', type=int, default=32, help='embedding dimension of vector k or q')
    parse.add_argument('-dim-v', type=int, default=32, help='embedding dimension of vector v')
    parse.add_argument('-num-embedding', type=int, default=2, help='number of sense in multi-sense')
    parse.add_argument('-k-mer', type=int, default=3, help='number of k(-mer) in multi-scaled')
    parse.add_argument('-embed-atten-size', type=int, default=8, help='size of soft attetnion')

    # parse.add_argument('-max-len', type=int, default=256, help='max length of input sequences')
    # parse.add_argument('-num-layer', type=int, default=3, help='number of encoder blocks')
    # parse.add_argument('-num-head', type=int, default=8, help='number of head in multi-head attention')
    # parse.add_argument('-dim-embedding', type=int, default=32, help='residue embedding dimension')
    # parse.add_argument('-dim-feedforward', type=int, default=32, help='hidden layer dimension in feedforward layer')
    # parse.add_argument('-dim-k', type=int, default=32, help='embedding dimension of vector k or q')
    # parse.add_argument('-dim-v', type=int, default=32, help='embedding dimension of vector v')
    # parse.add_argument('-num-embedding', type=int, default=2, help='number of sense in multi-sense')
    # parse.add_argument('-k-mer', type=int, default=3, help='number of k(-mer) in multi-scaled')
    # parse.add_argument('-embed-atten-size', type=int, default=8, help='size of soft attetnion')

    # training parameters
    parse.add_argument('-lr', type=float, default=0.0001, help='learning rate')
    # parse.add_argument('-lr', type=float, default=0.0005, help='learning rate')
    # parse.add_argument('-reg', type=float, default=0.0025, help='weight lambda of regularization')
    parse.add_argument('-reg', type=float, default=0.0000, help='weight lambda of regularization')
    # parse.add_argument('-batch-size', type=int, default=64, help='number of samples in a batch')
    parse.add_argument('-batch-size', type=int, default=32, help='number of samples in a batch')
    # parse.add_argument('-batch-size', type=int, default=16, help='number of samples in a batch')
    parse.add_argument('-epoch', type=int, default=70, help='number of iteration')
    parse.add_argument('-k-fold', type=int, default=-1, help='k in cross validation,-1 represents train-test approach')
    # parse.add_argument('-k-fold', type=int, default=5, help='k in cross validation,-1 represents train-test approach')
    parse.add_argument('-num-class', type=int, default=2, help='number of classes')
    parse.add_argument('-cuda', action='store_false', default=True, help='if use cuda') # TODO True-->False
    parse.add_argument('-device', type=int, default=0, help='device id')
    parse.add_argument('-interval-log', type=int, default=20,
                       help='how many batches have gone through to record the training performance')
    parse.add_argument('-interval-valid', type=int, default=1,
                       help='how many epoches have gone through to record the validation performance')
    parse.add_argument('-interval-test', type=int, default=1,
                       help='how many epoches have gone through to record the test performance')

    parse.add_argument('-infile', type=str, help='input file')
    parse.add_argument('-results', type=str, help='out results file')
    parse.add_argument('-plt_results', type=str, help='out plt_results file')    
    config = parse.parse_args()
    return config
