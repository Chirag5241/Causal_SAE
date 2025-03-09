import numpy as np
import torch
from debias.classifier import PytorchClassifier
import pickle
import torch.nn.functional as F


# todo: find more elegant solution than this
class MaskedLinear(torch.nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        # self.in_features = in_features
        # self.out_features = out_features
        # self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        # self.bias = torch.nn.Parameter(torch.zeros(out_features))

        assert in_features % 2 == 0
        self.mask = torch.cat(
            [torch.ones(in_features // 2), torch.zeros(in_features // 2)])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input * self.mask, self.weight, self.bias)


def define_network(W: np.ndarray, b: np.ndarray, projection_mat: np.ndarray = None,
                   device: str = 'cpu', concat_mask=False):
    # if concat_mask:
    #     embedding_net = MaskedLinear(in_features=W.shape[1], out_features=W.shape[0])
    # else:
    #     embedding_net = torch.nn.Linear(in_features=W.shape[1], out_features=W.shape[0])
    embedding_net = torch.nn.Linear(in_features=W.shape[1], out_features=W.shape[0])
    embedding_net.weight.data = torch.tensor(W)
    embedding_net.bias.data = torch.tensor(b)

    if projection_mat is not None:
        if concat_mask:
            projection_net = MaskedLinear(in_features=projection_mat.shape[1],
                                          out_features=projection_mat.shape[0],
                                          bias=False)
        else:
            projection_net = torch.nn.Linear(in_features=projection_mat.shape[1],
                                             out_features=projection_mat.shape[0],
                                             bias=False)
        projection_net.weight.data = torch.tensor(projection_mat, dtype=torch.float)
        for p in projection_net.parameters():
            p.requires_grad = False
        word_prediction_net = torch.nn.Sequential(projection_net, embedding_net)

    else:
        word_prediction_net = torch.nn.Sequential(embedding_net)

    net = PytorchClassifier(word_prediction_net, device=device)
    return net


def load_data(path):
    vecs = np.load(f"{path}/last_vec.npy", allow_pickle=True)
    vecs = np.array([x[1:-1] for x in vecs])

    with open(f"{path}/tokens.pickle", 'rb') as f:
        labels = pickle.load(f)

    return vecs, labels


def load_labels(labels_file):
    with open(labels_file, 'rb') as f:
        rebias_labels = pickle.load(f)

    return rebias_labels


def flatten_list(input_list):
    return [x for x_list in input_list for x in x_list]


def flatten_label_list(input_list, labels_list):
    flat_list = flatten_list(input_list)
    return np.array([labels_list.index(y) for y in flat_list]).flatten()


def flatten_tokens(all_vectors, all_labels, lm_tokenizer):
    x = np.array(flatten_list(all_vectors))
    y = np.array(
        [label for sentence_y in all_labels for label in
         lm_tokenizer.convert_tokens_to_ids(sentence_y)]).flatten()
    return x, y
