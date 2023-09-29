import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class TaskClassifier(nn.Module):
    def __init__(self, args):
        super(TaskClassifier, self).__init__()
        self.input_dim = args.rnn_hidden_dim
        self.embed_dim = args.rnn_hidden_dim
        self.out_dim = args.n_tasks

        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim), nn.ReLU())
        self.fc2 = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, sa_embeddings):
        x1 = self.fc1(sa_embeddings)
        pred_task_id = self.fc2(x1)
        return pred_task_id


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambd), None, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class Discriminator(nn.Module):
    def __init__(self, args, grl=True):
        super(Discriminator, self).__init__()
        self.input_dim = args.rnn_hidden_dim
        self.embed_dim = args.rnn_hidden_dim
        self.out_dim = args.n_tasks
        self.grl = grl
        self.mlp = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim), nn.ReLU(),
                                 nn.Linear(self.embed_dim, self.out_dim)
        )
        self.lambd = 0.0

    def set_lambd(self, epoch, num_epoch):
        p = epoch / num_epoch
        grl_weight = 1.0
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * grl_weight
        self.lambd = alpha

    def forward(self, sa_embeddings):
        if self.grl:
            sa_embeddings = grad_reverse(sa_embeddings, self.lambd)
        pred_tasks = self.mlp(sa_embeddings)
        return pred_tasks