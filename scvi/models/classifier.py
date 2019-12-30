from torch import nn as nn
from torch import distributions

from scvi.models.modules import FCLayers


class Classifier(nn.Module):
    def __init__(
        self,
        n_input,
        n_hidden=128,
        n_labels=10,
        n_layers=1,
        dropout_rate=0.1,
        use_batch_norm=True
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            FCLayers(n_in=n_input, n_out=n_hidden, n_layers=n_layers,
                     n_hidden=n_hidden, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm),
            nn.Linear(n_hidden, n_labels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        probas = self.classifier(x)
        probas = probas + 1e-8
        probas = probas / probas.sum(-1, keepdim=True)
        return probas


class GumbelClassifier(Classifier):
    def __init__(self, n_input, n_hidden=128, n_labels=10, n_layers=1, dropout_rate=0.1):
        super().__init__(
            n_input,
            n_hidden=n_hidden,
            n_labels=n_labels,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )

    def forward(self, x, temp=0.1):
        pi = super().forward(x)
        yi = distributions.Gumbel(loc=0., scale=1.).sample(*pi.shape)

        logits = (pi.log() + yi) / temp
        probas = nn.Softmax(logits)
        return probas
