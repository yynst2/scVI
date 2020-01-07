from typing import Sequence
import logging

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, Categorical, kl_divergence as kl, Bernoulli

from scvi.models.classifier import Classifier, GumbelClassifier
from scvi.models.modules import Decoder, Encoder, EncoderIAF, BernoulliDecoder
from scvi.models.regular_modules import (
    EncoderA,
    DecoderA,
    ClassifierA,
    BernoulliDecoderA,
)
from scvi.models.utils import broadcast_labels
from scvi.models.vae import VAE

logger = logging.getLogger(__name__)


class SemiSupervisedVAE(nn.Module):
    r"""

    """

    def __init__(
        self,
        n_input: int,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        y_prior=None,
        classifier_parameters: dict = dict(),
        prevent_saturation: bool = False,
        iaf_t=0,
        do_batch_norm=False,
    ):
        # TODO: change architecture so that it match something existing
        super().__init__()

        self.n_labels = n_labels
        # Classifier takes n_latent as input
        self.classifier = ClassifierA(
            n_latent,
            n_output=n_labels,
            do_batch_norm=do_batch_norm,
            dropout_rate=dropout_rate,
        )

        self.encoder_z1 = EncoderA(
            # n_input=n_input + n_labels,
            n_input=n_input,
            n_output=n_latent,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            do_batch_norm=do_batch_norm,
        )

        # q(z_2 \mid z_1, c)
        self.encoder_z2_z1 = EncoderA(
            n_input=n_latent + n_labels,
            n_output=n_latent,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            do_batch_norm=do_batch_norm,
        )

        self.decoder_z1_z2 = DecoderA(
            n_input=n_latent + n_labels, n_output=n_latent, n_hidden=n_hidden,
        )

        self.x_decoder = BernoulliDecoderA(
            n_input=n_latent, n_output=n_input, do_batch_norm=do_batch_norm,
        )

        y_prior_probs = (
            y_prior
            if y_prior is not None
            else (1 / n_labels) * torch.ones(1, n_labels, device="cuda")
        )

        self.y_prior = Categorical(probs=y_prior_probs)

        self.encoder_params = filter(
            lambda p: p.requires_grad,
            list(self.classifier.parameters())
            + list(self.encoder_z1.parameters())
            + list(self.encoder_z2_z1.parameters()),
        )

        self.decoder_params = filter(
            lambda p: p.requires_grad,
            list(self.decoder_z1_z2.parameters()) + list(self.x_decoder.parameters()),
        )

        self.all_params = filter(lambda p: p.requires_grad, list(self.parameters()))

    def classify(self, x, n_samples=1):
        # n_cat = self.n_labels
        # n_batch = len(x)
        # inp = torch.cat((x, torch.zeros(n_batch, n_cat, device=x.device)), dim=-1)
        inp = x
        q_z1 = self.encoder_z1(inp, n_samples=n_samples)
        z = q_z1["latent"]

        w_y = self.classifier(z)
        return w_y

    def get_latents(self, x, y=None):
        pass

    def inference(self, x, y=None, n_samples=1, reparam=True):
        """
        Dimension choice
            (n_categories, n_is, n_batch, n_latent)

            log_q
            (n_categories, n_is, n_batch)
        """
        n_batch = len(x)
        n_cat = self.n_labels

        if y is None:
            # deal with the case that the latent factorization is not the same in M1 and M2
            ys = (
                torch.eye(n_cat, device=x.device)
                .view(n_cat, 1, 1, n_cat)
                .expand(n_cat, n_samples, n_batch, n_cat)
            )
            # inp = torch.cat((x, torch.zeros(n_batch, n_cat, device=x.device)), dim=-1)
        else:
            ys = torch.cuda.FloatTensor(n_batch, n_cat)
            ys.zero_()
            ys.scatter_(1, y.view(-1, 1), 1)
            ys = ys.view(1, n_batch, n_cat).expand(n_samples, n_batch, n_cat)
            # inp = torch.cat((x, ys[0, :]), dim=-1)
        inp = x
        q_z1 = self.encoder_z1(inp, n_samples=n_samples, reparam=reparam, squeeze=False)
        # if not self.do_iaf:
        qz1_m = q_z1["q_m"]
        qz1_v = q_z1["q_v"]
        z1 = q_z1["latent"]
        assert z1.dim() == 3

        log_qz1_x = Normal(qz1_m, qz1_v.sqrt()).log_prob(z1).sum(-1)

        # Broadcast labels if necessary
        if y is None:
            n_latent = z1.shape[-1]
            z1s = z1.view(1, n_samples, n_batch, n_latent).expand(
                n_cat, n_samples, n_batch, n_latent
            )
            y_int = ys.argmax(dim=-1)

            # Dealing with qc
            qc_z1 = self.classifier(z1).permute(2, 0, 1)

        else:
            z1s = z1
            y_int = y

            # Dealing with qc
            qc_z1 = self.classifier(z1)[ys.bool()].view(n_samples, n_batch)

            # qc_z1_check = (self.classifier(z1) * ys.byte()).max(-1).values
            # assert (qc_z1_check == qc_z1).all()

        qc_z1_all_probas = self.classifier(z1)
        log_qc_z1 = qc_z1.log()

        log_pc = self.y_prior.log_prob(y_int)
        pc = log_pc.exp()

        z1_y = torch.cat([z1s, ys], dim=-1)
        q_z2_z1 = self.encoder_z2_z1(z1_y, n_samples=1, reparam=reparam)
        z2 = q_z2_z1["latent"]
        qz2_z1_m = q_z2_z1["q_m"]
        qz2_z1_v = q_z2_z1["q_v"]
        log_qz2_z1 = Normal(q_z2_z1["q_m"], q_z2_z1["q_v"].sqrt()).log_prob(z2).sum(-1)

        z2_y = torch.cat([z2, ys], dim=-1)
        pz1_z2m, pz1_z2_v = self.decoder_z1_z2(z2_y)
        log_pz1_z2 = Normal(pz1_z2m, pz1_z2_v.sqrt()).log_prob(z1).sum(-1)

        log_pz2 = Normal(torch.zeros_like(z2), torch.ones_like(z2)).log_prob(z2).sum(-1)

        px_z_loc = self.x_decoder(z1)
        log_px_z = Bernoulli(px_z_loc).log_prob(x).sum(-1)

        generative_density = log_pz2 + log_pc + log_pz1_z2 + log_px_z
        variational_density = log_qz1_x + log_qz2_z1
        log_ratio = generative_density - variational_density

        variables = dict(
            z1=z1,
            ys=ys,
            z2=z2,
            qz1_m=qz1_m,
            qz1_v=qz1_v,
            qz2_z1_m=qz2_z1_m,
            qz2_z1_v=qz2_z1_v,
            pz1_z2m=pz1_z2m,
            pz1_z2_v=pz1_z2_v,
            px_z_m=px_z_loc,
            log_qz1_x=log_qz1_x,
            qc_z1=qc_z1,
            log_qc_z1=log_qc_z1,
            log_qz2_z1=log_qz2_z1,
            log_pz2=log_pz2,
            log_pc=log_pc,
            pc=pc,
            log_pz1_z2=log_pz1_z2,
            log_px_z=log_px_z,
            generative_density=generative_density,
            variational_density=variational_density,
            log_ratio=log_ratio,
            qc_z1_all_probas=qc_z1_all_probas,
        )
        return variables

    def forward(
        self, x, loss_type="ELBO", y=None, n_samples=1, reparam=True,
    ):
        """

        n_categories, n_is, n_batch
        """
        is_labelled = False if y is None else True

        vars = self.inference(x=x, y=y, n_samples=n_samples, reparam=reparam)

        log_ratio = vars["generative_density"] - vars["log_qz1_x"] - vars["log_qz2_z1"]
        if not is_labelled:
            # Unlabelled case: c latent variable
            log_ratio -= vars["log_qc_z1"]

        everything_ok = log_ratio.ndim == 2 if is_labelled else log_ratio.ndim == 3
        assert everything_ok

        if loss_type == "ELBO":
            loss = self.elbo(log_ratio, is_labelled, **vars)
        elif loss_type == "CUBO":
            loss = self.cubo(log_ratio, is_labelled, **vars)
        elif loss_type == "REVKL":
            loss = self.forward_kl(log_ratio, is_labelled=is_labelled, **vars)
        elif loss_type == "IWELBO":
            loss = None
        else:
            raise ValueError("Mode {} not recognized".format(loss_type))
        if torch.isnan(loss).any() or not torch.isfinite(loss).any():
            print("NaN loss")

        return loss

    @staticmethod
    def elbo(log_ratios, is_labelled, **kwargs):
        if is_labelled:
            loss = -log_ratios.mean()
        else:
            categorical_weights = kwargs["qc_z1"]
            loss = (categorical_weights * log_ratios).sum(0)
            loss = -loss.mean()
        return loss

    @staticmethod
    def iwelbo(log_ratios, is_labelled, evaluate=False, **kwargs):
        if is_labelled:
            # (n_samples, n_batch)
            assert not evaluate
            ws = torch.softmax(log_ratios, dim=0)
            loss = -(ws.detach() * log_ratios).sum(dim=0)
        else:
            if evaluate:
                q_c = kwargs["log_qc_z1"].exp()
                n_samples = log_ratios.shape[1]
                res = q_c * (
                    torch.logsumexp(log_ratios, dim=1, keepdim=True) - np.log(n_samples)
                )
                res = res.mean(1)
                return res.sum(0)
            # loss =
            raise NotImplementedError
        return loss

    @staticmethod
    def forward_kl(log_ratios, is_labelled, **kwargs):
        # TODO Triple check
        if is_labelled:
            ws = torch.softmax(log_ratios, dim=0)
            rev_kl = ws.detach() * (-1) * kwargs["sum_log_q"]
            return rev_kl.sum(dim=0)
        else:
            log_pz1z2x_c = kwargs["log_pz1_z2"] + kwargs["log_pz2"] + kwargs["log_px_z"]
            log_z1z2_xc = kwargs["log_qz1_x"] + kwargs["log_qz2_z1"]
            # Shape (n_cat, n_is, n_batch)
            importance_weights = torch.softmax(log_pz1z2x_c - log_z1z2_xc, dim=1)
            rev_kl = (importance_weights.detach() * log_ratios).sum(dim=1)
            categorical_weights = kwargs["pc"].detach()
            print(categorical_weights.shape)
            # assert (categorical_weights[:, 0] == categorical_weights[:, 1]).all()
            categorical_weights = categorical_weights.mean(1)
            rev_kl = (categorical_weights * rev_kl).sum(1)
            return rev_kl

    @staticmethod
    def cubo(log_ratios, is_labelled, evaluate=False, **kwargs):
        if is_labelled:
            assert not evaluate
            ws = torch.softmax(2 * log_ratios, dim=0)  # Corresponds to squaring
            cubo_loss = ws.detach() * (-1) * log_ratios
            return cubo_loss.mean(dim=0)
        else:
            # Prefer to deal this case separately to avoid mistakes
            if evaluate:
                # q_c = kwargs["log_qc_z1"].exp()
                # n_samples_mc = log_ratios.shape[1]
                # res = q_c * (
                #     torch.logsumexp(2 * log_ratios, dim=1, keepdim=True)
                #     - np.log(n_samples_mc)
                # )
                # res = res.mean(1)
                # return res.sum(0)

                log_q_c = vals["log_qc_z1"]
                n_cat, n_samples, n_batch = log_ratios.shape
                res = torch.logsumexp((2 * log_ratios + log_q_c).view(n_cat*n_samples, n_batch), dim=0, keepdim=False)
                res = res - np.log(n_samples)
                return res

            assert log_ratios.dim() == 3
            log_qc_z1 = kwargs["log_qc_z1"]
            log_ratios += 0.5 * log_qc_z1
            ws = torch.softmax(2 * log_ratios, dim=1)
            cubo_loss = ws.detach() * (-1) * log_ratios
            cubo_loss = cubo_loss.mean(dim=1)  # samples
            cubo_loss = cubo_loss.sum(dim=0)  # cats
            return cubo_loss
