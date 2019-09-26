import warnings
import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from .log_likelihood import log_zinb_positive, log_nb_positive
from .modules import DecoderSCVI, Encoder, DecoderPoisson, DenseResNet
from .iaf_encoder import EncoderIAF
from .utils import one_hot
import logging

logger = logging.getLogger(__name__)


class IAVAE(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        t: int = 3,
        dropout_rate: float = 5e-2,
        dispersion: str = "gene",
        log_variational: bool = True,
        reconstruction_loss: str = "zinb",
        do_h: bool = False,
        n_blocks: int = 0,
        n_blocks_encoder: int = 0,
        res_connection_decoder: bool = False,
        decoder_do_last_skip: bool = False,
    ):
        """
        EXPERIMENTAL: Posterior functionalities may not be working

        Model does not implement Forward.
        Training should be performed with ratio_loss method

        :param n_input:
        :param n_batch:
        :param n_labels:
        :param n_hidden:
        :param n_latent:
        :param n_layers:
        :param t: Number of autoregressive steps
        :param dropout_rate:
        :param dispersion:
        :param log_variational:
        :param reconstruction_loss:
        """

        super().__init__()
        warnings.warn("EXPERIMENTAL: Posterior functionalities may not be working")
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.res_connection_decoder = res_connection_decoder
        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        else:  # gene-cell
            pass

        # latent space representation
        self.z_encoder = EncoderIAF(
            n_in=n_input,
            n_latent=n_latent,
            n_cat_list=None,
            n_layers=n_layers,
            t=t,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            n_blocks=n_blocks_encoder,
            do_h=do_h,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            prevent_saturation=True,
        )


        # decoder goes from n_latent-dimensional space to n_input-d data
        n_latent_size = n_latent
        if res_connection_decoder:
            size_skip = 2*n_latent if do_h else n_latent
            n_latent_size = n_latent + size_skip
        self.decoder = DecoderSCVI(
            n_latent_size,
            n_input,
            n_cat_list=[n_batch],
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_blocks=n_blocks,
            do_last_skip=decoder_do_last_skip
        )

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        """

        :param x:
        :param batch_index:
        :param y:
        :param n_samples:
        :return:
        """
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        ql_m, ql_v, library = self.l_encoder(x_)
        n_batch = len(ql_m)
        if n_samples > 1:
            last_inputs = None
            # Managing library
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = dist.Normal(ql_m, ql_v.sqrt()).sample()

            # Managing z Latent
            z = torch.zeros(
                n_samples, n_batch, self.n_latent, device=ql_m.device, dtype=ql_m.dtype
            )
            log_qz_x = torch.zeros(
                n_samples, n_batch, device=ql_m.device, dtype=ql_m.dtype
            )
            for idx in range(n_samples):

                outputs = self.z_encoder(x_, y)
                zi, log_qz_x_i = outputs["z"], outputs["qz_x"]
                z[idx, :] = zi
                log_qz_x[idx, :] = log_qz_x_i
        else:
            outputs = self.z_encoder(x_, y)
            z, log_qz_x, last_inputs = outputs["z"], outputs["qz_x"], outputs["last_inp"]

        assert z.shape[0] == library.shape[0], (z.shape, library.shape)
        # library = torch.clamp(library, max=13)

        decoder_inp = z
        if self.res_connection_decoder:
            assert last_inputs is not None
            decoder_inp = torch.cat([z, last_inputs], dim=-1)

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, decoder_inp, library, batch_index, y
        )
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)
        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            z=z,
            log_qz_x=log_qz_x,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    def ratio_loss(
        self, x, local_l_mean, local_l_var, batch_index=None, y=None, return_mean=True, outputs=None, train_library=True
    ):
        if outputs is None:
            outputs = self.inference(x, batch_index=batch_index, y=y, n_samples=1)
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        library = outputs["library"]
        px_rate = outputs["px_rate"]
        z = outputs["z"]
        log_qz_x = outputs["log_qz_x"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]

        # variationnal probas computation
        log_ql_x = dist.Normal(ql_m, torch.sqrt(ql_v)).log_prob(library).sum(dim=-1)

        # priors computation
        log_pz = (
            dist.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)
        )
        log_pl = (
            dist.Normal(local_l_mean, torch.sqrt(local_l_var))
            .log_prob(library)
            .sum(dim=-1)
        )

        # reconstruction proba computation
        log_px_zl = -self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)

        assert (
            log_px_zl.shape
            == log_pl.shape
            == log_pz.shape
            == log_qz_x.shape
            == log_ql_x.shape
        )

        if train_library:
            ratio = (log_px_zl + log_pz + log_pl) - (log_qz_x + log_ql_x)
        else:
            ratio = (log_px_zl + log_pz) - log_qz_x

        if not return_mean:
            return ratio
        elbo = ratio.mean(dim=0)
        return -elbo

    def get_reconstruction_loss(self, x, px_rate, px_r, px_dropout):
        # Reconstruction Loss
        if self.reconstruction_loss == "zinb":
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout)
        elif self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r)
        else:
            raise NotImplementedError
        return reconst_loss

    def iwelbo(
        self,
        x,
        local_l_mean,
        local_l_var,
        batch_index=None,
        y=None,
        k=3,
        single_backward=False,
    ):
        n_batch = len(x)
        log_ratios = torch.zeros(k, n_batch, device="cuda", dtype=torch.float)
        for it in range(k):
            log_ratios[it, :] = self.ratio_loss(
                x,
                local_l_mean,
                local_l_var,
                batch_index=batch_index,
                y=y,
                return_mean=False,
            )

        normalizers, _ = log_ratios.max(dim=0)
        w_tilde = torch.softmax(log_ratios - normalizers, dim=0).detach()
        if not single_backward:
            loss = -(w_tilde * log_ratios).sum(dim=0)
        else:
            selected_k = torch.distributions.Categorical(
                probs=w_tilde.transpose(-1, -2)
            ).sample()
            assert len(selected_k) == n_batch
            loss = -log_ratios[selected_k, torch.arange(n_batch)]
        return loss.mean(dim=0)


class IALogNormalPoissonVAE(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        t: int = 3,
        dropout_rate: float = 5e-2,
        log_variational: bool = True,
        gt_decoder: nn.Module = None,
        do_h: bool = False,
    ):
        """

        """
        self.trained_decoder = gt_decoder is None

        super().__init__()
        warnings.warn("EXPERIMENTAL: Posterior functionalities may not be working")
        self.n_latent = n_latent
        self.log_variational = log_variational
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels

        # decoder goes from n_latent-dimensional space to n_input-d data
        if self.trained_decoder:
            self.decoder = DecoderPoisson(
                n_latent,
                n_input,
                n_cat_list=[n_batch],
                n_layers=n_layers,
                n_hidden=n_hidden,
            )
        else:
            self.decoder = gt_decoder

        # latent space representation
        self.z_encoder = EncoderIAF(
            n_in=n_input,
            n_latent=n_latent,
            n_cat_list=None,
            n_layers=n_layers,
            t=t,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            do_h=do_h,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate
        )

    def inference(self, x, batch_index=None, y=None, n_samples=1):
        """

        :param x:
        :param batch_index:
        :param y:
        :param n_samples:
        :return:
        """
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # Sampling
        ql_m, ql_v, library = self.l_encoder(x_)
        n_batch = len(ql_m)
        if n_samples > 1:
            # Managing library
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = dist.Normal(ql_m, ql_v.sqrt()).sample()

            # Managing z Latent
            z = torch.zeros(
                n_samples, n_batch, self.n_latent, device=ql_m.device, dtype=ql_m.dtype
            )
            log_qz_x = torch.zeros(
                n_samples, n_batch, device=ql_m.device, dtype=ql_m.dtype
            )
            for idx in range(n_samples):
                zi, log_qz_x_i = self.z_encoder(x_, y)
                z[idx, :] = zi
                log_qz_x[idx, :] = log_qz_x_i
        else:
            z, log_qz_x = self.z_encoder(x_, y)

        # library = torch.clamp(library, max=13)
        assert z.shape[0] == library.shape[0]
        # assert z.shape[1] == library.shape[1], 'Different n_batch'

        # px_rate = self.decoder(self.dispersion, z, library, batch_index, y)
        px_rate = self.decoder(z, library, batch_index, y)

        return dict(
            px_rate=px_rate,
            z=z,
            log_qz_x=log_qz_x,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    def ratio_loss(
        self, x, local_l_mean, local_l_var, batch_index=None, y=None, return_mean=True
    ):
        outputs = self.inference(x, batch_index=batch_index, y=y, n_samples=1)
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        library = outputs["library"]
        px_rate = outputs["px_rate"]
        z = outputs["z"]
        log_qz_x = outputs["log_qz_x"]

        # variationnal probas computation
        log_ql_x = dist.Normal(ql_m, torch.sqrt(ql_v)).log_prob(library).sum(dim=-1)

        # priors computation
        log_pz = (
            dist.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z).sum(dim=-1)
        )
        log_pl = (
            dist.Normal(local_l_mean, torch.sqrt(local_l_var))
            .log_prob(library)
            .sum(dim=-1)
        )

        # reconstruction proba computation
        log_px_zl = -self.get_reconstruction_loss(x, px_rate)

        log_ratio = log_px_zl + log_pz - log_qz_x
        # ratio = (
        #     log_px_zl + log_pz + log_pl
        #     - log_qz_x - log_ql_x
        # )
        if not return_mean:
            return log_ratio
        elbo = log_ratio.mean(dim=0)
        return -elbo

    @staticmethod
    def get_reconstruction_loss(x, rate):
        rl = -torch.distributions.Poisson(rate).log_prob(x)
        assert rl.dim() == rate.dim()  # rl should be (n_batch, n_input)
        # or (n_samples, n_batch, n_input)
        return torch.sum(rl, dim=-1)

    def iwelbo(
        self,
        x,
        local_l_mean,
        local_l_var,
        batch_index=None,
        y=None,
        k=3,
        single_backward=False,
    ):
        n_batch = len(x)
        log_ratios = torch.zeros(k, n_batch, device="cuda", dtype=torch.float)
        for it in range(k):
            log_ratios[it, :] = self.ratio_loss(
                x,
                local_l_mean,
                local_l_var,
                batch_index=batch_index,
                y=y,
                return_mean=False,
            )

        normalizers, _ = log_ratios.max(dim=0)
        w_tilde = torch.softmax(log_ratios - normalizers, dim=0).detach()
        if not single_backward:
            loss = -(w_tilde * log_ratios).sum(dim=0)
        else:
            selected_k = torch.distributions.Categorical(
                probs=w_tilde.transpose(-1, -2)
            ).sample()
            assert len(selected_k) == n_batch
            loss = -log_ratios[selected_k, torch.arange(n_batch)]
        return loss.mean(dim=0)
