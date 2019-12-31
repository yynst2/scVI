import logging
from itertools import cycle

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from scvi.dataset import MnistDataset
from scvi.models import SemiSupervisedVAE

logger = logging.getLogger(__name__)


class MnistTrainer:
    def __init__(
        self,
        dataset: MnistDataset,
        model: SemiSupervisedVAE,
        batch_size: int = 128,
        use_cuda=True,
    ):
        self.dataset = dataset
        self.model = model
        self.train_loader = DataLoader(
            self.dataset.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_cuda,
        )
        self.train_annotated_loader = DataLoader(
            self.dataset.train_dataset_labelled,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_cuda,
        )
        self.test_loader = DataLoader(
            self.dataset.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=use_cuda,
        )
        self.cross_entropy_fn = CrossEntropyLoss()

        self.iterate = 0
        self.metrics = dict(
            train_theta_wake=[], train_phi_wake=[], train_phi_sleep=[], train_loss=[]
        )

    def train(
        self,
        n_epochs,
        lr=1e-3,
        overall_loss: bool = None,
        wake_theta: str = "ELBO",
        wake_psi: str = "ELBO",
        n_samples: int = 1,
        classification_ratio: float = 50.0,
        update_mode: str = "all",
    ):
        assert update_mode in ["all", "alternate"]
        optim = None
        optim_gen = None
        optim_var_wake = None
        if overall_loss is not None:
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            optim = Adam(params, lr=lr)
            logger.info("Monobjective using {} loss".format(overall_loss))
        else:
            params_gen = filter(
                lambda p: p.requires_grad,
                list(self.model.decoder_z1_z2.parameters())
                + list(self.model.x_decoder.parameters()),
            )
            optim_gen = Adam(params_gen, lr=lr)

            params_var = filter(
                lambda p: p.requires_grad,
                list(self.model.classifier.parameters())
                + list(self.model.encoder_z1.parameters())
                + list(self.model.encoder_z2_z1.parameters()),
            )
            optim_var_wake = Adam(params_var, lr=lr)
            logger.info(
                "Multiobjective training using {} / {}".format(wake_theta, wake_psi)
            )

        for epoch in tqdm(range(n_epochs)):
            for (tensor_all, tensor_superv) in zip(
                self.train_loader, cycle(self.train_annotated_loader)
            ):
                if overall_loss is not None:
                    loss = self.loss(
                        tensor_all,
                        tensor_superv,
                        loss_type=overall_loss,
                        n_samples=n_samples,
                        reparam=True,
                        classification_ratio=classification_ratio,
                        mode=update_mode,
                    )
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    if self.iterate % 100 == 0:
                        self.metrics["train_loss"].append(loss.item())
                else:
                    # Wake theta
                    theta_loss = self.loss(
                        tensor_all,
                        tensor_superv,
                        loss_type=wake_theta,
                        n_samples=n_samples,
                        reparam=True,
                        classification_ratio=classification_ratio,
                    )
                    optim_gen.zero_grad()
                    theta_loss.backward()
                    optim_gen.step()
                    if self.iterate % 100 == 0:
                        self.metrics["train_theta_wake"].append(theta_loss.item())

                    # Wake phi
                    psi_loss = self.loss(
                        tensor_all,
                        tensor_superv,
                        loss_type=wake_psi,
                        n_samples=n_samples,
                        reparam=True,
                        classification_ratio=classification_ratio,
                    )
                    optim_var_wake.zero_grad()
                    psi_loss.backward()
                    optim_var_wake.step()
                    if self.iterate % 100 == 0:
                        self.metrics["train_phi_wake"].append(psi_loss.item())

                self.iterate += 1

    def loss(
        self,
        tensor_all,
        tensor_superv,
        loss_type,
        n_samples=5,
        reparam=True,
        classification_ratio=50.0,
        mode="all",
    ):
        x_u, _ = tensor_all
        x_s, y_s = tensor_superv

        x_u = x_u.to("cuda")
        x_s = x_s.to("cuda")
        y_s = y_s.to("cuda")

        labelled_fraction = self.dataset.labelled_fraction
        s_every = int(1 / labelled_fraction)
        l_u = self.model.forward(
            x_u, loss_type=loss_type, n_samples=n_samples, reparam=reparam
        )
        l_s = self.model.forward(
            x_s, loss_type=loss_type, y=y_s, n_samples=n_samples, reparam=reparam
        )

        if mode == "all":
            l_s = labelled_fraction * l_s
            j = l_u.mean() + l_s.mean()
        elif mode == "alternate":
            if self.iterate % s_every == 0:
                j = l_s.mean()
            else:
                j = l_u.mean()
        else:
            raise ValueError("Mode {} not recognized".format(mode))

        y_pred = self.model.classify(x_s)
        l_class = self.cross_entropy_fn(y_pred, target=y_s)
        loss = j + classification_ratio * l_class
        return loss

    @torch.no_grad()
    def inference(self, data_loader, keys=None, n_samples: int = 10):
        all_res = dict()
        for tensor_all in data_loader:
            x, y = tensor_all
            res = self.model.inference(x, n_samples=n_samples)
            res["y"] = y
            if keys is not None:
                res = {key: val for (key, val) in res.items() if key in keys}
            all_res = dic_update(all_res, res)
        all_res = dic_concat(all_res)
        return all_res


def dic_update(dic: dict, new_dic: dict):
    """
    Updates dic by appending `new_dict` values
    """
    for key, li in new_dic.items():
        if key in dic:
            dic[key].append(li.cpu())
        else:
            dic[key] = [li.cpu()]
    return dic


def dic_concat(dic: dict, batch_size: int = 128):
    for key, li in dic.items():
        tensor_shape = np.array(li[0].shape)
        dim = np.where(tensor_shape == batch_size)[0][0]
        dim = int(dim)
        dic[key] = torch.cat(li, dim=dim)
    return dic
