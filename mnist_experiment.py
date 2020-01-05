import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score
from tqdm.auto import tqdm

from scvi.dataset import MnistDataset
from scvi.inference import MnistTrainer
from scvi.models import PSIS
from scvi.models import SemiSupervisedVAE

NUM = 500
N_EXPERIMENTS = 10
labelled_proportions = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
labelled_proportions = labelled_proportions / labelled_proportions.sum()
labelled_fraction = 0.05

n_input = 28 * 28
n_labels = 9

CLASSIFICATION_RATIO = 50.
N_EVAL_SAMPLES = 15
# N_EPOCHS = 100
N_EPOCHS = 100
LR = 1e-4

dataset = MnistDataset(
    labelled_fraction=labelled_fraction,
    labelled_proportions=labelled_proportions,
    root="/home/pierre/scVI/tests/mnist",
    download=True,
    do_1d=True,
    test_size=0.
    # test_size=0.6
)

print("train all examples", len(dataset.train_dataset.tensors[0]))
print("train labelled examples", len(dataset.train_dataset_labelled.tensors[0]))

scenarios = [  # WAKE updates
    ("ELBO", "ELBO", "ELBO", None, ),
    (None, "ELBO", "CUBO", None, ),
    # (None, "ELBO", "REVKL", None,),
]

df = []


# Utils functions
def compute_reject_score(y_true: np.ndarray, y_pred: np.ndarray, num=20):
    n_examples, n_pos_classes = y_pred.shape

    assert np.unique(y_true).max() == (n_pos_classes - 1) + 1
    thetas = np.linspace(0.1, 1., num=num)
    res = dict(
        precision_discovery=np.zeros(num),
        recall_discovery=np.zeros(num),
        accuracy=np.zeros(num),
        thresholds=thetas,
    )

    for idx, theta in enumerate(thetas):
        y_pred_theta = y_pred.argmax(1)
        reject = y_pred.max(1) <= theta
        y_pred_theta[reject] = (n_pos_classes - 1) + 1

        res["accuracy"][idx] = accuracy_score(y_true, y_pred_theta)

        y_true_discovery = y_true == (n_pos_classes - 1) + 1
        y_pred_discovery = y_pred_theta == (n_pos_classes - 1) + 1
        res["precision_discovery"][idx] = precision_score(y_true_discovery, y_pred_discovery)
        res["recall_discovery"][idx] = recall_score(y_true_discovery, y_pred_discovery)
    return res


# Main script
for scenario in tqdm(scenarios):
    overall_training, loss_gen, loss_wvar, loss_svar = scenario
    iwelbo = []
    cubo = []
    khat = []
    m_accuracy_arr = []
    m_ap_arr = []
    m_recall_arr = []
    auc_pr_arr = []

    for t in tqdm(range(N_EXPERIMENTS)):
        mdl = SemiSupervisedVAE(
            n_input=n_input,
            n_labels=n_labels,
            n_latent=50,
            n_hidden=500,
            n_layers=1,
            do_batch_norm=True,
        )
        mdl = mdl.cuda()
        trainer = MnistTrainer(dataset=dataset, model=mdl, use_cuda=True, batch_size=256)
        trainer.train(
            n_epochs=N_EPOCHS,
            lr=LR,
            overall_loss=overall_training,
            wake_theta=loss_gen,
            wake_psi=loss_wvar,
            classification_ratio=CLASSIFICATION_RATIO,
            update_mode="all"
        )

        # Eval
        with torch.no_grad():
            train_res = trainer.inference(
                trainer.train_loader,
                keys=[
                    "qc_z1_all_probas",
                    "y",
                    "CUBO",
                    "IWELBO",
                    "log_ratios",
                ],
                n_samples=N_EVAL_SAMPLES,
            )
        y_pred = train_res["qc_z1_all_probas"].mean(0).numpy()
        y_true = train_res["y"].numpy()

        # Choice right now: all log-ratios related metrics are computed in the unsupervised case

        # Precision / Recall for discovery class
        # And accuracy
        res_baseline = compute_reject_score(y_true=y_true, y_pred=y_pred, num=NUM)
        m_accuracy = res_baseline["accuracy"]
        m_ap = res_baseline["precision_discovery"]
        m_recall = res_baseline["recall_discovery"]
        auc_pr = np.trapz(
            x=res_baseline["recall_discovery"],
            y=res_baseline["precision_discovery"],
        )

        m_accuracy_arr.append(m_accuracy)
        m_ap_arr.append(m_ap)
        m_recall_arr.append(m_recall)
        auc_pr_arr.append(auc_pr)

        # Cubo / Iwelbo
        cubo_sam = train_res["CUBO"]
        cubo.append(cubo_sam.mean())

        # k_hat
        log_ratios = train_res["log_ratios"]
        ratios = log_ratios.exp().cpu().numpy()
        psis = PSIS(num_samples=N_EVAL_SAMPLES)
        psis.fit(ratios)

        shapes = psis.shape
        khat.append(np.array(shapes).mean())

    res = {
        "CONFIGURATION": scenario,
        "OVERALL_TRAINING": overall_training,
        "LOSS_GEN": loss_gen,
        "LOSS_WVAR": loss_wvar,
        "LOSS_SVAR": loss_svar,
        "IWELBO": (np.mean(iwelbo), np.std(iwelbo)),
        "IWELBO_SAMPLES": np.array(iwelbo),
        "CUBO": (np.mean(cubo), np.std(cubo)),
        "CUBO_SAMPLES": np.array(cubo),
        "KHAT": np.array(khat),
        "M_ACCURACY": np.array(m_accuracy_arr),
        "MEAN_AP": np.array(m_ap_arr),
        "MEAN_RECALL": np.array(m_recall_arr),
        "AUC": np.array(auc_pr_arr),
    }
    df.append(res)

df = pd.DataFrame(df)
df.to_csv("simu_gaussian_res_paper.csv", sep="\t")
