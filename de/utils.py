from functools import partial
from scvi.dataset import PowSimSynthetic, SymSimDataset, Dataset10X


name_to_dataset = {
    'powsimr': PowSimSynthetic,
    'symsim': partial(SymSimDataset, save_path='/home/pierre/scVI/data/symsim'),
    'mouse_vs_human': partial(Dataset10X, filename='hgmm_5k_v3')
}
