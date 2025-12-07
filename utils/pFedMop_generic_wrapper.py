# pfedmoap_generic_wrappers.py
import os, sys
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.data.datasets.build import build_dataset
import torch


def _as_client_lists(ds, cfg):
    """Return [ [Datum,...], [Datum,...], ... ] for train/test, building if needed."""
    def _ensure_lists(name, value, fallback_source):
        # already good: list of per-client lists
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
            return value

        # list but not lists of Datum -> rebuild
        if isinstance(value, list) and (len(value) == 0 or not isinstance(value[0], list)):
            pass  # fall through to rebuild

        # dict of user->list
        if isinstance(value, dict):
            keys = sorted(value.keys())
            return [value[k] for k in keys]

        # anything else (None/int/str/whatever): rebuild
        src = getattr(ds, fallback_source)  # e.g., "train" or "test"
        built = ds.generate_federated_dataset(
            src, num_users=cfg.DATASET.USERS,
            is_iid=cfg.DATASET.IID, repeat_rate=cfg.DATASET.REPEATRATE
        )
        return built

    # prefer precomputed federated_* if valid, else generate from train/test
    fed_train = getattr(ds, "federated_train_x", None)
    fed_test  = getattr(ds, "federated_test_x",  None)

    train_lists = _ensure_lists("federated_train_x", fed_train, "train")
    test_lists  = _ensure_lists("federated_test_x",  fed_test,  "test")

    return train_lists, test_lists


def build_pfedmoap_dataset(dataset_name, root, users, batch_size, preprocess,
                           iid=False, repeat_rate=0.0, use_all=True):
    cfg = get_cfg_default()
    cfg.defrost(); cfg.set_new_allowed(True); cfg.DATASET.set_new_allowed(True)

    # cfg.DATASET.ROOT = root
    cfg.DATASET.ROOT = "/home/pal194/FedAPT/data"  # or just: cfg.DATASET.ROOT = root
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.USERS = users
    cfg.DATASET.IID = iid
    cfg.DATASET.REPEATRATE = repeat_rate
    cfg.DATASET.USEALL = bool(use_all)
    # some dataset files read these:
    if not hasattr(cfg.DATASET, "SUBSAMPLE_CLASSES"):        cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    if not hasattr(cfg.DATASET, "SUBSAMPLE_CLASSES_RANGE"):  cfg.DATASET.SUBSAMPLE_CLASSES_RANGE = None

    # sanity checks for layout
    if dataset_name == "Caltech101":
        exp = "/home/pal194/FedAPT/data/caltech101/101_ObjectCategories"
        assert os.path.isdir(exp), f"Missing dir: {exp}"
    if dataset_name == "OxfordFlowers":
        base = "/home/pal194/FedAPT/data/oxford_flowers"
        for p in ["imagelabels.mat", "setid.mat", "cat_to_name.json", "jpg"]:
            assert os.path.exists(os.path.join(base, p)), f"Missing: {os.path.join(base,p)}"


    ds = build_dataset(cfg)  # DON'T freeze before this

    # --- new: make sure we actually have per-client Datum lists ---
    fed_train_lists, fed_test_lists = _as_client_lists(ds, cfg)

    # tiny sanity print (1x)
    if isinstance(fed_train_lists, list):
        print(f"[pfedmoap] users={len(fed_train_lists)}  ex0={len(fed_train_lists[0]) if fed_train_lists else 0}")

    class DatumListWrapper(torch.utils.data.Dataset):
        def __init__(self, datum_list, preprocess):
            self.items = datum_list
            self.preprocess = preprocess
        def __len__(self): 
            # guard against accidental ints/None
            if not isinstance(self.items, (list, tuple)):
                raise TypeError(f"Expected list of Datum, got {type(self.items)}")
            return len(self.items)
        # def __getitem__(self, i):
        #     d = self.items[i]
        #     img = Image.open(d.impath).convert("RGB")
        #     return self.preprocess(img), d.label
        def __getitem__(self, i):
            d = self.items[i]
            img = Image.open(d.impath).convert("RGB")
            img = self.preprocess(img)
            # d usually has .label and .classname
            cname = getattr(d, "classname", None)
            return img, d.label, cname


    train_loaders = [DataLoader(DatumListWrapper(x, preprocess),
                                batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
                     for x in fed_train_lists]
    test_loaders  = [DataLoader(DatumListWrapper(x, preprocess),
                                batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
                     for x in fed_test_lists]

    return train_loaders, test_loaders, ds.classnames
