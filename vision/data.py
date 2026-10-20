"""Datasets: Oxford Flowers-102 (plants, ~330MB), Food-101 subset, or any ImageFolder directory."""
import os

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T

from vision.flower_names import FLOWER_NAMES
from vision.predict import MEAN, STD


def train_transform(size):
    return T.Compose([T.RandomResizedCrop(size, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transform(size):
    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


class Remap(Dataset):
    """View of `ds` restricted to `indices`, with labels remapped through `mapping`."""

    def __init__(self, ds, indices, mapping):
        self.ds, self.indices, self.mapping = ds, list(indices), mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.ds[self.indices[i]]
        return x, self.mapping[y]


def stratified_split(labels, fractions=(0.7, 0.15, 0.15), seed=0):
    idx = np.arange(len(labels))
    tr, rest = train_test_split(idx, train_size=fractions[0], stratify=labels, random_state=seed)
    val, te = train_test_split(rest, train_size=fractions[1] / (fractions[1] + fractions[2]),
                               stratify=np.asarray(labels)[rest], random_state=seed)
    return tr, val, te


def _identity(n):
    return {i: i for i in range(n)}


def get_datasets(name, size, root="data", data_dir=None, n_classes=20, max_per_class=200, seed=0):
    """Returns dict(train, val, test, raw_test, classes). `raw_test` yields PIL images (no transform)."""
    tr_tf, ev_tf = train_transform(size), eval_transform(size)

    if name == "flowers102":
        make = lambda split, tf: datasets.Flowers102(root, split=split, transform=tf, download=True)
        assert len(FLOWER_NAMES) == 102
        return dict(train=make("train", tr_tf), val=make("val", ev_tf), test=make("test", ev_tf),
                    raw_test=make("test", None), classes=list(FLOWER_NAMES))

    if name == "food101":
        make = lambda split, tf: datasets.Food101(root, split=split, transform=tf, download=True)
        base, rng = make("train", None), np.random.default_rng(seed)
        chosen = sorted(rng.choice(len(base.classes), n_classes, replace=False).tolist())
        mapping = {c: i for i, c in enumerate(chosen)}
        classes = [base.classes[c].replace("_", " ") for c in chosen]

        def pick(ds, cap=None):
            labels, out, seen = ds._labels, [], {c: 0 for c in chosen}
            for i, y in enumerate(labels):
                if y in mapping and (cap is None or seen[y] < cap):
                    out.append(i)
                    seen[y] += 1
            return out

        pool = pick(base, max_per_class)
        pool_y = [mapping[base._labels[i]] for i in pool]
        tr_i, val_i = train_test_split(np.arange(len(pool)), test_size=0.15, stratify=pool_y, random_state=seed)
        train_ds, val_ds, test_base = make("train", tr_tf), make("train", ev_tf), make("test", ev_tf)
        return dict(train=Remap(train_ds, [pool[i] for i in tr_i], mapping),
                    val=Remap(val_ds, [pool[i] for i in val_i], mapping),
                    test=Remap(test_base, pick(test_base), mapping),
                    raw_test=Remap(make("test", None), pick(test_base), mapping), classes=classes)

    if name == "folder":
        if data_dir is None:
            raise SystemExit("--data-dir is required for --dataset folder")
        if all(os.path.isdir(os.path.join(data_dir, s)) for s in ("train", "val", "test")):
            f = lambda s, tf: datasets.ImageFolder(os.path.join(data_dir, s), tf)
            return dict(train=f("train", tr_tf), val=f("val", ev_tf), test=f("test", ev_tf),
                        raw_test=f("test", None), classes=f("train", None).classes)
        base = datasets.ImageFolder(data_dir)                      # one folder per class -> split randomly
        tr_i, val_i, te_i = stratified_split(base.targets, seed=seed)
        ident = _identity(len(base.classes))
        mk = lambda tf: datasets.ImageFolder(data_dir, tf)
        return dict(train=Remap(mk(tr_tf), tr_i, ident), val=Remap(mk(ev_tf), val_i, ident),
                    test=Remap(mk(ev_tf), te_i, ident), raw_test=Remap(mk(None), te_i, ident), classes=base.classes)

    raise SystemExit(f"Unknown dataset {name}")
