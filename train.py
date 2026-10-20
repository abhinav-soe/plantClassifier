"""Fine-tune a pretrained CNN in two phases (head only -> whole network), then evaluate on the test split.
  python train.py                                   # Oxford Flowers-102, ResNet18  (~330MB download)
  python train.py --arch efficientnet_b0 --epochs 10
  python train.py --dataset food101 --classes 20    # Food-101 subset (~5GB download)
  python train.py --dataset folder --data-dir path/to/class_folders   # e.g. PlantVillage from Kaggle"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from vision.data import get_datasets
from vision.model import ARCHS, build_model, set_backbone_trainable, split_params
from vision.predict import get_device, save_checkpoint


def run_epoch(model, loader, device, loss_fn, optimizer=None, scaler=None, use_amp=False):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    probs_all, labels_all = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
            if not train:
                probs_all.append(out.float().softmax(1).cpu().numpy())
                labels_all.append(y.cpu().numpy())
    res = dict(loss=loss_sum / total, acc=correct / total)
    if not train:
        res.update(probs=np.concatenate(probs_all), labels=np.concatenate(labels_all))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flowers102", choices=["flowers102", "food101", "folder"])
    ap.add_argument("--data-dir")
    ap.add_argument("--root", default="data")
    ap.add_argument("--arch", default="resnet18", choices=ARCHS)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--freeze-epochs", type=int, default=2, help="epochs training only the new head")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    ap.add_argument("--backbone-lr", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--classes", type=int, default=20, help="food101: number of classes to use")
    ap.add_argument("--max-per-class", type=int, default=200, help="food101: training images per class")
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device, os_out = get_device(), a.out
    os.makedirs(os_out, exist_ok=True)
    ds = get_datasets(a.dataset, a.img_size, a.root, a.data_dir, a.classes, a.max_per_class, a.seed)
    classes = ds["classes"]
    mk = lambda d, shuffle: DataLoader(d, a.batch_size, shuffle=shuffle, num_workers=a.workers,
                                       pin_memory=device.type == "cuda")
    train_dl, val_dl, test_dl = mk(ds["train"], True), mk(ds["val"], False), mk(ds["test"], False)
    print(f"device={device} | classes={len(classes)} | train/val/test = "
          f"{len(ds['train'])}/{len(ds['val'])}/{len(ds['test'])}")

    model = build_model(a.arch, len(classes), pretrained=not a.no_pretrained).to(device)
    backbone, head_p = split_params(model, a.arch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_val, t0 = [], -1.0, time.time()
    optimizer = scheduler = None
    for epoch in range(a.epochs):
        if epoch == 0 or epoch == a.freeze_epochs:              # (re)build optimiser at each phase change
            frozen = epoch < a.freeze_epochs
            set_backbone_trainable(model, a.arch, not frozen)
            groups = [dict(params=head_p, lr=a.lr)] + ([] if frozen else [dict(params=backbone, lr=a.backbone_lr)])
            optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
            remaining = (a.freeze_epochs if frozen else a.epochs - a.freeze_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining, 1))
            print(f"-- phase: {'head only' if frozen else 'full fine-tuning'}")
        tr = run_epoch(model, train_dl, device, loss_fn, optimizer, scaler, use_amp)
        scheduler.step()
        va = run_epoch(model, val_dl, device, loss_fn)
        history.append(dict(epoch=epoch + 1, train_loss=tr["loss"], train_acc=tr["acc"],
                            val_loss=va["loss"], val_acc=va["acc"]))
        print(f"epoch {epoch + 1:2d}/{a.epochs}  train {tr['loss']:.3f}/{tr['acc']:.3f}  "
              f"val {va['loss']:.3f}/{va['acc']:.3f}")
        if va["acc"] > best_val:
            best_val = va["acc"]
            save_checkpoint(f"{os_out}/best.pt", model, classes, a.arch, a.img_size)
    pd.DataFrame(history).to_csv(f"{os_out}/history.csv", index=False)

    # ---------------- final evaluation on the untouched test split ----------------
    ck = torch.load(f"{os_out}/best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])
    te = run_epoch(model, test_dl, device, loss_fn)
    probs, y = te["probs"], te["labels"]
    pred = probs.argmax(1)
    k = min(5, max(2, len(classes) // 3))
    top5 = float(np.mean([y[i] in np.argsort(-probs[i])[:k] for i in range(len(y))]))
    rep = classification_report(y, pred, labels=range(len(classes)), target_names=classes,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    off = [(int(cm[i, j]), i, j) for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j]]
    conf = [dict(true=classes[i], predicted=classes[j], count=c) for c, i, j in sorted(off, reverse=True)[:10]]
    metrics = dict(dataset=a.dataset, arch=a.arch, pretrained=not a.no_pretrained, epochs=a.epochs, device=str(device),
                   n_train=len(ds["train"]), n_val=len(ds["val"]), n_test=len(ds["test"]),
                   best_val_accuracy=best_val, test_accuracy=float((pred == y).mean()),
                   macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)), topk_accuracy=top5, k=k,
                   per_class={c: dict(f1=rep[c]["f1-score"], support=int(rep[c]["support"])) for c in classes},
                   top_confusions=conf, train_seconds=round(time.time() - t0))
    json.dump(metrics, open(f"{os_out}/metrics.json", "w"), indent=2)

    # a few raw test images for the app's "try a sample" picker
    os.makedirs(f"{os_out}/samples", exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for n, i in enumerate(rng.choice(len(ds["raw_test"]), min(8, len(ds["raw_test"])), replace=False)):
        img, label = ds["raw_test"][int(i)]
        img.thumbnail((500, 500))
        img.convert("RGB").save(f"{os_out}/samples/{n}__{classes[label].replace(' ', '_')}.jpg", quality=88)

    print(f"\nTEST  accuracy {metrics['test_accuracy']:.3f} | macro-F1 {metrics['macro_f1']:.3f} | "
          f"top-{k} {top5:.3f}   ({metrics['train_seconds']}s)")


if __name__ == "__main__":
    main()
