"""Fine-tune an ImageNet-pretrained ResNet to classify the item count (0-5). Model selection on val only.

Each run trains for --epochs, evaluates val after every epoch and keeps the checkpoint with the best
val accuracy. Test data is not read here (see final_test.py).
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision

from invcount.abid.cache import load_split
from invcount.abid.common import DATA, RESULTS
from invcount.metrics import count_metrics

MODELS = DATA / "models"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


def build(arch, n_classes=6, pretrained=True):
    w = {"resnet18": "IMAGENET1K_V1", "resnet34": "IMAGENET1K_V1"}[arch] if pretrained else None
    net = getattr(torchvision.models, arch)(weights=w)
    net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net


def to_input(xb_uint8, dev):
    x = xb_uint8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return (x - MEAN.to(dev)) / STD.to(dev)


def augment(x):
    # bins are photographed from above: horizontal + vertical flips keep the count; mild brightness/contrast jitter
    n = x.shape[0]
    hf = torch.rand(n, device=x.device) < 0.5
    vf = torch.rand(n, device=x.device) < 0.5
    x = torch.where(hf.view(-1, 1, 1, 1), x.flip(3), x)
    x = torch.where(vf.view(-1, 1, 1, 1), x.flip(2), x)
    b = (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.4
    c = 1 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.4
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - m) * c + m + b


def autocast(dev):
    # bf16 mixed precision: ~1.7x faster ResNet-18 training on the M5 GPU (measured); weights stay fp32
    return torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type in ("mps", "cuda"))


@torch.no_grad()
def predict(net, X, dev, bs=128):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        with autocast(dev):
            out.append(net(to_input(torch.from_numpy(X[i:i + bs]), dev)).float().argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs-run", type=int, default=None, help="stop early (timing runs)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device()
    Xtr, ytr, _ = load_split("train")
    Xva, yva, _ = load_split("val")
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()

    net = build(args.arch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(Xtr) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss()
    tag = f"cnn_{args.arch}_lr{args.lr:g}_wd{args.wd:g}_e{args.epochs}_s{args.seed}"
    hist, best = [], -1
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        t0 = time.time()
        net.train()
        perm = torch.randperm(len(Xtr))
        tot, correct, seen = 0.0, 0, 0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            x = augment(to_input(Xtr_t[idx], dev))
            y = ytr_t[idx].to(dev)
            with autocast(dev):
                out = net(x)
                loss = lossf(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(idx)
        m = count_metrics(yva, predict(net, Xva, dev))
        row = {"epoch": ep, "train_loss": tot / seen, "train_acc": correct / seen, "val_acc": m["accuracy"],
               "val_within1": m["within1"], "val_mae": m["mae"], "epoch_s": round(time.time() - t0, 1)}
        hist.append(row)
        print(row, flush=True)
        if m["accuracy"] > best:
            best = m["accuracy"]
            torch.save(net.state_dict(), MODELS / f"{tag}.pt")
        if args.max_epochs_run and ep + 1 >= args.max_epochs_run:
            break
    bi = int(np.argmax([h["val_acc"] for h in hist]))
    json.dump({"args": vars(args), "history": hist, "best_epoch": bi, "best_val_acc": hist[bi]["val_acc"]},
              open(RESULTS / f"{tag}.json", "w"), indent=1)
    print("BEST", tag, hist[bi])


if __name__ == "__main__":
    main()
