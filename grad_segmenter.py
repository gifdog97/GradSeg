import argparse
import os
from typing import Any, List

import numpy as np
import numpy.typing as npt
import torch
from sklearn.linear_model import LogisticRegression, Ridge

import data_loader

parser = argparse.ArgumentParser(description="GradSeg word segmentation")

parser.add_argument(
    "--train_path",
    type=str,
    default="/cs/labs/yedid/yedid/wordcluster/buckeye_processed/train",
    help="dir of .wav, .wrd files for training data",
)
parser.add_argument(
    "--val_path",
    type=str,
    default="/cs/labs/yedid/yedid/wordcluster/buckeye_processed/val",
    help="dir of .wav, .wrd files for val data",
)
parser.add_argument(
    "--boundary_root_path",
    type=str,
    default="/data/skando/speechLM/experiment/boundaries/librispeech/train-clean-100/word",
    help="path to output word boundary",
)


parser.add_argument(
    "--extension",
    type=str,
    default="wav",
    help="extension of audio file",
)
parser.add_argument(
    "--train_n", type=int, default=200, help="number of files from training data to use"
)
parser.add_argument(
    "--eval_n",
    type=int,
    default=200,
    help="number of files from evaluation data to use",
)
parser.add_argument("--layer", type=int, default=-6, help="layer index (output)")
parser.add_argument("--offset", type=int, default=0, help="offset to window center")
parser.add_argument(
    "--arc",
    type=str,
    default="BASE",
    help="model architecture options: BASE, LARGE, LARGE_LV60K, XLSR53, HUBERT_BASE, HUBERT_LARGE, HUBERT_XLARGE",
)

parser.add_argument(
    "--min_separation", type=int, default=4, help="min separation between words"
)
parser.add_argument(
    "--target_perc", type=int, default=40, help="target quantization percentile"
)

parser.add_argument(
    "--frames_per_word", type=int, default=10, help="5 words in a second"
)
parser.add_argument("--loss", type=str, default="ridge", help="ridge || logres")
parser.add_argument(
    "--C", type=float, default=1.0, help="logistic regression parameter"
)
parser.add_argument("--reg", type=float, default=1e4, help="ridge regularization")

args = parser.parse_args()
print(args)


def get_grad_mag(e: npt.NDArray[Any]) -> npt.NDArray[Any]:
    e = np.pad(e, 1, mode="reflect")
    e = e[2:] - e[:-2]
    mag = e**2
    return mag.mean(1)


def get_seg(d, num_words, min_separation):
    idx = np.argsort(d)
    selected = []
    for i in idx[::-1]:
        if len(selected) >= num_words:
            break
        if (
            len(selected) == 0
            or (np.abs(np.array(selected) - i)).min() > min_separation
        ):
            selected.append(i)
    return np.sort(np.array(selected))


frames_per_embedding = 160  # (not 320), instead of multiplying by 2 later

# init seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Model init
model, dim = data_loader.get_model(args.arc)
model = model.cuda()


train_paths, train_wavs = data_loader.get_data(
    args.train_path, args.train_n, args.extension
)
val_audio_paths, val_wavs = data_loader.get_data(
    args.val_path, args.eval_n, args.extension
)

print("train data loading...")
train_e = data_loader.get_emb(train_wavs, model, args.layer)
print("validation data loading...")
val_e = data_loader.get_emb(val_wavs, model, args.layer)

print("frame duration (s): %f" % (frames_per_embedding / 16000))


ds: List[npt.NDArray[Any]] = []
for idx in range(len(train_e)):
    d = get_grad_mag(train_e[idx])
    ds.append(d)

ds = np.concatenate(ds)
th = np.percentile(ds, args.target_perc)
targets = ds > th
if args.loss == "ridge":
    clf = Ridge(alpha=args.reg)
else:
    clf = LogisticRegression(C=args.C, max_iter=1000)
train_e_np = np.concatenate(train_e)
mu = train_e_np.mean(0)[None, :]
std = train_e_np.std(0)[None, :]
clf.fit((train_e_np - mu) / std, targets)

seg_bounds: List[npt.NDArray[Any]] = []
for idx in range(len(val_e)):
    if args.loss == "logres":
        d = clf.predict_proba((val_e[idx] - mu) / std)[:, 1]
    else:
        d = clf.predict((val_e[idx] - mu) / std)
    num_words = int(len(val_e[idx]) / args.frames_per_word)
    p = get_seg(d, num_words, args.min_separation)

    p = p * 2 + args.offset
    p = np.minimum(p, 2 * (len(d) - 1))
    p = p.astype("int")

    seg_bound = np.zeros(len(d) * 2)
    seg_bound[p] = 1
    seg_bound[-1] = 1
    seg_bounds.append(seg_bound)

# Write to boundary file
for val_audio_path, seg_bound in zip(val_audio_paths, seg_bounds):
    boundary_path = data_loader.generate_aligned_path(
        args.boundary_root_path, args.val_path, val_audio_path
    )
    os.makedirs(boundary_path, exist_ok=True)
    with open(boundary_path.with_suffix(".txt"), "w") as f:
        for idx, b in enumerate(seg_bound):
            # idx * 10 gives millisecond time
            if b == 1:
                ms = idx * 10
                f.write(str(ms) + "\n")
