import argparse
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import numpy.typing as npt
import torch
import torchaudio
from boltons import fileutils
from sklearn.linear_model import LogisticRegression, Ridge
from tqdm import tqdm

import data_loader

SECOND_THRESHOLD = 300

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
    default="/data/skando/speechLM/experiment/boundaries/LibriSpeech/train-clean-100",
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

print("train data loading...")
train_e = data_loader.get_emb(train_wavs, model, args.layer)

print("frame duration (s): %f" % (frames_per_embedding / 16000))

print("getting grad magnitude...")
ds: List[npt.NDArray[Any]] = []
for idx in range(len(train_e)):
    d = get_grad_mag(train_e[idx])
    ds.append(d)

print("training classifier...")
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

print("segmenting validation data...")
val_audio_paths = list(fileutils.iter_find_files(args.val_path, f"*.{args.extension}"))
for val_audio_path in tqdm(val_audio_paths[: args.eval_n]):
    boundary_path = data_loader.generate_aligned_path(
        f"{args.boundary_root_path}/word", args.val_path, Path(val_audio_path)
    ).with_suffix(".txt")
    # skip if boundary file already exists
    if boundary_path.exists():
        continue
    os.makedirs(boundary_path.parent, exist_ok=True)

    # laod audio and get embedding
    val_wav, sr = torchaudio.load(val_audio_path)
    assert isinstance(val_wav, torch.Tensor)
    if len(val_wav[0]) > sr * SECOND_THRESHOLD:
        continue
    emb = data_loader.embed(val_wav, model, args.layer)

    # predict grad magnitude and perform segmentation
    if args.loss == "logres":
        d = clf.predict_proba((emb - mu) / std)[:, 1]
    else:
        d = clf.predict((emb - mu) / std)
    num_words = int(len(emb) / args.frames_per_word)
    p = get_seg(d, num_words, args.min_separation)

    # post-process segments
    p = p * 2 + args.offset
    p = np.minimum(p, 2 * (len(d) - 1))
    p = p.astype("int")

    seg_bound = np.zeros(len(d) * 2)
    seg_bound[p] = 1
    seg_bound[-1] = 1

    # write boundary file
    with open(boundary_path, "w") as f:
        for idx, b in enumerate(seg_bound):
            # idx * 10 gives millisecond time
            if b == 1:
                ms = idx * 10
                f.write(str(ms) + "\n")
