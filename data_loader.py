from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torchaudio
from boltons import fileutils
from torchaudio.models import Wav2Vec2Model
from tqdm import tqdm

SECOND_THRESHOLD = 300


def generate_aligned_path(root_path: str, audio_root_path: str, audio_path: Path):
    return Path(root_path) / audio_path.relative_to(audio_root_path)


def get_data(
    path: str, max_files: int, extension: str
) -> Tuple[List[Path], List[torch.Tensor]]:
    wavs = list(fileutils.iter_find_files(path, f"*.{extension}"))
    all_paths: List[Path] = []
    all_wavs: List[torch.Tensor] = []
    rp = np.random.permutation(len(wavs))
    wavs: List[Any] = [wavs[i] for i in rp]
    for wav in tqdm(wavs[:max_files]):
        waveform, sr = torchaudio.load(wav)
        assert isinstance(waveform, torch.Tensor)
        if len(waveform[0]) > sr * SECOND_THRESHOLD:
            continue

        all_paths.append(Path(wav))
        all_wavs.append(waveform)
    return all_paths, all_wavs


def get_emb(
    wavs: List[torch.Tensor], model: Wav2Vec2Model, layer: int = -1, feat_idx: int = -1
):
    es: List[npt.NDArray[Any]] = []
    for waveform in tqdm(wavs):
        e = embed(waveform, model, layer, feat_idx)
        es.append(e)
    return es


def embed(
    y: torch.Tensor, model: Wav2Vec2Model, extract_layer: int = -1, feat_idx: int = -1
) -> npt.NDArray[Any]:
    with torch.no_grad():
        model.eval()
        y = torch.Tensor(y).cuda()
        x, _ = model.extract_features(y)
        x = x[extract_layer]
        if not feat_idx == -1:
            x = x[:, :, feat_idx]
    return x.data.cpu().numpy()[0]


def get_model(arc: str) -> Tuple[Wav2Vec2Model, int]:
    if arc == "BASE":  # 12 output layers
        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        my_dim = 768
    elif arc == "LARGE":  # 24 output layers
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE
        my_dim = 1024
    elif arc == "LARGE_LV60K":  # 24 output layers
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
        my_dim = 1024
    elif arc == "XLSR53":  # 24 output layers
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
        my_dim = 1024
    elif arc == "HUBERT_BASE":
        bundle = torchaudio.pipelines.HUBERT_BASE
        my_dim = 768
    elif arc == "HUBERT_LARGE":
        bundle = torchaudio.pipelines.HUBERT_LARGE
        my_dim = 1024
    elif arc == "HUBERT_XLARGE":
        bundle = torchaudio.pipelines.HUBERT_XLARGE
        my_dim = 1280  # ?
    else:
        bundle = (
            torchaudio.pipelines.WAV2VEC2_BASE
        )  # WAV2VEC2_BASE WAV2VEC2_LARGE WAV2VEC2_LARGE_LV60K WAV2VEC2_XLSR53
        my_dim = 768
    return bundle.get_model(), my_dim


def get_bounds(boundaries: List[Tuple[int, int]]):
    l = [0]
    for i in range(len(boundaries) - 1):
        l.append((boundaries[i][1] + boundaries[i + 1][0]) // 2)
    l.append(boundaries[-1][1])
    return l
