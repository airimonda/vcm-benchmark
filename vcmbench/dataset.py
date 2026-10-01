"""Load the holdout split (196 clips: 93 variations x 2 + 10 out-of-scope).

Source: the class dataset on Hugging Face (airimonda/ai231-me2-voice-commands,
split `holdout`), cached locally after the first download. A local copy of
`dataset/holdout` (manifest.csv + audio/) also works.
"""
from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

HF_REPO = "airimonda/ai231-me2-voice-commands"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/data/holdout-00000-of-00001.parquet"
SR = 16000


@dataclass
class Clip:
    idx: int
    file: str
    transcript: str
    intent: str           # one of the 19 intents or OUT_OF_SCOPE
    variation: str        # one of the 93 variation names, "" for out of scope
    slot_value: str
    speaker_id: str
    is_synthetic: bool
    accent_group: str
    audio: np.ndarray = field(repr=False)   # float32 mono, 16 kHz


def _to_float_mono(data: np.ndarray, sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != SR:
        from .audio import resample
        data = resample(data, sr, SR)
    return data


def download_holdout(cache_dir: Path, url: str = HF_URL) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "holdout.parquet"
    if not path.exists():
        tmp = path.with_suffix(".part")
        print(f"  Downloading holdout set from Hugging Face ({HF_REPO}) ...")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return path


def _row_to_clip(i: int, r: dict, audio: np.ndarray) -> Clip:
    oos = int(r.get("out_of_scope") or 0) == 1 or r.get("command") == "OUT_OF_SCOPE"
    return Clip(
        idx=i, file=str(r.get("file", "")), transcript=str(r.get("transcript", "")),
        intent="OUT_OF_SCOPE" if oos else str(r["command"]),
        variation="" if oos else str(r.get("variation") or ""),
        slot_value="" if oos else str(r.get("slot_value") or ""),
        speaker_id=str(r.get("speaker_id", "")), is_synthetic=int(r.get("is_synthetic") or 0) == 1,
        accent_group=str(r.get("accent_group", "")), audio=audio,
    )


def load_holdout(source: str | None, cache_dir: Path) -> list[Clip]:
    """`source`: None/"hf" = Hugging Face, a .parquet file, or a folder with manifest.csv."""
    import pandas as pd

    if source and Path(source).expanduser().is_dir():
        root = Path(source).expanduser()
        df = pd.read_csv(root / "manifest.csv", keep_default_na=False)
        clips = []
        for i, r in enumerate(df.to_dict("records")):
            data, sr = sf.read(root / r["file"], always_2d=False)
            clips.append(_row_to_clip(i, r, _to_float_mono(data, sr)))
        return clips

    path = Path(source).expanduser() if source and source != "hf" else download_holdout(cache_dir)
    df = pd.read_parquet(path)
    clips = []
    for i, r in enumerate(df.to_dict("records")):
        data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), always_2d=False)
        r = {k: ("" if v is None else v) for k, v in r.items() if k != "audio"}
        clips.append(_row_to_clip(i, r, _to_float_mono(data, sr)))
    return clips
