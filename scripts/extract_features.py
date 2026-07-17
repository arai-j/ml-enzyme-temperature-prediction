"""Extract all applicable protlearn sequence descriptors from
data/brenda_topt_with_sequences.csv.

Every protlearn feature function requires sequences made up only of the 20
natural amino acids (rejects B/U/X/Z etc. with a ValueError). A handful of
functions (paac, apaac, socn, qso) additionally require the sequence to be
longer than their lag/lambda parameter (default 30). Rather than dropping
rows that fail either constraint, each feature block is computed only over
the eligible subset and the ineligible rows get NaN for that block's
columns -- so no row is silently lost from the table.

Three protlearn functions are intentionally NOT included:
- binary: position-specific one-hot encoding, requires padding every
  sequence to the dataset's longest sequence (7096 aa here), which would
  produce ~140,000 columns. Only meaningful for fixed-length windows
  (e.g. around a known motif), not whole variable-length proteins.
- posrich: requires an explicit (position, amino acid) pair -- a targeted
  single-site check, not an automatic whole-sequence descriptor.
- motif: requires an explicit regex-like pattern to search for -- likewise
  targeted, not automatic.

Note: protlearn (last released 2020) imports pkg_resources, removed from
recent setuptools. This environment has setuptools pinned to <81.
"""
import sys
import time

import numpy as np
import pandas as pd
from protlearn import features as f

NATURAL_AA = set("ACDEFGHIKLMNPQRSTVWY")

AAINDEX1_DEFAULT_PROPERTIES = [
    "CIDH920105", "BHAR880101", "CHAM820101", "CHAM820102",
    "CHOC760101", "BIGC670101", "CHAM810101", "DAYM780201",
]


def _aac(seqs):
    arr, aas = f.aac(seqs)
    return arr, [f"aac_{a}" for a in aas]


def _aaindex1(seqs):
    arr, names = f.aaindex1(seqs)
    return arr, [f"aaindex1_{n}" for n in names]


def _atc(seqs):
    atoms, bonds = f.atc(seqs)
    arr = np.hstack([atoms, bonds])
    cols = ["atc_C", "atc_H", "atc_N", "atc_O", "atc_S",
            "bond_total", "bond_single", "bond_double"]
    return arr, cols


def _cksaap(seqs):
    arr, names = f.cksaap(seqs, k=1)
    return arr, [f"cksaap_k1_{n}" for n in names]


def _ctd(seqs):
    arr, names = f.ctd(seqs)
    return arr, [f"ctd_{n}" for n in names]


def _entropy(seqs):
    arr = f.entropy(seqs)
    return arr, ["entropy"]


def _geary(seqs):
    arr = f.geary(seqs)
    return arr, [f"geary_{p}" for p in AAINDEX1_DEFAULT_PROPERTIES]


def _moran(seqs):
    arr = f.moran(seqs)
    return arr, [f"moran_{p}" for p in AAINDEX1_DEFAULT_PROPERTIES]


def _moreau_broto(seqs):
    arr = f.moreau_broto(seqs)
    return arr, [f"moreaubroto_{p}" for p in AAINDEX1_DEFAULT_PROPERTIES]


def _ngram2(seqs):
    arr, names = f.ngram(seqs, n=2)
    return arr, [f"ngram2_{n}" for n in names]


def _ngram3(seqs):
    arr, names = f.ngram(seqs, n=3)
    return arr, [f"ngram3_{n}" for n in names]


def _paac(seqs):
    arr, names = f.paac(seqs)
    return arr, [f"paac_{n}" for n in names]


def _apaac(seqs):
    arr, names = f.apaac(seqs)
    return arr, [f"apaac_{n}" for n in names]


def _socn(seqs):
    sw, g = f.socn(seqs)
    d = sw.shape[1]
    arr = np.hstack([sw, g])
    cols = [f"socn_sw_{i + 1}" for i in range(d)] + [f"socn_g_{i + 1}" for i in range(d)]
    return arr, cols


def _qso(seqs):
    sw, g, desc = f.qso(seqs)
    arr = np.hstack([sw, g])
    cols = [f"qso_sw_{d}" for d in desc] + [f"qso_g_{d}" for d in desc]
    return arr, cols


def _ctdc(seqs):
    arr, names = f.ctdc(seqs)
    return arr, [f"ctdc_{n}" for n in names]


def _ctdt(seqs):
    arr, names = f.ctdt(seqs)
    return arr, [f"ctdt_{n}" for n in names]


def _ctdd(seqs):
    arr, names = f.ctdd(seqs)
    return arr, [f"ctdd_{n}" for n in names]


# (name, min_length, extractor). min_length=30 for the lag/lambda-based
# descriptors whose default parameter is 30 (sequence must be strictly
# longer than that). ngram3 needs length >= 3 to form any triplet.
FEATURE_SPECS = [
    ("aac", 0, _aac),
    ("aaindex1", 0, _aaindex1),
    ("atc", 0, _atc),
    ("cksaap", 0, _cksaap),
    ("ctd", 0, _ctd),
    ("entropy", 0, _entropy),
    ("geary", 1, _geary),
    ("moran", 1, _moran),
    ("moreau_broto", 1, _moreau_broto),
    ("ngram2", 0, _ngram2),
    ("ngram3", 2, _ngram3),
    ("paac", 30, _paac),
    ("apaac", 30, _apaac),
    ("socn", 30, _socn),
    ("qso", 30, _qso),
    ("ctdc", 0, _ctdc),
    ("ctdt", 0, _ctdt),
    ("ctdd", 0, _ctdd),
]


def run_feature(df: pd.DataFrame, name: str, min_length: int, extractor, is_natural: pd.Series) -> pd.DataFrame:
    mask = is_natural & (df["sequence_length"] > min_length)
    valid_seqs = df.loc[mask, "sequence"].tolist()

    t0 = time.time()
    arr, cols = extractor(valid_seqs)
    arr = np.asarray(arr, dtype=float)
    elapsed = time.time() - t0

    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    out.loc[mask, cols] = arr
    print(f"{name}: {len(cols)} cols, {mask.sum()}/{len(df)} rows computed ({elapsed:.1f}s)", flush=True)
    return out


def add_all_features(df: pd.DataFrame, seq_col: str = "sequence") -> pd.DataFrame:
    is_natural = df[seq_col].apply(lambda s: set(s) <= NATURAL_AA)
    print(f"{(~is_natural).sum()} / {len(df)} sequences contain non-natural residues "
          f"(e.g. B/U/X/Z) -- excluded from every feature block below (NaN filled)")

    blocks = [df]
    for name, min_length, extractor in FEATURE_SPECS:
        blocks.append(run_feature(df, name, min_length, extractor, is_natural))

    return pd.concat(blocks, axis=1)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/brenda_topt_with_sequences.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/brenda_features.csv"

    df = pd.read_csv(src)
    df = add_all_features(df)

    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows, {len(df.columns)} columns)")