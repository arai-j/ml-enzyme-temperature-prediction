"""Fetch full amino-acid sequences for a list of UniProt accessions using
UniProt's REST API (https://rest.uniprot.org), batched to stay well under
documented request-size limits.
"""
import io
import sys
import time

import pandas as pd
import requests

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/accessions"
BATCH_SIZE = 90
MAX_RETRIES = 3


def _parse_fasta(text: str) -> dict:
    """Return {accession: sequence} from a multi-record FASTA string."""
    seqs = {}
    accession = None
    chunks = []
    for line in text.splitlines():
        if line.startswith(">"):
            if accession is not None:
                seqs[accession] = "".join(chunks)
            # header looks like: >sp|P43408|KADA_METIG ...
            accession = line.split("|")[1] if "|" in line else line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line.strip())
    if accession is not None:
        seqs[accession] = "".join(chunks)
    return seqs


def _query_batch(batch: list) -> dict:
    """Query one batch, retrying transient errors. Raises on a persistent
    non-2xx (e.g. a malformed accession in the batch causing a 400)."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                UNIPROT_URL,
                params={"accessions": ",".join(batch), "format": "fasta"},
                timeout=30,
            )
            resp.raise_for_status()
            return _parse_fasta(resp.text)
        except requests.RequestException as e:
            last_exc = e
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 400:
                break  # malformed accession in batch, not transient -- don't retry
            time.sleep(2 * attempt)
    raise last_exc


def _fetch_with_bisection(batch: list, invalid: list) -> dict:
    """Fetch a batch; on failure (e.g. one malformed accession), bisect
    until the bad accession(s) are isolated into `invalid` and the rest
    still resolve."""
    if not batch:
        return {}
    try:
        return _query_batch(batch)
    except requests.RequestException:
        if len(batch) == 1:
            invalid.append(batch[0])
            return {}
        mid = len(batch) // 2
        seqs = _fetch_with_bisection(batch[:mid], invalid)
        seqs.update(_fetch_with_bisection(batch[mid:], invalid))
        return seqs


def fetch_sequences(accessions: list, batch_size: int = BATCH_SIZE) -> dict:
    """Return {accession: sequence} for as many of `accessions` as UniProt resolves."""
    all_seqs = {}
    invalid = []
    accessions = list(dict.fromkeys(accessions))  # dedupe, keep order
    n_batches = (len(accessions) + batch_size - 1) // batch_size

    for i in range(0, len(accessions), batch_size):
        batch = accessions[i : i + batch_size]
        batch_num = i // batch_size + 1
        all_seqs.update(_fetch_with_bisection(batch, invalid))
        print(f"batch {batch_num}/{n_batches} done, {len(all_seqs)} sequences so far")

    if invalid:
        print(f"{len(invalid)} accessions rejected by UniProt as malformed: {invalid}")
    missing = [a for a in accessions if a not in all_seqs]
    if missing:
        print(f"{len(missing)} accessions not resolved by UniProt (obsolete/merged/malformed/etc.)")
    return all_seqs


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/brenda_topt_extracted.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/brenda_topt_with_sequences.csv"

    df = pd.read_csv(src)
    accessions = df["uniprot_accession"].dropna().unique().tolist()
    print(f"Fetching sequences for {len(accessions)} unique UniProt accessions...")

    seq_map = fetch_sequences(accessions)

    df["sequence"] = df["uniprot_accession"].map(seq_map)
    df["sequence_length"] = df["sequence"].str.len()
    n_before = len(df)
    df = df.dropna(subset=["sequence"])
    print(f"Dropped {n_before - len(df)} rows with no resolvable sequence")

    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")

