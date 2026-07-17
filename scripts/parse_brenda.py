"""Parse the BRENDA flat-file download into a long-format table of
EC number / organism / UniProt accession / temperature optimum.

BRENDA flat-file structure (verified against data/brenda_2026_1.txt):
- Entries for one EC number run from a line "ID\t<ec>" to a line "///".
- Within an entry, data is grouped under all-caps section headers on
  their own line (e.g. "PROTEIN", "TEMPERATURE_OPTIMUM").
- Each datum is a line "TAG\t...."; a line that does NOT start with a
  recognized
  tag is a continuation of the previous datum (BRENDA wraps
  long lines with a leading tab).
- PR lines assign a protein number to an organism, e.g.:
    PR	#1# Gallus gallus <44>
    PR	#98# Saccharolobus solfataricus {Q9UXF1; source: UniProt} <165>
- TO lines give a temperature optimum tied to one or more protein
  numbers, e.g.:
    TO	#12,47,101# 35 (#12# immobilized enzyme <199>; ...) <125,171,199>
  Values of "-999" are BRENDA's sentinel for "undetermined" and are
  dropped.
"""
import re
from dataclasses import dataclass, field

EC_ID_RE = re.compile(r"^ID\t(\d+\.\d+\.\d+\.\d+)")
TAG_RE = re.compile(r"^([A-Z][A-Z0-9]{1,4})\t(.*)$")
SECTION_HEADER_RE = re.compile(r"^[A-Z][A-Z_]*$")
PR_RE = re.compile(r"^#(\d+)#\s*(.+)$")
ACCESSION_RE = re.compile(
    r"\{([A-Za-z0-9]+);\s*source:\s*(UniProt|SwissProt|TrEMBL)\}", re.IGNORECASE
)
TO_RE = re.compile(r"^#([\d,]+)#\s*([\-\d.]+(?:-[\-\d.]+)?)")


@dataclass
class EcEntry:
    ec_number: str = ""
    protein_map: dict = field(default_factory=dict)  # num -> (organism, accession)
    to_lines: list = field(default_factory=list)  # raw "TO\t..." payload strings


def _parse_pr_line(payload: str):
    """Return (protein_num, organism, accession_or_None) or None."""
    m = PR_RE.match(payload)
    if not m:
        return None
    num, rest = m.group(1), m.group(2)
    acc_match = ACCESSION_RE.search(rest)
    accession = acc_match.group(1) if acc_match else None
    # Organism name is everything before the first "{" or "<"
    organism = re.split(r"[{<]", rest)[0].strip()
    return num, organism, accession


def _parse_to_line(payload: str):
    """Return (list_of_protein_nums, temp_value_or_None, raw_temp_str) or None."""
    m = TO_RE.match(payload)
    if not m:
        return None
    nums = m.group(1).split(",")
    raw = m.group(2)
    if raw.strip() == "-999":
        return nums, None, raw
    if "-" in raw[1:]:  # range like "45-50", but not a leading negative sign
        lo, hi = raw[1:].split("-", 1) if raw[0] == "-" else raw.split("-", 1)
        lo = float(("-" + lo) if raw[0] == "-" else lo)
        hi = float(hi)
        value = (lo + hi) / 2
    else:
        value = float(raw)
    return nums, value, raw


def _flush_entry(entry: EcEntry, rows: list):
    for payload in entry.to_lines:
        parsed = _parse_to_line(payload)
        if parsed is None:
            continue
        nums, value, raw = parsed
        if value is None:
            continue
        for num in nums:
            hit = entry.protein_map.get(num)
            if hit is None:
                continue
            organism, accession = hit
            if accession is None:
                continue
            rows.append(
                {
                    "ec_number": entry.ec_number,
                    "protein_number": num,
                    "organism": organism,
                    "uniprot_accession": accession,
                    "temperature_optimum_c": value,
                    "temperature_optimum_raw": raw,
                }
            )


def parse_brenda_file(path: str) -> "pandas.DataFrame":
    import pandas as pd

    rows = []
    entry = None
    section = None
    pending_tag = None
    pending_payload = None

    def flush_pending():
        nonlocal pending_tag, pending_payload
        if entry is None or pending_tag is None:
            pending_tag = None
            pending_payload = None
            return
        if section == "PROTEIN" and pending_tag == "PR":
            parsed = _parse_pr_line(pending_payload)
            if parsed:
                num, organism, accession = parsed
                entry.protein_map[num] = (organism, accession)
        elif section == "TEMPERATURE_OPTIMUM" and pending_tag == "TO":
            entry.to_lines.append(pending_payload)
        pending_tag = None
        pending_payload = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            ec_match = EC_ID_RE.match(line)
            if ec_match:
                flush_pending()
                entry = EcEntry(ec_number=ec_match.group(1))
                section = None
                continue

            if line.strip() == "///":
                flush_pending()
                if entry is not None:
                    _flush_entry(entry, rows)
                entry = None
                section = None
                continue

            if entry is None:
                continue  # inside preamble / non-EC entries (e.g. "spontaneous")

            if line.strip() == "":
                flush_pending()
                continue

            if SECTION_HEADER_RE.match(line):
                flush_pending()
                section = line.strip()
                continue

            tag_match = TAG_RE.match(line)
            if tag_match:
                flush_pending()
                pending_tag, pending_payload = tag_match.group(1), tag_match.group(2)
                continue

            # continuation line
            if pending_payload is not None:
                pending_payload += " " + line.strip()

    flush_pending()
    if entry is not None:
        _flush_entry(entry, rows)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/brenda_2026_1.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/brenda_topt_extracted.csv"
    df = parse_brenda_file(src)
    print(f"Parsed {len(df)} (EC, organism, accession, Topt) rows")
    print(f"Unique EC numbers: {df['ec_number'].nunique()}")
    print(f"Unique UniProt accessions: {df['uniprot_accession'].nunique()}")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")