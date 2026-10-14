"""De-duplication by strong id (CVE / commit hash) or exact text hash.

Strong ids catch the same vulnerability re-packaged across sources (e.g.
``cyberdata-2`` embeds PrimeVul/CVEfixes rows); the text hash catches exact
duplicates that carry no id. Paraphrase-level overlap is NOT caught — a known
limitation, documented in DATA_PLAN.md.
"""
import hashlib
import re

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)


def strong_keys(rec):
    """Namespaced id keys from meta.ids, e.g. {'cve:CVE-2021-1', 'commit:abc'}."""
    return {f"{k}:{v}" for k, v in rec["meta"].get("ids", {}).items() if v}


def text_hash(rec):
    blob = "\n".join(f"{m['role']}:{m['content']}" for m in rec["messages"])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def dedup(records):
    """Keep first occurrence; drop any later record sharing a strong key or exact text."""
    seen_keys, seen_text, out = set(), set(), []
    for rec in records:
        keys = strong_keys(rec)
        if keys & seen_keys:
            continue
        th = text_hash(rec)
        if th in seen_text:
            continue
        seen_keys |= keys
        seen_text.add(th)
        out.append(rec)
    return out


def extract_ids(row, field_map, text=""):
    """Dedup/decontam ids for a record: declared id columns plus an optional CVE scan.

    `id_fields` maps a key (e.g. 'commit') to a source column; `extract_cve: true`
    pulls the first CVE id out of the record's text. Returns {} when nothing applies.
    """
    ids = {}
    for key, col in field_map.get("id_fields", {}).items():
        val = row.get(col)
        if val not in (None, ""):
            ids[key] = str(val)
    if field_map.get("extract_cve"):
        m = _CVE_RE.search(text or "")
        if m:
            ids["cve"] = m.group(0).upper()
    return ids
