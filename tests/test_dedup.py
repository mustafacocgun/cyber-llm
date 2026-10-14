from prep.dedup import dedup, strong_keys, text_hash
from prep.schema import message, record


def _rec(text, ids=None):
    return record([message("user", text), message("assistant", "a")],
                  source="s", bucket="b", license="l", ids=ids or {})


def test_strong_keys_from_ids():
    rec = _rec("x", ids={"cve": "CVE-2021-1", "commit": "abc123"})
    assert strong_keys(rec) == {"cve:CVE-2021-1", "commit:abc123"}


def test_text_hash_stable_and_distinct():
    assert text_hash(_rec("same")) == text_hash(_rec("same"))
    assert text_hash(_rec("a")) != text_hash(_rec("b"))


def test_dedup_removes_shared_cve_across_sources():
    out = dedup([_rec("a", ids={"cve": "CVE-2021-1"}), _rec("b", ids={"cve": "CVE-2021-1"})])
    assert len(out) == 1


def test_dedup_removes_identical_text():
    out = dedup([_rec("same"), _rec("same")])
    assert len(out) == 1


def test_dedup_keeps_distinct():
    out = dedup([_rec("one"), _rec("two")])
    assert len(out) == 2
