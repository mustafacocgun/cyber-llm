"""Multiple-choice -> open reasoning+answer conversion.

Training on raw "A/B/C/D" teaches quiz-picking, not explanation. So we drop the
letter framing: the user gets the question, the assistant gives the explanation
then states the correct option's TEXT.
"""
from prep.schema import message, record


def mcq_to_record(question, choices, answer_idx, explanation, meta):
    if not (0 <= answer_idx < len(choices)):
        raise ValueError(f"answer_idx {answer_idx} out of range for {len(choices)} choices")
    answer = str(choices[answer_idx]).strip()
    reasoning = str(explanation).strip()
    assistant = f"{reasoning}\n\nAnswer: {answer}" if reasoning else f"Answer: {answer}"
    return record([message("user", str(question).strip()),
                   message("assistant", assistant)], **meta)


def _resolve_idx(raw, fmt):
    """Map a source's answer to a 0-based index. Format is explicit, never guessed."""
    if fmt == "letter":
        return ord(str(raw).strip().upper()) - ord("A")
    if fmt == "index1":
        return int(raw) - 1
    if fmt == "index0":
        return int(raw)
    raise ValueError(f"unknown answer_format {fmt!r}")


def adapt_mcq(row, fm, meta):
    idx = _resolve_idx(row[fm["answer_idx"]], fm.get("answer_format", "index0"))
    return mcq_to_record(
        row[fm["question"]],
        row[fm["choices"]],
        idx,
        row.get(fm.get("explanation", ""), ""),
        meta,
    )
