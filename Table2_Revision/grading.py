"""CPU-only answer grading, preserving the native legacy and strict conventions."""
import hashlib
from importlib.metadata import version
import multiprocessing as mp
from pathlib import Path
import sys

from spec import ROOT

sys.path.insert(0, str(ROOT / "vendor"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def gold_answer(item):
    """MATH ships worked solutions; GSM ships an already extracted answer."""
    if "answer" in item:
        return str(item["answer"])
    from evaluation.parser import extract_answer
    # Preserve textual answers such as \\text{east}; unit removal can erase them.
    answer = extract_answer(item["solution"], data_name="minerva_math", use_last_number=False)
    if not answer:
        raise ValueError(f"Cannot extract gold answer for index {item['index']}")
    return answer


def evaluator_identity():
    files = {str(f.relative_to(ROOT)):sha(f) for f in list((ROOT/"vendor").rglob("*.py"))+[Path(__file__)]}
    packages = {name:version(name) for name in ("sympy", "mpmath", "antlr4-python3-runtime", "regex", "word2number", "numpy")}
    return dict(files=files, packages=packages, python=sys.version)


def compare_child(connection, text, response, gold):
    try:
        from evaluation.parser import extract_answer, strip_string
        from evaluation.grader import math_equal
        old = strip_string(extract_answer(text, data_name="omni-math"), skip_unit=False)
        full = strip_string(extract_answer(response, data_name="minerva_math", use_last_number=False), skip_unit=True)
        target = strip_string(str(gold).strip(), skip_unit=False)
        strict_target = strip_string(str(gold).strip(), skip_unit=True)
        connection.send(dict(status="ok", prediction=old, strict_prediction=full,
                             correct=bool(math_equal(old, target)),
                             strict_correct=bool(full and strict_target and math_equal(full, strict_target, include_percentage=False))))
    except Exception as exc:
        connection.send(dict(status="error", error=type(exc).__name__ + ": " + str(exc)))
    finally:
        connection.close()


def evaluate(row, gold, timeout=8):
    # Only used in the separate CPU command, never a CUDA-initialized process.
    ctx = mp.get_context("fork")
    rx, tx = ctx.Pipe(duplex=False)
    p = ctx.Process(target=compare_child, args=(tx, row["text"], row["response_text"], gold))
    p.start()
    tx.close()
    try:
        if rx.poll(timeout):
            try:
                result = rx.recv()
            except EOFError:
                result = dict(status="error", error="evaluator exited without result")
        else:
            result = dict(status="timeout")
    finally:
        rx.close()
        if p.is_alive():
            p.terminate()
        p.join()
    return result
