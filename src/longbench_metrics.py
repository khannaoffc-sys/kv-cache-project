# longbench_metrics.py — Official LongBench scoring functions.
#
# Source:  https://github.com/THUDM/LongBench
# Files:   LongBench/metrics.py  +  LongBench/eval.py
# Commit:  2e00731f8d0bff23dc4325161044d0ed8af94c1e  (2025-01-15)
# URL:     https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/metrics.py
#          https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/eval.py
#
# The scorer functions (qa_f1_score, rouge_score, etc.) are copied verbatim from
# the THUDM/LongBench repository. The dispatch table (dataset2metric) and the
# per-example scoring wrapper (score()) are adapted from eval.py.
#
# Dependencies:
#   - rouge  (pip install rouge)
#   - fuzzywuzzy  (pip install fuzzywuzzy python-Levenshtein)
#   - jieba  (pip install jieba)   — only needed for Chinese tasks
#
# IMPORTANT: narrativeqa uses qa_f1_score (character F1 after normalisation),
# NOT token F1. The fallback _token_f1 in run_benchmark.py / accuracy_sweep.py
# is a worse approximation — replace it with score() from this module.
#
# Usage:
#   from longbench_metrics import score
#   f1 = score(prediction, reference, dataset="narrativeqa")

import re
import string
from collections import Counter
from typing import List, Optional


# ─── Normalisation helpers (verbatim from LongBench/metrics.py) ──────────────

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    """Lower text and remove punctuation, extra whitespace (Chinese)."""

    def white_space_fix(text: str) -> str:
        return "".join(text.split())

    def remove_punc(text: str) -> str:
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、"
            "〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛\"\"„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


# ─── Metric functions (verbatim from LongBench/metrics.py) ───────────────────

def _f1_score_chars(prediction: list, ground_truth: list) -> float:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """QA token-F1 (English) — character-list F1 after normalisation."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return _f1_score_chars(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """QA token-F1 (Chinese) via jieba segmentation."""
    try:
        import jieba  # type: ignore[import]
    except ImportError:
        return qa_f1_score(prediction, ground_truth)
    prediction_tokens = [normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
    ground_truth_tokens = [normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
    prediction_tokens = [t for t in prediction_tokens if t]
    ground_truth_tokens = [t for t in ground_truth_tokens if t]
    return _f1_score_chars(prediction_tokens, ground_truth_tokens)


def rouge_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """ROUGE-L F1."""
    try:
        from rouge import Rouge  # type: ignore[import]
        r = Rouge()
        scores = r.get_scores([prediction], [ground_truth], avg=True)
        return scores["rouge-l"]["f"]
    except Exception:
        return 0.0


def rouge_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """ROUGE-L for Chinese (jieba tokenisation)."""
    try:
        import jieba  # type: ignore[import]
        prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
        ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    except ImportError:
        pass
    return rouge_score(prediction, ground_truth)


def count_score(prediction: str, ground_truth: str, **kwargs) -> float:
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if not numbers else float(right_num / len(numbers))


def retrieval_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"Paragraph (\d+)", ground_truth)
    ground_truth_id = matches[0] if matches else ""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if not numbers else float(right_num / len(numbers))


def retrieval_zh_score(prediction: str, ground_truth: str, **kwargs) -> float:
    matches = re.findall(r"段落(\d+)", ground_truth)
    ground_truth_id = matches[0] if matches else ""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if not numbers else float(right_num / len(numbers))


def code_sim_score(prediction: str, ground_truth: str, **kwargs) -> float:
    try:
        from fuzzywuzzy import fuzz  # type: ignore[import]
    except ImportError:
        return 0.0
    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if "`" not in line and "#" not in line and "//" not in line:
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100.0


def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    all_classes = kwargs.get("all_classes", []) or []
    em_match_list = [c for c in all_classes if c in prediction]
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


# ─── Dispatch table (verbatim from LongBench/eval.py) ────────────────────────

DATASET2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# Tasks where the first non-empty, non-commented line is taken as the answer
_FIRST_LINE_TASKS = {"trec", "triviaqa", "samsum", "lsht"}


# ─── Public API ───────────────────────────────────────────────────────────────

def score(
    prediction: str,
    ground_truth: str,
    dataset: str,
    all_classes: Optional[List[str]] = None,
) -> float:
    """Score one prediction against one reference for a given LongBench task.

    This is a per-example wrapper that mirrors the inner loop of eval.py scorer().

    Parameters
    ----------
    prediction   : model output string
    ground_truth : single reference answer string
    dataset      : LongBench task name, e.g. "narrativeqa"
    all_classes  : list of candidate classes (used by classification tasks)

    Returns
    -------
    float in [0, 1]  (the official scores are ×100 in the paper; we return the
    raw fraction so the caller can choose to multiply or not).
    """
    if dataset not in DATASET2METRIC:
        raise ValueError(
            f"Unknown LongBench dataset: {dataset!r}. "
            f"Valid tasks: {sorted(DATASET2METRIC)}"
        )

    # Apply first-line extraction for classification-style tasks
    if dataset in _FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]

    metric_fn = DATASET2METRIC[dataset]
    return metric_fn(prediction, ground_truth, all_classes=all_classes or [])


def score_multi(
    prediction: str,
    ground_truths: List[str],
    dataset: str,
    all_classes: Optional[List[str]] = None,
) -> float:
    """Score one prediction against multiple references; return the maximum."""
    return max(
        score(prediction, gt, dataset=dataset, all_classes=all_classes)
        for gt in ground_truths
    )


# ─── LongBench data loading — Phase 4 approach ───────────────────────────────

def load_longbench_jsonl(task: str, n: int = -1):
    """Load LongBench examples from data.zip (the only file in the HF repo).

    The THUDM/LongBench HF dataset repo stores all data in a single data.zip
    archive. The loading script (LongBench.py) extracts task-specific JSONL
    files from it. We replicate that extraction here without trust_remote_code.

    Returns a list of dicts with keys: input, context, answers, length,
    dataset, language, all_classes, _id.
    """
    import json
    import zipfile

    from huggingface_hub import hf_hub_download  # type: ignore[import]

    local_zip = hf_hub_download(
        repo_id="THUDM/LongBench",
        filename="data.zip",
        repo_type="dataset",
    )
    examples = []
    target_name = f"data/{task}.jsonl"
    with zipfile.ZipFile(local_zip, "r") as zf:
        if target_name not in zf.namelist():
            available = [n for n in zf.namelist() if n.endswith(".jsonl")]
            raise ValueError(
                f"Task {task!r} not found in data.zip. "
                f"Available tasks: {[p.split('/')[-1].replace('.jsonl','') for p in available]}"
            )
        with zf.open(target_name) as fh:
            for line in fh:
                ex = json.loads(line)
                examples.append(ex)
                if n > 0 and len(examples) >= n:
                    break
    return examples
