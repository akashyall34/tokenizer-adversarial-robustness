"""Scoring for the LLaVA-Med (MedITok) VQA red-team, one grader per question type.

Clinical yes/no (`parse_yesno`, `parse_benign_malignant`, `score_clinical_csv`):
take the first committed word of the diagnostic answer (pneumonia, referable DR,
or malignant) and report clean accuracy plus the diagnosis-flip rate from clean
to adv and from clean to random.

Modality and body-region verdict (`is_correct`): take the modality or anatomy the
answer declares in its first sentence and check it against the dataset. Whichever
of the expected or distractor vocabulary appears first wins. The lists are
enumerated from the distinct answer openings the model produces on this data.

`eps_pooled`, `full_accuracy`, and `direction_counts` aggregate a raw-answer CSV
into `tab:vqa-eps`, `tab:vqa-full`, and `tab:vqa-direction`, which
`check_paper_figures.py` checks. `tab:vqa-direction` reproduces exactly and
`tab:vqa-eps` to within one image. Two single-eps `tab:vqa-full` modality cells
(Kermany and ISIC at eps=2) land one to two images off, on the boundary where the
smallest budget flips an answer.
"""
import re

import pandas as pd

MODALITY_Q = "What modality is used to take this image?"
REGION_Q = "Which part of the body does this image belong to?"
CLINICAL_QUESTION = {
    "kermany": "Is there any sign of pneumonia in this image?",
    "aptos": "Is there evidence of diabetic retinopathy in this image?",
    "isic": "Does this skin lesion appear benign or malignant?",
}
POSITIVE_LABEL = {"kermany": "pneumonia", "aptos": "referable DR", "isic": "malignant"}


# --- clinical yes/no -------------------------------------------------------
def parse_yesno(text):
    """First word yes/no. True (yes) / False (no) / None if it does not commit."""
    t = str(text).strip().lower().lstrip(".,!\"' ")
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    return None


def parse_benign_malignant(text):
    """True = malignant, False = benign, None if both/neither. When both words
    appear the earlier one is taken as the model's answer."""
    t = str(text).lower()
    has_b, has_m = "benign" in t, "malignant" in t
    if has_m and not has_b:
        return True
    if has_b and not has_m:
        return False
    if has_b and has_m:
        return t.index("malignant") < t.index("benign")
    return None


PARSER = {"kermany": parse_yesno, "aptos": parse_yesno, "isic": parse_benign_malignant}


def score_clinical_csv(df: pd.DataFrame, attack_label: str) -> list[dict]:
    """One summary row per (dataset, eps): clean diagnostic accuracy, adv/random
    diagnosis-flip %, and the token-flip % already recorded on the same rows."""
    rows = []
    for dataset in sorted(df["dataset"].unique()):
        q, pos, parse = CLINICAL_QUESTION[dataset], POSITIVE_LABEL[dataset], PARSER[dataset]
        sub = df[(df["dataset"] == dataset) & (df["question"] == q)]
        if sub.empty:
            continue
        clean_by_image = sub.drop_duplicates("image")
        clean_preds = clean_by_image["clean"].apply(parse)
        truth = clean_by_image["true_label"].eq(pos)
        scored = clean_preds.notna()
        clean_acc = (clean_preds[scored] == truth[scored]).mean() if scored.any() else float("nan")
        for eps in sorted(sub["eps"].unique()):
            at = sub[sub["eps"] == eps]
            cp, ap, rp = at["clean"].apply(parse), at["adv"].apply(parse), at["random"].apply(parse)
            adv_ok, rnd_ok = cp.notna() & ap.notna(), cp.notna() & rp.notna()
            rows.append({
                "attack": attack_label, "dataset": dataset, "eps": eps, "n_images": len(at),
                "clean_diag_acc_pct": round(clean_acc * 100, 1) if scored.any() else float("nan"),
                "adv_diag_flip_pct": round((cp[adv_ok] != ap[adv_ok]).mean() * 100, 1) if adv_ok.any() else float("nan"),
                "rnd_diag_flip_pct": round((cp[rnd_ok] != rp[rnd_ok]).mean() * 100, 1) if rnd_ok.any() else float("nan"),
                "adv_token_flip_pct": round(at["adv_flip_pct"].mean(), 1),
                "rnd_token_flip_pct": round(at["rnd_flip_pct"].mean(), 1),
                "ambiguous_pct": round(100 * (1 - adv_ok.mean()), 1),
            })
    return rows


# --- modality / body-region verdict ----------------------------------------
# The declared modality/anatomy that counts as correct for each dataset, and the
# distractors the attack drives answers toward. Enumerated from the actual answer
# openings, not guessed.
_CORRECT = {
    ("kermany", MODALITY_Q): ("chest x-ray", "chest x ray", "radiograph", "radiographic", "plain film"),
    ("aptos", MODALITY_Q): ("fundus", "retinal imaging", "retinal photograph", "ophthalmic imaging",
                            "ophthalmoscop", "fluorescein angiograph"),
    ("isic", MODALITY_Q): ("dermoscop", "dermatoscop", "dermascop"),
    ("kermany", REGION_Q): ("chest", "thorax", "thoracic", "lung", "pulmonary"),
    ("aptos", REGION_Q): ("eye", "retina", "fundus", "optic", "macula", "cornea", "ocular"),
}
_WRONG = {
    MODALITY_Q: ("ct scan", "computed tomography", "mri", "magnetic resonance", "ultrasound",
                 "histolog", "histopath", "microscop", "h&e", "hematoxylin", "mammogram", "pet",
                 "endoscop", "arthroscop", "fluoroscop", "barium", "angiography", "immunohistochem",
                 "staining", "gross specimen", "3d", "trichrome", "digital subtraction"),
    REGION_Q: ("abdomen", "abdominal", "brain", "cranial", "head and neck", "pelvi", "liver",
               "esophagus", "knee", "extremity", "leg", "heart", "placenta", "prostate", "trachea",
               "spine", "skin of", " ear", "stomach", "bladder", "kidney"),
}
def _first_hit(text, terms):
    """Leftmost position at which any of `terms` appears in `text`, or -1."""
    hits = [text.find(t) for t in terms if t in text]
    return min(hits) if hits else -1


def _verdict(answer, dataset, question):
    """True / False / None for one modality- or body-region answer.

    The model states its verdict in the first sentence ("The image is a chest
    X-ray...", "This image belongs to the abdominal region..."); whichever of the
    correct vs. distractor vocabulary appears first there wins. Neither present
    -> None (excluded from the accuracy denominator)."""
    a = str(answer).lower()
    seg = re.split(r"(?<=[.?!])\s", a, maxsplit=1)[0][:160]
    good = _first_hit(seg, _CORRECT[(dataset, question)])
    bad = _first_hit(seg, _WRONG[question])
    if good >= 0 and (bad < 0 or good < bad):
        return True
    if bad >= 0 and (good < 0 or bad < good):
        return False
    return None


def is_correct(question, answer, true_label, dataset):
    """Verdict for one (question, answer) cell: True / False / None (ambiguous).
    None is treated as incorrect by the accuracy tables (n is fixed at 20)."""
    if question == MODALITY_Q:
        return _verdict(answer, dataset, MODALITY_Q)
    if question == REGION_Q:
        return _verdict(answer, dataset, REGION_Q)
    if question == CLINICAL_QUESTION.get(dataset):
        pred = PARSER[dataset](answer)
        return None if pred is None else pred == (true_label == POSITIVE_LABEL[dataset])
    return None  # the excluded questions (pneumonia-always-no, ISIC lesion-type)


# Questions that carry a scoreable verdict, per dataset. ISIC has no body-region
# question, so its clean-correct pool is modality only (the paper's n=99 for tab:vqa-eps).
SCORED_QUESTIONS = {
    "kermany": [MODALITY_Q, REGION_Q, CLINICAL_QUESTION["kermany"]],
    "aptos": [MODALITY_Q, REGION_Q, CLINICAL_QUESTION["aptos"]],
    "isic": [MODALITY_Q, CLINICAL_QUESTION["isic"]],
}


def full_accuracy(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """tab:vqa-full: accuracy (%) over n=20 for each (dataset, question) at each
    eps, for `column` in {clean, adv, random}. `clean` ignores eps."""
    rows = []
    for ds, questions in SCORED_QUESTIONS.items():
        for q in questions:
            sub = df[(df.dataset == ds) & (df.question == q)]
            if column == "clean":
                s = sub.drop_duplicates("image")
                v = s.apply(lambda r: is_correct(q, r["clean"], r["true_label"], ds), axis=1)
                rows.append(dict(dataset=ds, question=q, eps=None, acc=round(100 * (v == True).mean(), 1)))
            else:
                for eps in sorted(sub.eps.unique()):
                    e = sub[sub.eps == eps]
                    v = e.apply(lambda r: is_correct(q, r[column], r["true_label"], ds), axis=1)
                    rows.append(dict(dataset=ds, question=q, eps=int(eps), acc=round(100 * (v == True).mean(), 1)))
    return pd.DataFrame(rows)


def _clean_correct_cells(df, eps):
    """The modality/body-region rows at `eps` whose clean answer is correct."""
    for ds in SCORED_QUESTIONS:
        for q in (MODALITY_Q, REGION_Q):
            if q not in SCORED_QUESTIONS[ds]:
                continue
            sub = df[(df.dataset == ds) & (df.question == q) & (df.eps == eps)]
            for _, r in sub.iterrows():
                if is_correct(q, r["clean"], r["true_label"], ds) is True:
                    yield ds, q, r


def eps_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """tab:vqa-eps: per eps, pooled over the three datasets' modality and
    body-region cells that were correct when clean (the paper's n=99). Reports
    mean adv and random token flip %, mean adv displacement, and the % of cells
    still correct under adv and under random."""
    rows = []
    for eps in sorted(df.eps.unique()):
        cells = list(_clean_correct_cells(df, eps))
        n = len(cells)
        adv_ok = sum(is_correct(q, r["adv"], r["true_label"], ds) is True for ds, q, r in cells)
        rnd_ok = sum(is_correct(q, r["random"], r["true_label"], ds) is True for ds, q, r in cells)
        rows.append(dict(
            eps=int(eps), n=n,
            adv_flip_pct=round(sum(r["adv_flip_pct"] for _, _, r in cells) / n, 1),
            adv_disp=round(sum(r["adv_disp"] for _, _, r in cells) / n, 1),
            rnd_flip_pct=round(sum(r["rnd_flip_pct"] for _, _, r in cells) / n, 1),
            adv_correct_pct=round(100 * adv_ok / n, 1),
            rnd_correct_pct=round(100 * rnd_ok / n, 1),
        ))
    return pd.DataFrame(rows)


def direction_counts(df: pd.DataFrame) -> pd.DataFrame:
    """tab:vqa-direction: under the adv column, on images answered correctly when
    clean, counts (not rates) of disease erased (false negative) vs invented
    (false positive) for APTOS, and malignant vs benign flips for ISIC."""
    rows = []
    for ds in ("aptos", "isic"):
        q = CLINICAL_QUESTION[ds]
        for eps in sorted(df.eps.unique()):
            sub = df[(df.dataset == ds) & (df.question == q) & (df.eps == eps)]
            neg_flip = neg_tot = pos_flip = pos_tot = 0
            for _, r in sub.iterrows():
                clean_ok = is_correct(q, r["clean"], r["true_label"], ds)
                if clean_ok is not True:
                    continue
                is_pos = r["true_label"] == POSITIVE_LABEL[ds]
                flipped = is_correct(q, r["adv"], r["true_label"], ds) is False
                if is_pos:
                    pos_tot += 1
                    pos_flip += flipped
                else:
                    neg_tot += 1
                    neg_flip += flipped
            rows.append(dict(dataset=ds, eps=int(eps),
                             erased=pos_flip, erased_of=pos_tot,
                             invented=neg_flip, invented_of=neg_tot))
    return pd.DataFrame(rows)
