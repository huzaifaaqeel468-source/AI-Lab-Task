import argparse
import csv
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------------------------------------------------------
# Synthetic data
# ----------------------------------------------------------------------------
DOCS: Dict[str, str] = {
    "late": (
        "Assessment Handbook, Section 4 (synthetic teaching text). Every "
        "assignment in the course has a published deadline, and the deadline is "
        "shown on the learning portal at the start of the semester. Late "
        "submission is normally penalized. Illness extensions require medical "
        "evidence before the deadline. The penalty is ten percent of the marks "
        "for each day after the deadline, and work that is more than five days "
        "late is not marked at all unless the coordinator has approved it in "
        "writing. A certificate from a registered doctor is accepted, and the "
        "certificate must show the dates on which the student was unwell and "
        "the name of the clinic. Group projects follow the same rules, although "
        "the group leader submits the request on behalf of the whole team. Dr. "
        "Rahman, the course coordinator, answers extension questions during "
        "office hours on Tuesday afternoons, and the same office hours apply to "
        "questions about grading. In an older version of this handbook the "
        "penalty was five percent per day, so students should always check "
        "which version of the document they are reading before they plan their "
        "work around any deadline or penalty."
    ),
    "password": (
        "IT Help Desk Guide: Accounts and Passwords (synthetic teaching text). "
        "The university portal, the library system and the campus wireless "
        "network all use the same student login, so a problem with one of them "
        "usually means a problem with the account itself. Reset a forgotten "
        "password using the recovery link. The link is on the login page under "
        "the password box and it sends a one-time code to the phone number "
        "registered at admission. Never share recovery codes with anyone, "
        "including friends, classmates and people who claim to be calling from "
        "the help desk, because the real help desk will never ask for a code "
        "over the phone. If the registered phone number has changed, the "
        "student must visit the IT counter in the main building with a student "
        "card before the account can be recovered, and this visit cannot be "
        "replaced by an email or a phone call. Accounts that are locked after "
        "five wrong attempts open again by themselves after thirty minutes. "
        "Staff accounts follow a different process, where the department head "
        "approves the reset first. An older guide mentioned a security "
        "questions page, but that page was removed and should not be used any "
        "more."
    ),
    "assignment": (
        "Submission Guide for Coursework (synthetic teaching text). Most "
        "courses collect coursework electronically, and the instructions below "
        "apply unless a course outline says something different in writing. "
        "Upload assignments through the learning portal. Each upload page shows "
        "the deadline, the maximum file size of twenty megabytes and the list "
        "of accepted formats, and the page closes automatically a few seconds "
        "after the deadline passes. Submit the final file as a PDF, because "
        "Word documents sometimes change their layout when they are opened on "
        "another computer. Students who face a problem with the portal should "
        "take a screenshot of the error message and send it to the course "
        "coordinator before the deadline. Source code is submitted as a single "
        "zip file with a short readme, while hand written work must be scanned "
        "clearly and combined into one document. Email submissions were allowed "
        "in the past but are not accepted now, except when the portal is down "
        "for the whole university and the coordinator announces an alternative "
        "address. Resubmitting before the deadline replaces the previous "
        "upload, and only the latest file is marked."
    ),
    "contractor": (
        "Remote Work Policy: Employees and Contractors (synthetic teaching "
        "text). The university allows some roles to work away from campus when "
        "the work does not depend on laboratory equipment or face to face "
        "teaching. Employees may request remote work with manager approval, and "
        "the approval is recorded by the human resources office for a period of "
        "up to six months at a time. Contractors can work remotely only if the "
        "signed contract permits it. Employee rules do not apply to "
        "contractors. A contractor whose contract is silent on the subject "
        "should ask the project owner to add a written clause, and until that "
        "clause is signed the contractor is expected to work on campus during "
        "the agreed hours. Remote workers must stay reachable during core hours "
        "from ten in the morning to three in the afternoon and must use the "
        "university virtual private network when they open internal systems. "
        "Equipment such as laptops remains the property of the university, and "
        "a contractor who ends the contract has to return it within seven days. "
        "Some older notices said that contractors follow the employee rules, "
        "but those notices were replaced when the contractor policy was "
        "written."
    ),
    "printer": (
        "Equipment Troubleshooting: Office Printers (synthetic teaching text). "
        "This sheet covers the shared printers in the department offices, and "
        "it does not cover the scanners or the large plotter in the design lab, "
        "which have their own sheets. Before opening any panel, switch the "
        "printer off and wait for the internal parts to cool down, because the "
        "fuser unit stays hot for several minutes after use. For blank printer "
        "pages, check ink levels and run the cleaning utility. Print a test "
        "page afterwards and keep it, since the technician will ask to see it "
        "if the problem returns. If the pages are still blank after two "
        "cleaning cycles, do not keep repeating the utility, because every "
        "cycle uses ink and a clogged nozzle rarely clears on the third "
        "attempt. Replace the cartridge only with the model listed on the label "
        "inside the front cover. A blank page from a scanner is a different "
        "problem and usually means the document was placed face up on the glass "
        "or the wrong scanning source was selected. Paper jams, streaks and "
        "error codes are explained on the next sheet in this folder."
    ),
}

EVIDENCE: Dict[str, str] = {
    "late": "Illness extensions require medical evidence before the deadline.",
    "password": "Reset a forgotten password using the recovery link.",
    "assignment": "Upload assignments through the learning portal.",
    "contractor": "Contractors can work remotely only if the signed contract "
                  "permits it.",
    "printer": "For blank printer pages, check ink levels and run the "
               "cleaning utility.",
}

QUESTIONS: List[Tuple[str, Optional[str], str]] = [
    ("What is required for illness extensions?", "late", "dev"),
    ("How do I reset a forgotten password?", "password", "dev"),
    ("Where do I upload assignments?", "assignment", "dev"),
    ("Can contractors work remotely?", "contractor", "dev"),
    ("What are tomorrow's flight prices?", None, "dev"),
    ("What do I do for blank printer pages?", "printer", "test"),
    ("How can I recover my login access?", "password", "test"),
    ("How should I hand in my coursework?", "assignment", "test"),
    ("What proof is needed when I was ill?", "late", "test"),
    ("Who won the football match?", None, "test"),
]

PAIRS: Dict[str, Tuple[str, str]] = {
    "late": ("Late submission is normally penalized.",
             "Illness extensions require medical evidence before the deadline."),
    "password": ("Reset a forgotten password using the recovery link.",
                 "If the registered phone number has changed, the student must "
                 "visit the IT counter in the main building with a student card "
                 "before the account can be recovered, and this visit cannot be "
                 "replaced by an email or a phone call."),
    "assignment": ("Upload assignments through the learning portal.",
                   "Email submissions were allowed in the past but are not "
                   "accepted now, except when the portal is down for the whole "
                   "university and the coordinator announces an alternative "
                   "address."),
}

FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.",
     "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.",
     "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.",
     "source": "handbook.txt"},
    {"id": "faq:wifi", "question": "How do I connect to the campus wifi?",
     "answer": "Select the campus network and sign in with your student login.",
     "source": "it_help.txt"},
    {"id": "faq:library",
     "question": "How many books can I borrow from the library?",
     "answer": "Undergraduate students can borrow up to five books for "
               "fourteen days.",
     "source": "library_rules.txt"},
]

TOKEN_SAMPLES = [
    ("punctuation", "AI, search!"),
    ("contraction", "I can't log in."),
    ("email", "help@example.com"),
    ("identifier", "CS4944"),
    ("empty", ""),
    ("urdu", "یہ ایک مثال ہے"),
    ("mixed", "Mail it@uni.edu.pk about CS4944's v2.1 re-sit"),
]

LABEL = {"0": "window 40, overlap 0", "10": "window 40, overlap 10",
         "sentence": "sentence-aware, budget 40"}


def snip(text: str, n: int = 75) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 3] + "..."

def show_tokens(model) -> None:
    for label, text in TOKEN_SAMPLES:
        ws = text.split()
        words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
        print(f"[{label}] original: {text!r}")
        print(f"    whitespace ({len(ws)}): {ws}")
        print(f"    word-like  ({len(words)}): {words}")
    assert re.findall(r"\b\w+\b", "") == []

    if model is None:
        print("Model sub-word tokens: skipped (sentence-transformers not available)")
        return
    print("Model sub-word tokens (all-MiniLM-L6-v2):")
    for text in ["I can't log in.", "help@example.com"]:
        toks = model.tokenizer.tokenize(text)
        print(f"    {text!r}: {toks} ({len(toks)})")

def word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def windows(text: str, size: int = 40, overlap: int = 10,
            offset: int = 0) -> List[Tuple[int, int]]:
    """Fixed word windows; size and overlap are counted in whitespace words."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    spans = word_spans(text)
    result = []
    for i in range(0, len(spans), size - overlap):
        j = min(i + size, len(spans))
        result.append((offset + spans[i][0], offset + spans[j - 1][1]))
        if j == len(spans):
            break
    return result


def sentence_chunks(text: str, size: int = 40) -> List[Tuple[int, int]]:
    """Pack whole sentences up to a word budget; long sentences use windows."""
    if size <= 0:
        raise ValueError("size must be positive")
    # simple teaching splitter: . ! or ? followed by whitespace
    cuts = [0] + [m.end() for m in re.finditer(r"[.!?]\s+", text)] + [len(text)]
    out, pending, count = [], None, 0
    for a, b in zip(cuts, cuts[1:]):
        n = len(text[a:b].split())
        if not n:
            continue
        if n > size:
            if pending is not None:
                out.append(pending)
            out.extend(windows(text[a:b], size, 0, a))
            pending, count = None, 0
        elif pending is not None and count + n <= size:
            pending, count = (pending[0], b), count + n
        else:
            if pending is not None:
                out.append(pending)
            pending, count = (a, b), n
    if pending is not None:
        out.append(pending)
    return out


def make_chunks(docs: Dict[str, str], strategy: str) -> List[dict]:
    result = []
    for doc_id, text in docs.items():
        spans = (sentence_chunks(text) if strategy == "sentence"
                 else windows(text, 40, int(strategy)))
        for i, (a, b) in enumerate(spans):
            result.append(dict(id=f"{doc_id}:{i}", doc=doc_id,
                               start=a, end=b, text=text[a:b]))
    return result


def run_chunk_tests() -> None:
    assert windows("") == []
    print("empty text            -> windows:", windows(""),
          "| sentence:", sentence_chunks(""))
    assert windows("short text") == [(0, 10)]
    print("2-word document       -> one window:", windows("short text"))
    for ov in (2, 5, -1):
        try:
            windows("a b c", 2, ov)
        except ValueError:
            print(f"size=2, overlap={ov:<2}    -> ValueError (rejected)")
        else:
            raise AssertionError("invalid overlap accepted")
    long_text = " ".join(["word"] * 95) + "."
    sizes = [len(long_text[a:b].split()) for a, b in sentence_chunks(long_text)]
    assert all(s <= 40 for s in sizes)
    print("95-word sentence      -> fallback chunk sizes", sizes)
    tiny = {"example": "Reset your password. Keep recovery codes private."}
    for strategy in ["0", "10", "sentence"]:
        for c in make_chunks(tiny, strategy):
            assert c["text"] == tiny[c["doc"]][c["start"]:c["end"]]
    print("offset check          -> text == source[start:end] for all chunks")

    demo = "Dr. Rahman approves late work. Ask him on Tuesday, e.g. at noon. Thanks."
    print("\nSentence splitter failure (abbreviations), budget of 6 words:")
    print("   input:", demo)
    for a, b in sentence_chunks(demo, size=6):
        print("   chunk:", repr(demo[a:b]))


def compare_chunking() -> None:
    docs = {k: DOCS[k] for k in ["late", "password", "assignment"]}
    original = sum(len(t.split()) for t in docs.values())
    print(f"Documents: {list(docs)} | original words: {original}")
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(docs, strategy)
        sizes = [len(c["text"].split()) for c in chunks]
        print(f"\n{LABEL[strategy]}")
        print(f"   chunks={len(chunks)}  words min/avg/max="
              f"{min(sizes)}/{sum(sizes) / len(sizes):.1f}/{max(sizes)}  "
              f"duplicate words={sum(sizes) - original}")
        for d, (rule, exc) in PAIRS.items():
            rule_ok = any(c["doc"] == d and rule in c["text"] for c in chunks)
            exc_ok = any(c["doc"] == d and exc in c["text"] for c in chunks)
            both = any(c["doc"] == d and rule in c["text"] and exc in c["text"]
                       for c in chunks)
            print(f"   {d:<11} rule whole: {str(rule_ok):<5} "
                  f"exception whole: {str(exc_ok):<5} together: {both}")
        tiny = [c["id"] for c in chunks if len(c["text"].split()) < 4]
        if tiny:
            print("   very small chunks:", tiny)


def load_model():
    """Load the sentence encoder if it is available; otherwise return None."""
    try:
        from sentence_transformers import SentenceTransformer
        start = time.perf_counter()
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        return model, time.perf_counter() - start
    except Exception as err:                       # not installed / no download
        print(f"Embedding model not available ({type(err).__name__}); "
              "only the TF-IDF baseline will be run.")
        return None, 0.0


class Retriever:
    """Index text records once, then score any number of queries."""

    def __init__(self, records: List[dict], method: str = "tfidf", model=None):
        if not records or any(not r["text"].strip() for r in records):
            raise ValueError("Provide nonempty records")
        self.records, self.method, self.model = records, method, model
        texts = [r["text"] for r in records]
        start = time.perf_counter()
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.matrix = model.encode(texts, normalize_embeddings=True)
        self.index_seconds = time.perf_counter() - start

    def scores(self, queries: List[str]) -> np.ndarray:
        if self.method == "tfidf":
            return (self.vectorizer.transform(queries) @ self.matrix.T).toarray()
        q = self.model.encode(queries, normalize_embeddings=True)
        return q @ self.matrix.T

    def search(self, query: str, k: int = 3) -> List[Tuple[dict, float]]:
        if not query.strip() or k <= 0:
            return []
        if self.method == "tfidf" and self.vectorizer.transform([query]).nnz == 0:
            return []                              # no shared vocabulary
        row = self.scores([query])[0]
        top = np.argsort(-row, kind="stable")[:k]
        return [(self.records[i], float(row[i])) for i in top]


def first_relevant_rank(row: np.ndarray, chunks: List[dict], q: dict):
    """Rank of the first relevant chunk in the top 3 (None if there is none).

    A chunk is relevant only if it comes from the right document and holds
    the whole labelled evidence sentence.
    """
    for rank, i in enumerate(np.argsort(-row, kind="stable")[:3], 1):
        if chunks[i]["doc"] == q["doc"] and q["evidence"] in chunks[i]["text"]:
            return rank
    return None


def hit_mrr(scores: np.ndarray, chunks: List[dict], items: List[dict]):
    hits, rr = [], []
    for row, q in zip(scores, items):
        if q["doc"] is None:
            continue                               # unanswerable: not scored
        r = first_relevant_rank(row, chunks, q)
        hits.append(r is not None)
        rr.append(1 / r if r else 0.0)
    return float(np.mean(hits)), float(np.mean(rr))


def evaluate(model, model_load: float):
    items = [dict(query=q, doc=d, evidence=EVIDENCE[d] if d else None, split=s)
             for q, d, s in QUESTIONS]
    for q in items:
        if q["doc"] is not None:
            assert q["evidence"] in DOCS[q["doc"]]
    query_texts = [q["query"] for q in items]
    original = sum(len(t.split()) for t in DOCS.values())
    methods = ["tfidf"] + (["embedding"] if model is not None else [])

    results, logs = [], []
    if model is not None:
        print(f"Model load time: {model_load:.3f} s (not included below)")
    for strategy in ["0", "10", "sentence"]:
        chunks = make_chunks(DOCS, strategy)
        if model is not None:                      # token length audit
            longest = max(len(model.tokenizer(t, truncation=False)["input_ids"])
                          for t in [c["text"] for c in chunks] + query_texts)
            print(f"Token audit, {LABEL[strategy]}: longest input {longest} "
                  f"tokens, limit {model.max_seq_length}")
            if longest > model.max_seq_length:
                raise ValueError("Input exceeds encoder limit; rechunk")
        for method in methods:
            engine = Retriever(chunks, method, model)
            start = time.perf_counter()
            scores = engine.scores(query_texts)
            query_seconds = time.perf_counter() - start
            row = dict(strategy=strategy, method=method, chunks=chunks,
                       scores=scores, items=items, engine=engine,
                       duplicate=sum(len(c["text"].split()) for c in chunks) - original,
                       index_s=engine.index_seconds, query_s=query_seconds)
            for split in ["dev", "test"]:
                idx = [i for i, q in enumerate(items) if q["split"] == split]
                row[split] = hit_mrr(scores[idx], chunks, [items[i] for i in idx])
            # rejection threshold calibrated on DEV unanswerable queries only
            top1 = scores.max(axis=1)
            dev_un = [top1[i] for i, q in enumerate(items)
                      if q["split"] == "dev" and q["doc"] is None]
            row["threshold"] = float(max(dev_un))
            row["test_kept"] = [bool(top1[i] > row["threshold"])
                                for i, q in enumerate(items)
                                if q["split"] == "test" and q["doc"] is not None]
            row["test_rejected"] = [bool(top1[i] <= row["threshold"])
                                    for i, q in enumerate(items)
                                    if q["split"] == "test" and q["doc"] is None]
            results.append(row)
            for row_scores, q in zip(scores, items):
                for rank, i in enumerate(np.argsort(-row_scores, kind="stable")[:3], 1):
                    c = chunks[i]
                    logs.append(dict(strategy=LABEL[strategy], method=method,
                                     query=q["query"], split=q["split"],
                                     answerable=q["doc"] is not None, rank=rank,
                                     score=round(float(row_scores[i]), 4),
                                     chunk_id=c["id"], start=c["start"],
                                     end=c["end"], text=c["text"]))
    return results, logs, items


def print_results(results, items) -> dict:
    print(f"{'strategy':<26}{'method':<10}{'chunks':>6}{'dup':>5}  "
          f"{'dev H@3':>7}{'dev MRR':>8}{'test H@3':>9}{'test MRR':>9}"
          f"{'index s':>9}{'query s':>9}")
    for r in results:
        print(f"{LABEL[r['strategy']]:<26}{r['method']:<10}{len(r['chunks']):>6}"
              f"{r['duplicate']:>5}  {r['dev'][0]:>7.3f}{r['dev'][1]:>8.3f}"
              f"{r['test'][0]:>9.3f}{r['test'][1]:>9.3f}"
              f"{r['index_s']:>9.4f}{r['query_s']:>9.5f}")

    print("\nStrategy chosen on DEV results only (best MRR@3, then Hit@3, "
          "then fewer duplicate words):")
    chosen = {}
    for method in sorted({r["method"] for r in results}):
        pool = [r for r in results if r["method"] == method]
        best = max(pool, key=lambda r: (round(r["dev"][1], 6), round(r["dev"][0], 6),
                                        -r["duplicate"]))
        chosen[method] = best
        print(f"   {method:<10}-> {LABEL[best['strategy']]} | held-out test: "
              f"Hit@3={best['test'][0]:.3f}, MRR@3={best['test'][1]:.3f} "
              f"(not used for tuning)")

    print("\nRejection threshold from the dev unanswerable query "
          "(top-1 score must be above it):")
    for r in results:
        print(f"   {LABEL[r['strategy']]:<26}{r['method']:<10}"
              f"threshold={r['threshold']:.3f}  test answerable kept "
              f"{sum(r['test_kept'])}/{len(r['test_kept'])}, test unanswerable "
              f"rejected {sum(r['test_rejected'])}/{len(r['test_rejected'])}")
    return chosen


def print_top3(row, items) -> None:
    print(f"\nTop-3 results: {LABEL[row['strategy']]} + {row['method']}")
    for scores, q in zip(row["scores"], items):
        tag = "" if q["doc"] else "   <unanswerable>"
        print(f"  [{q['split']}] {q['query']}{tag}")
        for rank, i in enumerate(np.argsort(-scores, kind="stable")[:3], 1):
            c = row["chunks"][i]
            print(f"     {rank}. {c['id']:<14} score={scores[i]:.3f} "
                  f"chars {c['start']}-{c['end']} | {snip(c['text'])}")


def print_failures(results) -> None:
    shown = 0
    for r in results:
        if r["method"] != "tfidf":
            continue
        for scores, q in zip(r["scores"], r["items"]):
            if q["doc"] is None:
                continue
            rank = first_relevant_rank(scores, r["chunks"], q)
            if rank == 1:
                continue
            top = r["chunks"][int(np.argsort(-scores, kind="stable")[0])]
            where = "not in top 3" if rank is None else f"first relevant at rank {rank}"
            print(f"- {LABEL[r['strategy']]} | {q['query']} ({q['split']})")
            print(f"    {where}; top-1 = {top['id']} -> {snip(top['text'], 90)}")
            shown += 1
    print(f"({shown} answerable queries did not have a relevant chunk at rank 1 "
          "in the TF-IDF configurations)")

    # the two unanswerable queries still get ranked text
    best = [r for r in results if r["method"] == "tfidf" and r["strategy"] == "sentence"][0]
    print("\nUnanswerable queries still return text (sentence-aware, TF-IDF):")
    for scores, q in zip(best["scores"], best["items"]):
        if q["doc"] is None:
            c = best["chunks"][int(np.argsort(-scores, kind="stable")[0])]
            print(f"- {q['query']} -> {c['id']} (score {scores.max():.3f}): "
                  f"{snip(c['text'], 70)}")

    tiny = [c["id"] + " = " + repr(c["text"].strip())
            for c in make_chunks(DOCS, "sentence") if len(c["text"].split()) < 3]
    print("\nOrphan chunk created by the long-sentence fallback:", tiny)


def build_faq_bot(method: str = "tfidf", model=None) -> Retriever:
    records = [dict(r, text=r["question"] + " " + r["answer"]) for r in FAQ]
    return Retriever(records, method, model)


def faq_reply(engine: Retriever, query: str) -> str:
    if not query.strip():
        return "Please enter a question."
    # only an exact canonical question is answered automatically
    for r in FAQ:
        if query.strip().casefold() == r["question"].casefold():
            return r["answer"] + " [source: " + r["source"] + "]"
    hits = engine.search(query)
    if not hits:
        return "No matching FAQ. Please contact the help desk."
    lines = ["Possible matches; choose or rephrase your question:"]
    for r, score in hits:
        lines.append("   " + r["question"] + f" (similarity {score:.3f})")
    return "\n".join(lines)


def evidence_reply(engine: Retriever, query: str, choose) -> None:
    """Show candidate evidence; display a passage only after the user picks one."""
    hits = engine.search(query)
    if not hits:
        print("Bot: No candidate evidence; rephrase or contact a human.")
        return
    print("Bot: Candidate evidence; relevance must be checked:")
    for n, (r, score) in enumerate(hits, 1):
        print(f"   {n}. {r['id']} score={score:.3f} characters {r['start']}-{r['end']}")
    chosen = choose()
    if not chosen.isdigit() or not 1 <= int(chosen) <= len(hits):
        print("Bot: No answer selected. Rephrase or contact a human.")
        return
    r, score = hits[int(chosen) - 1]
    print("Bot: Retrieved passage:", " ".join(r["text"].split()))
    print(f"Bot: Source: {r['doc']}, characters {r['start']}-{r['end']}")


def demo_chatbots(selected: dict, model) -> None:
    print("FAQ chatbot (TF-IDF), exact answers only for canonical questions:")
    faq = build_faq_bot()
    for q in ["How do I reset my password?",
              "How many books can I borrow from the library?",
              "password recovery", "wifi connect", "", "volcano geology"]:
        print("You:", q)
        print("Bot:", faq_reply(faq, q))

    row = selected["tfidf"]
    print(f"\nDocument evidence chatbot ({LABEL[row['strategy']]}, TF-IDF):")
    script = [("Can contractors work remotely?", "1"),
              ("Who won the football match?", ""),
              ("volcano geology", ""), ("", ""),
              ("How can I recover my login access?", "1")]
    for query, choice in script:
        print("You:", query)
        evidence_reply(row["engine"], query,
                       lambda c=choice: (print("You (choice):", repr(c)), c)[1])


def interactive(kind: str, selected: dict, model) -> None:
    if kind == "faq":
        engine = build_faq_bot("tfidf" if model is None else "embedding", model)
        print("FAQ assistant. Type exit to stop.")
        while True:
            try:
                query = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() == "exit":
                break
            print("Bot:", faq_reply(engine, query))
    else:
        method = "embedding" if "embedding" in selected else "tfidf"
        engine = selected[method]["engine"]
        print("Evidence assistant. Type exit to stop.")
        while True:
            try:
                query = input("Question: ")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() == "exit":
                break
            evidence_reply(engine, query,
                           lambda: input("Choose 1-3 to inspect; Enter if none apply: "))


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", choices=["faq", "docs"])
    args = parser.parse_args()

    print("LAB 6 - TOKENIZATION, EMBEDDINGS AND TEXT CHUNKING")
    model, model_load = load_model()
    words = {k: len(v.split()) for k, v in DOCS.items()}
    print(f"Corpus: {len(DOCS)} documents, words per document {words}")
    print(f"Queries: {sum(1 for q in QUESTIONS if q[1])} answerable "
          f"+ {sum(1 for q in QUESTIONS if not q[1])} unanswerable")

    banner("1. TOKENIZATION")
    show_tokens(model)

    banner("2. CHUNKING TESTS")
    run_chunk_tests()

    banner("3. CHUNK COMPARISON (rule + exception survival)")
    compare_chunking()

    banner("4. RETRIEVAL EVALUATION")
    results, logs, items = evaluate(model, model_load)
    selected = print_results(results, items)
    with open("retrieval_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    print("\nTop-3 log for every configuration written to retrieval_log.csv")

    banner("5. TOP-3 SOURCE-TRACEABLE RESULTS")
    print_top3(selected["tfidf"], items)

    banner("6. FAILURE AND EDGE CASES")
    print_failures(results)

    banner("7. CHATBOT TRANSCRIPTS")
    demo_chatbots(selected, model)

    banner("8. SUMMARY")
    best = selected["tfidf"]
    print(f"Selected on dev (TF-IDF): {LABEL[best['strategy']]}")
    print("Hit@3 and MRR@3 are computed on 4 queries per split, so one query "
          "changes Hit@3 by 0.25;")
    print("the results show tendencies only. Retrieved text is evidence to "
          "inspect, not a verified answer.")

    if args.chat:
        banner("INTERACTIVE MODE")
        interactive(args.chat, selected, model)


if __name__ == "__main__":
    main()
