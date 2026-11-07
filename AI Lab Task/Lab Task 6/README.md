# Lab 6 - NLP Fundamentals: Tokenization, Embeddings and Text Chunking

## Files
- `lab6_nlp_retrieval.py` - complete Python implementation (tokenization, chunking,
  retrieval evaluation, FAQ chatbot and evidence chatbot in one file).
- `program_output.txt` - output produced by running the program.
- `retrieval_log.csv` - top-3 results with scores, chunk IDs and source offsets for
  every query and every configuration (written by the program).
- `Lab6_Report.docx` - written explanation, tables, failure cases and reflection.

## Requirements
- Python 3.10 or later
- `numpy` and `scikit-learn` (needed for the TF-IDF baseline)
- `sentence-transformers` (optional, adds the real sentence-embedding comparison;
  uses `sentence-transformers/all-MiniLM-L6-v2` on CPU, no GPU or API key)

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS or Linux: source .venv/bin/activate
python -m pip install numpy scikit-learn sentence-transformers
```

## How to run
1. Keep `lab6_nlp_retrieval.py` in a folder.
2. Open Command Prompt or a terminal in that folder.
3. Run:

```bash
python lab6_nlp_retrieval.py
```

The program compares whitespace, word-like and model tokens, tests the chunkers
(empty text, short text, invalid overlap, long sentence), compares word windows
(overlap 0 and 10) with sentence-aware packing, evaluates Hit@3 and MRR@3 on
development and test queries, lists failure cases and prints chatbot transcripts.

Optional interactive modes:

```bash
python lab6_nlp_retrieval.py --chat faq     # FAQ bot (exact answers, otherwise suggestions)
python lab6_nlp_retrieval.py --chat docs    # evidence bot (shows source and offsets)
```
Type `exit` to stop.

## Note about the embedding model
If `sentence-transformers` is not installed or the model cannot be downloaded, the
program says so and runs only the TF-IDF baseline. The first run with the library
installed downloads the encoder (internet needed once); after that the cache is
used. The program then also reports the model load time, the token-length audit
(limit 256 word pieces) and the three embedding configurations.

## Note about the data
All documents, FAQs and queries are synthetic teaching data written for this lab.
They are not real university policies and must not be used for real decisions.
The evaluation has only 8 answerable and 2 unanswerable queries (4 dev + 4 test
answerable), so the results show tendencies only.
