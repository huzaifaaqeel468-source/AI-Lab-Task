# Lab 5 - Minimax and Alpha-Beta Pruning

## Files
- `lab5_minimax_alpha_beta.py` - complete Python implementation.
- `program_output.txt` - output produced by running the program.
- `Lab5_Report.docx` - written explanation, hand trace, comparison and reflection.

## Requirements
- Python 3.8 or later
- No third-party Python packages are needed for the program.

## How to run
1. Install Python if it is not already installed.
2. Keep `lab5_minimax_alpha_beta.py` in a folder.
3. Open Command Prompt or a terminal in that folder.
4. Run:

```bash
python lab5_minimax_alpha_beta.py
```

The program prints the Minimax trace, counts evaluated leaves for both
algorithms, reverses the leaf order, and checks that the root value stays
the same.

## Note about the supplied values
The Minimax example in the lab manual gives six scores: `[3, 5, 2, 9, 0, 7]`.
Its sample code pads the list to eight leaves with two zeroes to form a complete
binary tree. This submission follows that same approach so the trace matches
the supplied example.
