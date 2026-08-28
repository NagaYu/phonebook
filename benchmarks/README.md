# benchmarks/

Output directory for `scripts/evaluate.py`.

| File | Contents |
|---|---|
| `results.json` | Raw metrics for every condition and every evaluation set; the source for the figures |
| `RESULTS.md` | Per-split Markdown tables (pasted into the README) |
| `llm_cache.jsonl` | Cached responses for condition (B), for reproducibility and cost control |

## How to read these

1. **Read per split.** An overall average is dominated by the abundance of known
   entities.
2. **Check `metadata.synthetic`.** If it is `true` the run only validates the
   pipeline and is not evidence for any claim.
3. **Conditions with `available: false` are blank on purpose.** No number is
   invented for a condition that was not executed.
4. **Look at both the strict and the long-vowel-normalized exact match.**
   Existing G2P emits a pronunciation form, so the strict metric alone
   understates it for reasons of notation rather than knowledge.
5. **Do not compare a rejection-filtered accuracy against other conditions.**
   The main tables are computed without rejection; rejection has its own table.
