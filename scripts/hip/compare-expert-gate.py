"""Compare two gate result files: tokens and top logprobs must be identical.
usage: compare.py A.json B.json"""
import json, math, sys

a = json.load(open(sys.argv[1], encoding="utf-8"))
b = json.load(open(sys.argv[2], encoding="utf-8"))
assert len(a) == len(b), "different prompt counts"
n_tok = n_diff_tok = n_diff_lp = 0
first = None
for ra, rb in zip(a, b):
    ta, tb = ra["tokens"], rb["tokens"]
    if len(ta) != len(tb):
        print("prompt %s: token count %d vs %d" % (ra["prompt_id"], len(ta), len(tb)))
        n_diff_tok += abs(len(ta) - len(tb))
        first = first or (ra["prompt_id"], "token-count", len(ta), len(tb))
    invalid = set()
    for side, tokens in (("A", ta), ("B", tb)):
        for i, token in enumerate(tokens):
            values = [("logprob", token.get("logprob"))]
            values += [("top[%d]" % j, t[1]) for j, t in enumerate(token.get("top", []))]
            bad = [field for field, value in values
                   if not isinstance(value, (int, float)) or not math.isfinite(value)]
            if bad:
                print("prompt %s: invalid probability in %s token %d (%s)" %
                      (ra["prompt_id"], side, i, ", ".join(bad)))
                invalid.add(i)
                first = first or (ra["prompt_id"], i, side, "invalid probability")
    n_diff_lp += len(invalid)
    for i, (x, y) in enumerate(zip(ta, tb)):
        n_tok += 1
        if x["id"] != y["id"]:
            n_diff_tok += 1
            first = first or (ra["prompt_id"], i, x, y)
            break  # after a divergence the rest is a different sequence
        if i not in invalid and (x["logprob"] != y["logprob"] or x["top"] != y["top"]):
            n_diff_lp += 1
            first = first or (ra["prompt_id"], i, x, y)
tg_a = sum(r["timings"].get("predicted_per_second", 0) for r in a)/len(a)
tg_b = sum(r["timings"].get("predicted_per_second", 0) for r in b)/len(b)
print("tokens compared %d, token mismatches %d, logprob mismatches %d; mean tg %.2f vs %.2f t/s" % (n_tok, n_diff_tok, n_diff_lp, tg_a, tg_b))
if first:
    print("first difference:", json.dumps(first, ensure_ascii=False)[:400])
sys.exit(0 if n_diff_tok == 0 and n_diff_lp == 0 else 1)
