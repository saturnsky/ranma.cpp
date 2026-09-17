# Expert cache: profile banks and the prefill swap (`--expert-prefill-swap`)

## What it is

The expert cache (`expert-cache.md`) profiles which MoE experts the router selects and keeps the
most valuable ones in a VRAM arena. A request has two phases that select different experts: the
prompt is processed in batches of hundreds of tokens, then tokens are generated one at a time. This
page is about keeping one profile per phase, and about the option that lets the arena hold a
different plan in each of them.

Two banks, `decode` and `prefill`, each with its own directory of records under
`--expert-profile-dir`. The split is not optional: the prompt phase of the profiled slot is always
recorded in `prefill` and the generation phase always in `decode`, with the swap on or off, so that
no plan is ever derived from a mixture of the two phases weighted by token count.

By default only the `decode` plan is ever installed, at the end of every request, so a prompt is
processed with the set of experts that generation wanted. With `--expert-prefill-swap` the arena
holds the `prefill` plan while a prompt is processed and the `decode` plan while tokens are
generated, and every request costs two installs. The swap is off by default; whether it pays depends
on how much of the model is resident, and the numbers below say what it was worth on this machine
in both directions.

## Why it exists

A prompt batch and a decode step read the same weights through different kernels, but that is not
why the two phases differ. They differ in which experts they select: a long prompt of ordinary text
spreads its selections over most of the 512 experts of a layer, while generation of one reply
concentrates on far fewer. A plan made from generation records is therefore a poor plan for prompt
processing, and a plan made from prompt records is a poor plan for generation.

Generation comes first. This cache exists for decode throughput, and a bank that counted both phases
would plan from a histogram in which one 512-token prompt ubatch outweighs hundreds of generated
tokens; the decode hit rate would fall and with it the decode throughput. So the banks are always
split and, with the swap off, only the decode bank ever plans the arena. The swap is the option for
a server that prefills long prompts: it accepts a small cost on short requests to process a long
prompt faster, so that generation starts sooner.

Profiling both banks whether or not the swap is on is a deliberate choice: a user who turns the
swap on later finds a prompt-processing history already there, instead of a cold arena for the
first few requests.

The plan is installed at a request boundary, not on an idle timer. The server knows exactly when a
request starts and ends, and a boundary is a better moment than a stopwatch: it is the moment at
which the next phase is known.

## How it works

The backend knows bank ids and plan ids and nothing else (`ggml/include/ggml-expert.h`). The policy
object (`common/expert-policy.h`), driven by the server and by `llama-bench`, names the moments. It
profiles slot 0 only; other slots use the cache but feed no bank.

| Server event | What the policy does |
|---|---|
| model load end | the backend installs the seed plan, see below |
| the banks are opened | `bank_discard` of both, so nothing from the load-time warm-up is a record |
| slot 0 starts prompt processing | `bank_mark(prefill)`, profile slot 0 into `prefill`; **swap on:** install the newest `prefill` plan |
| the prompt is done, generation starts | profile slot 0 into `decode`; `bank_commit(prefill)` with the processed prompt tokens; **swap on:** install the newest `decode` plan; `bank_mark(decode)` |
| the request ends, normally or by a client abort | stop profiling; `bank_commit(decode)` with the generated tokens; **swap off:** install the newest `decode` plan |
| a decode fails or a slot is reset mid-request | `bank_discard` of whatever interval is open |
| an install is refused because another slot is computing | remember the plan and install it on the first `update_slots` pass in which every slot is idle |

With the swap off no plan is ever installed from the `prefill` bank: the newest `decode` plan goes in
at the end of every request and stays through the next prompt. With the swap on the prefill plan
goes in when a prompt starts and the decode plan when generation starts.

An interval that saw no token is a discard, not a record: a prompt served entirely from the prompt
cache, or a request that generated nothing, would otherwise tell the score window that every expert
was unwanted for one request.

**The seed plan.** The config field `initial_bank` is a preference list of labels. The server passes
`prefill,decode` when the swap is on and `decode` when it is off, and the backend seeds the
load-time plan from the first of those banks that has stored records.

**The drain.** An install rewrites the arena slots and the device tables, so nothing may be reading
them. There are two checked steps. `llama_context` counts the encode and decode calls on its stack
and `llama_expert_plan_install` refuses with a debug line when the count is not zero; then it
synchronizes the context, which resolves the outputs of the last compute. The backend then
synchronizes the device and asserts its own preconditions. The server's `update_slots` is single
threaded, so with `-np 1` the counter never trips; it is there so that the deferred-install rule is a
checked condition and not an assumption.

**One live plan per bank.** A commit builds a plan and returns its id. The controller keeps the
newest plan of each bank plus the installed one and drops the rest, so the plan table cannot grow
with the request count. Plan ids are never reused: installing an id that has been dropped fails with
a log line instead of installing a different plan.

**Installing what is already there is free.** The controller compares the plan's selection with the
arena's current one before it drains the device, so a boundary that does not change the set costs
one log line and no synchronize.

**What an interval read.** Under `RANMA_EXPERT_TRACE` bit 3 every commit reports a round line: the
bytes its interval selected, split into the ones that came from the VRAM arena, from host memory and
from the file, and the profile-save time. It is computed from the bank histogram that the commit
reads back anyway, against the plan that was installed while the interval ran, so it costs nothing
on the hot path. This is the only direct measurement of what a plan is worth during prompt
processing.

**Which bank is the prompt bank.** `bank_open` takes a `prompt_bank` flag. The backend stores it and
compares no label string: the flag decides which ring an install carries with a finite host tier
(`expert-cache-l2.md`). The policy opens `decode` with the flag clear and `prefill` with it set.

**Freeze refuses installs in the policy.** `--expert-freeze` returns from the policy's install before
it reaches the backend, so a frozen run never installs a plan; banks are still marked, committed and
scored.

## How to use it

```
llama-server -m MODEL -ngl 999 -c 8192 --load-mode none \
    --expert-l1-mib 20000 --expert-profile-dir DIR --expert-prefill-swap
```

with `GGML_CUDA_HOST_DIRECT=1` and `GGML_CUDA_HOST_DIRECT_MAX_BATCH=512`. Without
`--expert-prefill-swap` the two banks are still filled; `DIR/prefill/` and `DIR/decode/` each hold
their own records and manifest.

The validator refuses `--expert-prefill-swap` before the model loads when it cannot work: without an
expert cache budget, without a profile directory, with `-np` greater than 1, with a multimodal
projector, or with speculative decoding of any kind. The swap installs a plan inside a request, so
it needs a server with exactly one sequence and no second decode path.

`llama-bench` drives the same policy object: its depth fill and PP512 test feed the prefill bank and
its TG128 test the decode bank, and `--expert-prefill-swap` is a Warm option there (`benchmark.md`).
`tests/test-server-expert.cpp` checks the whole call order above against a fake backend, with the
swap off and on, so the policy is tested without a GPU.

## Measured effect

Development machine: Radeon AI PRO R9700 (32 GiB, gfx1201) on PCIe 5.0 x16, headless, power limit
-30 %, voltage offset 0 mV; Ryzen 9 7950X3D, 128 GiB DDR5-5600, Windows 11, ROCm 10. Model:
Qwen3.8-Flash-Next UD-Q4_K_XL (73450 MiB of routed expert weights). `llama-bench` PP512 / TG128,
exclusive mode, host-direct on, `--load-mode none`. `ctl ms` is the control time per test that
`llama-bench` reports next to the throughput (`benchmark.md`): the commits and the installs, which
are not inside the tokens-per-second figures.

**This commit alone.** 3072 MiB budget, unlimited host memory, `-r 1`, depths `8192,0,8192` with the
first pair discarded, both rows Warm from one Cold seed that carries both banks:

| swap | PP512 @ 0 | TG128 @ 0 | ctl ms PP / TG |
|---|---:|---:|---|
| off | 557.02 | 30.28 | 8.6 / 24.5 |
| on | 575.41 | 29.81 | 17.9 / 46.3 |

**With a large budget and unlimited host memory.** From the published curve, depths 0 / 4096 /
8192 / 32768 / 65536 after a discarded 65536 pass, R9700 row, 20000 MiB budget:

| mode | swap | PP512 | TG128 | ctl ms TG @ 0 |
|---|---|---|---|---:|
| inclusive | off | 864.89, 863.78, 842.60, 733.35, 609.81 | 38.00, 37.71, 37.02, 32.04, 26.89 | 45 |
| inclusive | on | 942.85, 943.39, 897.43, 762.83, 616.78 | 38.49, 38.16, 37.46, 31.75, 26.71 | 103 |
| exclusive | off | 937.43, 927.21, 898.39, 771.49, 631.95 | 38.90, 38.52, 37.62, 32.01, 26.89 | 74 |
| exclusive | on | 947.95, 953.39, 916.02, 786.23, 643.71 | 38.20, 37.78, 37.36, 31.90, 26.75 | 183 |

The RX 9070 XT emulation row (3072 MiB budget) shows the same picture with smaller numbers: the four
Warm modes are within the base-to-base drift of each other on both axes.

With every expert resident in VRAM or host memory the swap buys 1 to 9 % of prompt throughput and
costs 1 to 2 % of decode, plus a boundary install of about 0.1 to 0.2 s per request. The reason is
that a prompt ubatch reads an expert once whether one or a hundred of its tokens selected it, so
what the arena saves is decided by whether an expert is selected at all in the ubatch. At 512
tokens the experts at the top of either bank are selected in nearly every ubatch, so exchanging the
top of the decode bank for the top of the prefill bank changes little; the two banks differ in their
tails, and the tails are not in the arena.

**With a finite host tier.** Same protocol, `--expert-l2-mib 40960` (the placement of a 64 GB
machine, `benchmark.md`):

| device row | mode | PP512 | TG128 | ctl ms TG @ 0 |
|---|---|---|---|---:|
| R9700, 20000 MiB | exclusive | 756.18, 772.05, 759.96, 692.15, 576.16 | 37.92, 37.63, 36.76, 31.44, 26.44 | 76 |
| R9700, 20000 MiB | exclusive + swap | 916.86, 915.62, 878.40, 765.02, 626.96 | 37.85, 37.58, 36.77, 31.39, 26.50 | 2510 |
| RX 9070 XT emulation, 3072 MiB | exclusive | 517.19, 501.77, 486.27, 484.53, 401.05 | 29.20, 30.10, 29.75, 25.84, 22.40 | 28 |
| RX 9070 XT emulation, 3072 MiB | exclusive + swap | 589.51, 570.30, 564.62, 524.99, 453.37 | 29.12, 29.40, 29.19, 26.09, 22.51 | 3327 |

Here the swap is worth +15 % (R9700) and +13 % (emulated 16 GiB card) of prompt throughput at every
depth, at the same decode throughput. The tails now matter: the plan also decides which experts stay
in host memory and which stay in the file, and a file read costs an SSD access instead of a PCIe
read. The prefill plan chooses that boundary by prompt-processing demand, so a prompt ubatch reads
fewer bytes from the file.

The price is in the last column. With a finite tier the install at the prompt-to-generation
boundary has to bring back the decode residents that the prefill plan pushed to the file, and that
is a read from the SSD: 2.5 s on the R9700 row and 3.3 s on the emulated card per request, against
0.1 to 0.2 s with unlimited host memory. `llama-bench` reports it separately; a server pays it as
time to the first token of every request. On a request with a short prompt that install costs more
than the better prompt plan returns.

So the honest answer is that the swap is a long-prompt feature for a host tier that does not hold
the whole model. On a server that prefills thousands of tokens per request with a finite host tier
it is worth double-digit percent of the prefill rate; with the whole model resident it is worth a
few percent; on short requests it is a per-request cost with nothing to show for it. That is why it
is an option and why it is off.

**Correctness.** Eight prompts x 48 greedy tokens with top-3 logprobs, exclusive 3072 MiB, finite
8192 MiB host tier, swap on, against the cache off on the same binary: identical
(`expert-cache.md`).

## What it costs

- **Two installs per request instead of one**, when the swap is on. With unlimited host memory the
  extra install is a delta of the arena, 0.1 to 0.2 s at a 20000 MiB budget; with a finite host
  tier it includes SSD reads and is seconds, see above. The prompt-to-generation install is inside
  the request, so it is added to the time to the first token.
- **A second histogram in VRAM.** Four banks are allocated at model load (`n_counts * 4` bytes each,
  96 KiB per bank here); the second one is simply used.
- **A second directory of records** under the profile directory, the same size as the first.
- **Per commit**: one device synchronize and one histogram read-back per bank, so two per request
  instead of one.
- Nothing per token. The profiler kernel already ran for the decode bank; it now also runs for the
  prompt batches of slot 0, which is one atomic add per selected expert per row.

## Limitations

- **Slot 0 only, `-np 1` for the swap.** Other slots use the cache and feed no bank. The swap is
  refused with more than one slot, because it would move the arena under a request that is being
  served on another slot.
- **A prompt served from the prompt cache profiles nothing**, by construction: the tokens are not
  processed, so the interval is empty and discarded. A server with a warm prompt cache therefore
  builds its prefill history more slowly than its decode history.
- **The boundary is prompt processing against generation, nothing finer.** No thinking/body split,
  no language heuristic, no client cooperation. Splitting the generation phase further would be one
  bank more and is not done here.
- **A request that generates nothing produces no decode record**, and only the profiled slot feeds
  the banks.
- Everything the expert cache is limited by still applies: HIP only, host-direct required,
  `--load-mode none`, one model per process.

## Design notes

- **Both banks are profiled whether or not the swap is on.** The alternative, profiling `prefill`
  only when the swap is on, saves one commit per request and makes turning the option on later start
  from nothing. The commit is a 96 KiB read-back; the cold start is a real cost to the user.
- **No idle timer.** See "Why it exists". The only idle notion left is "no compute in flight", used
  for the deferred install.
- **The policy is driven through a table of function pointers.** The policy object calls the backend
  through `common_expert_backend`, which defaults to the `llama_expert_*` functions. The production
  call sites do not mention it; `tests/test-server-expert.cpp` substitutes a fake backend that
  records every call, so the whole policy is tested without a GPU and without a model against the
  call sequences written out in the test.
- **The server logs the plan and the reason, the backend logs the cost.** An install prints one
  server line saying which plan of which bank goes in and why, and the backend's line right after it
  has the slices, MiB and ms. Returning the install statistics through the llama API would have been
  an ABI change for two adjacent log lines.
- **The interrupted-request discard is wired to the decode error path only.** A client abort is a
  normal request end and its tokens are a record; a failed decode discards the open interval, because
  the request produced no meaningful selection history.

## Revision

Every number above comes from the binary built from the commits that add this feature on ranma
`ccd8fd1e9` (upstream llama.cpp `093a2f86c`, plus the fork's earlier commits). The exact commits
are kept on the dated snapshot branch (`ranma_YYYYMMDD`) pushed together with `ranma`; that branch
is never rebased.
