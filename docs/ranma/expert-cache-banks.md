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

**Which bank is the prompt bank.** `bank_open` takes a `prompt_bank` flag. The policy opens `decode`
with the flag clear and `prefill` with it set.

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

## What it costs

- **Two installs per request instead of one**, when the swap is on. The prompt-to-generation install
  is inside the request, so it is added to the time to the first token.
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
