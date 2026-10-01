# Smart draft length for draft-mtp (`--spec-smart`)

## What it is

In llama-server, `draft-mtp` chooses the draft length of every round itself, from what the server measures while it
runs (`--spec-smart`, on by default). It needs no per-model thresholds: the same settings fit a model whose
verification cost rises steeply with the batch width (MoE experts read from host memory) and one where it is nearly
flat.

| Option | Env | Default | Meaning |
| --- | --- | --- | --- |
| `--spec-smart`, `--no-spec-smart` | `LLAMA_ARG_SPEC_SMART` | on | Choose the draft length at every step. `--no-spec-smart` uses the single-argument rule instead (see below). |
| `--spec-smart-store PATH` | `LLAMA_ARG_SPEC_SMART_STORE` | none | File that keeps the estimates across restarts. Without it every start is cold and nothing is kept. |
| `--spec-smart-half-life N` | `LLAMA_ARG_SPEC_SMART_HALF_LIFE` | 250 | The confidence of every estimate but the level halves every N rounds of the server, whether the estimate is observed or not. |
| `--spec-smart-log PATH` | `LLAMA_ARG_SPEC_SMART_LOG` | none | Append one JSON line per round to PATH for diagnosis (see "Round log"). The decisions do not change. |

`--spec-draft-n-max` is the upper bound of the draft length. Without it the bound is 7 (the default of 3 belongs to
the single-argument rule and is unchanged). The bound also sizes the target batch, the recurrent state rollback and
the expert cache rows, so a model that gains from longer drafts gets a higher `--spec-draft-n-max`; it never turns the
controller off.

It applies to `draft-mtp` only. With another speculative type it is off (an explicit `--spec-smart` logs a warning).

### The single-argument rule

Before this controller, the draft length of `draft-mtp` came from fixed thresholds
([spec-draft-thresholds.md](spec-draft-thresholds.md)): `--spec-draft-p-min`, `--spec-draft-p-continue` and
`--spec-draft-n-max` (default 3). Such values are tuned for one model, one draft head and one machine. To use them:

- `--no-spec-smart` (or `LLAMA_ARG_SPEC_SMART=0`): the controller is off, the thresholds and `--spec-draft-n-max` apply
  as given;
- giving `--spec-draft-p-min` or `--spec-draft-p-continue` (on the command line or in the environment, any value, `0`
  included) also turns the controller off, with a warning at startup:
  `--spec-smart is off because --spec-draft-p-min or --spec-draft-p-continue was given ...`. This holds also when
  `--spec-smart` is given at the same time: a threshold given by the user wins.

### Fixed constants

The controller has a few more constants. They are fixed, not options (the values below are those of every measurement
so far): the half-life of the level 64 rounds, the extension threshold 4 observations and its interval 64 rounds, the
cap of the wait confidence 16 observations and the lookahead mix 0.5. The text below calls them by these values.

## How a round is chosen

A round is: draft k tokens, verify them in one target batch of k + 1 rows, keep the accepted prefix, roll back the rest,
and let the draft head catch up. Before the first draft step and after each step the server compares two expected
throughputs:

    throughput = (expected accepted tokens + 1) / (draft time so far + predicted verification time of the width)

- **stop now**: the best prefix of the tokens drafted so far, 0 tokens included (a round can skip drafting, or drop a
  weak tail);
- **one more step**: then the best stopping depth, with a predicted acceptance of each position j not drafted yet:
  `A * q(last drafted token) + (1 - A) * qbar(j)`, A = 0.5 (the lookahead mix). Before the first step there is no
  drafted token and the prediction is qbar(j) alone.

It drafts one more step while the second is larger, up to the bound and up to the widest width found so
far (see "Extension"). The expected accepted tokens of a prefix are the sum of the products of the calibrated
acceptances q of its tokens.

## Extension

A width is measured only in rounds that verify it. There is no forced round: no cold sweep over all widths and no
round forced to a width the decision did not reach.

- **Waiting widths.** A width whose wait confidence is below 4 observations (the extension threshold) waits. A
  width never observed (nothing stored, no time measured) has confidence 0 and waits.
- **Wait confidence.** The shape of every width keeps a second confidence next to that of its mean. It takes the same
  observations and decays with the same half-life, but it is at most 16 observations (the cap) after every
  observation. It decides the wait and nothing else: the mean, its confidence and variance and the level do not see
  the cap. So a width observed hundreds of times and then left waits again after about log2(16 / 4) = 2 half-lives
  (500 rounds at half-life 250), however much it was used; without the cap the confidence of its mean would have to
  decay from its full size first. After one extension the wait confidence is about 5, so the width waits again after
  log2(5 / 4) = 0.3 half-lives (about 80 rounds at half-life 250) until the decisions take it or its confidence grows.
  A width in use keeps it at 16 and never waits.
- **Observing a waiting width.** When a width that waits is observed (in an extended round or because a decision
  took it), the confidence of its old mean is first cut to its wait confidence (below 4), and then the observation
  goes in as usual. The new time then weighs at least 1 / 5, so a few extensions move a mean that stayed high long
  after the width was left; without the cut one observation moves it by 1 / (confidence of the mean + 1), which can
  be 1 / 30 and more. The variance is kept (a weighted variance does not change when all old weights shrink alike).
  The level takes that observation as before (the cut comes after it). A width in use never waits and is never cut;
  the level, the acceptance cells and the draft step are not cut.
- **One width more.** When the decision of a round stops and keeps k tokens, and width k + 1 waits, the round verifies
  k + 1 instead: it drafts one more step (or keeps the token it would have dropped, when it drafted more than k) and
  verifies all k + 1. The observation of width k + 1 then goes in as in any round. Otherwise the round verifies k.
  Only width k + 1 is looked at; a waiting width two or more widths beyond the decision is not extended in that
  round, and a round is extended by one width at most. A decision that drafts nothing (k = 0) extends to width 1.
- **Interval.** At most one extended round every 64 closed rounds, counted from the start of the process; only widths
  up to the bound (`--spec-draft-n-max`, else 7) and the limit of the request.
- **Frontier.** A decision does not draft beyond the widest width w for which every width 1..w was verified by this
  process or has a shape (stored or observed). The width after it is reached only by an extension of a round that
  stopped at w. So a cold start begins at width 0 (no draft) and widens one width per extension, as far as the
  decisions go; widths no decision reaches are never verified. With a full store the frontier is the bound from the
  start.
- **First use.** The first verification of every width in a process builds its graphs; its time is not observed (in
  a cold and in a warm start). A width the process has verified once is inside the frontier, so the decision can take
  it (its predicted time until then comes from its neighbours, below) and the next verification is timed.

## What it learns

All estimates are means with a confidence (a number of virtual observations). The clock is the number of closed
rounds of the server: the confidence of every cell halves every 250 rounds (`--spec-smart-half-life`; the level: 64
rounds, fixed), also when the cell gets no observation. Time lowers only the confidence; a mean changes only with a new observation, which then
weighs 1 / (decayed confidence + 1) (for a waiting width: 1 / (wait confidence + 1), see "Extension"). The decay is
computed when a cell is read or updated, not every round.

- **Acceptance.** One cell per draft position group (0, 1, 2 and later) and per 0.05 bin of the draft top
  probability. Only verified positions are observed: the accepted ones and the first rejected one; positions after a
  rejection are not. Each cell starts at q = p with a weight of 2 observations. As the confidence of a cell decays,
  its acceptance moves back to this starting value.
- **Mean acceptance per position** (qbar), for the positions not drafted yet: one cell per draft position 0..bound (at
  most 16 cells; the last one also takes the positions beyond), observed like the acceptance cells, starting at 0.5
  with a weight of 2 observations. Since a position is observed only when all positions before it were accepted, it is
  the acceptance given that the prefix was accepted. With some drafters it falls fast with depth (for example
  0.73 / 0.47 / 0.29 / 0.22 / 0.15 at positions 0..4), so positions 2 and later are not pooled.
- **The last drafted token predicts the next ones.** With some drafters the next position is accepted more often
  after a confident token (0.78 after p >= 0.99, 0.51 after p < 0.3, where the mean was 0.655). The lookahead
  therefore mixes the calibrated acceptance of the last drafted token into the position mean (see above). A drafter
  whose probabilities do not carry this signal loses little: the mix only moves the prediction halfway to a
  calibrated value.
- **Request boundary.** When a request starts, the confidence of the acceptance cells is capped at 16 observations and
  their values are kept, so that a change of task (roleplay to code) moves the estimates quickly.
- **Verification time per width.** The time from the end of a draft to the next draft of the same sequence: the target
  batch, the draft catch-up, sampling, a checkpoint replay if one happened, and the per-token server work. A round
  whose request ended is not timed (the gap to the next request is not verification), nor the first verification of
  a width in the process (graph builds).
- **Level x shape.** The predicted time of width k is `L * S_k`. The level L is one number for all widths; the shape
  S_k is the ratio of width k. A change of task changes the experts in use, and often makes every width slower or
  faster by about the same factor (a change of level), while the ratios of the widths change less. So the level
  follows fast (half-life 64 rounds) and the shape slowly (half-life 250 rounds), and a width that is not in use
  follows a change of level through L.
  - An observation x of width k: L takes `x / S_k` and then S_k takes `x / L`, both with the values from before this
    observation. The first observation of the process (without a stored level) sets L = x.
  - A width without a shape (never observed, nothing stored) is not used for L; its first observation sets
    `S_k = x / L`.
  - L takes only the observations of widths whose shape has a confidence of at least 4 (the extension threshold),
    so that the noise of a width with few observations does not move it. When no width has that confidence (a cold
    start), every width with a shape moves L. In that state one observation moves L and S_k together, so the product
    can move a little more than the observation (for example 30 ms then 36 ms on one width give 36.3 ms, not 33 ms);
    it settles as the confidences grow.
- **Value of a shape.** The decision uses `L * S_k` with the mean of the shape. Every shape cell (and the draft step)
  also keeps the weighted variance of its observations, with the same decayed weights (weighted Welford; time does not
  change it); the store keeps it and the request summary prints that of the draft step, the decision does not use it.
- **Widths without a shape** take the line through the two nearest widths that have one (in shape, before L).
- A width not measured by this process is never predicted cheaper than the nearest narrower width that was (in
  shape, before L).
- **Stored values** (from the store file) are scaled by the level of this process like the others, up or down.
- **Extension** (above). A width measured much too slow once might not be chosen again. While its wait confidence is
  below 4 it waits, and a round that stops one width before it verifies it. A width that is chosen often never needs
  one; the width after the one in use is verified again once its wait confidence (capped at 16, see "Extension")
  decays below the threshold. The level needs no extension.
- **Draft step time**, one decayed mean, measured anew by every process.

There is no regression over the traffic or position of a batch: each width has its own ratio.

## The store

`--spec-smart-store PATH` is a cache. It holds the level, the shape per width (mean and variance), the acceptance cells and the
mean acceptance per position, each
with its confidence decayed to the time of the save, and is written every 256 rounds and at exit (through a temporary file and
a rename). It is read at
startup only when its key matches exactly: the target and draft model files (path, size, time), the build (build
number, commit, and size and time of the executable and of the llama / ggml libraries next to it) and the expert
cache options that change the cost of a width (`--expert-l1-mib`, `--expert-l2-mib`, `--expert-cache-mode`,
`--expert-cache-draft`, the joint cache weights, `--spec-draft-ngl`). Another key means a cold start; values are never
mixed between models. The stored means and variances are restored exactly and their confidence is capped at 32
observations; the wait confidence of a shape starts at that confidence capped at 16 (the store keeps only the
confidence of the mean); the level is restored with a confidence of at most 2, so the first observations of the new process
move it. The clock of the new process starts at 0. A store of an older format (without the variance, with the
position groups 0, 1, 2+ instead of one mean acceptance per position, or with a time per width instead of level and
shape) is not read: cold start. A store of another bound is read for the widths and positions that both have.

Without `--spec-smart-store`, or without a usable store, the server starts cold: it decides from the first round,
drafts nothing until the first extension (round 64) verifies width 1, and widens one width per extension (see
"Extension"). A warm start follows the same rules; its stored shapes put the frontier at the stored bound at once.

## Log

At startup: `--spec-smart: the draft length is chosen per step, up to N ...` and `spec-smart: n_max = N, warm|cold,
half-life H rounds, level half-life H rounds, extension to a width below T observations at most every E rounds, wait
confidence cap C (0: none), lookahead A q(last) + 1-A qbar(position)`. At the end of every request one line, for example (the
layout only; the numbers are made up):

    spec-smart: rounds 107, mean k 2.17, acceptance 0.634 (147/232), draft step 2.74 ms, extensions 1 (total 3), width share / predicted / observed ms: k0 1.9% 28.99/29.35 k1 61.7% 38.51/43.55 k2 8.4% 51.93/53.70 ..., level 29.10 ms n 91.8, shape (confidence): k0 0.996 (5.1) k1 1.323 (212.4) k2 1.784 (30.8) ..., draft step mean 2.745 sd 0.120 n 640.2, lookahead mix 0.50, qbar: p0 0.731 n 16.0 p1 0.468 n 12.2 p2 0.291 n 5.6 ...

The share is the fraction of the request's rounds with that draft length, the predicted time is the current
estimate of that width, the observed one the mean of this request's timed rounds. `extensions` counts the extended
rounds of the request (and of the process). `level` is L and its decayed confidence; `shape` gives for every width the mean
of S_k (before the line through neighbours and the floor) and its decayed confidence. `draft step` gives the mean,
sd and confidence of the draft step time; the mean is the draft step time above. `qbar` gives the mean acceptance of the first 8 positions (with the prior,
as the lookahead uses it) and its confidence.

## Round log

`--spec-smart-log PATH` appends one JSON object per line to PATH. It only records: with or without it the decisions,
the estimates and the store are the same. The lines are kept in memory and written at the end of every request, at
exit, and whenever 1 MiB is buffered.

- `{"ev":"start", ...}` at startup: the bound, warm or cold, the half-life and the fixed constants (`conf_cap` among
  them).
- `{"ev":"req","req":R,"seq":S,"round":N,"n_closed":C}` when a request starts on sequence S: R counts the requests of
  the process, N the rounds logged so far.
- `{"ev":"round", ...}` for every round, when it ends: at the next draft of the sequence, or, for the last round of a
  request, when the next request starts or the server exits. Fields:

| Field | Meaning |
| --- | --- |
| `req`, `seq`, `round` | request id, sequence, round number of the process (1, 2, ...) |
| `forced` | `none` (the controller decided) or `extend` (the decision stopped at `k_nat` and the round verified `k_nat` + 1). |
| `timed` | false for the first verification of the width in the process (its time is not observed); meaningful for closed rounds |
| `n_max_seq` | the widest width allowed in this round |
| `n_lim` | the limit of the decision: `n_max_seq` or the frontier, the smaller |
| `steps` | draft steps done, the dropped tail and the extra step of an extension included |
| `k` | draft tokens kept and verified |
| `k_nat` | the width the decision stopped at (`k` - 1 in an extended round, else `k`) |
| `ext_step` | true when the extension drafted one more step, false when it kept a token the decision dropped (false in other rounds) |
| `p` | the top probability of every draft step (`steps` values) |
| `q` | the calibrated acceptance q(position, p) of every drafted position that the last decision of the round used (in an extended round: of every draft step) |
| `exp_acc` | the expected accepted tokens of the kept prefix (sum of the products of q) |
| `la` | the last decision (null when there was none, `n_max_seq` 0): `stop` and `cont`, the expected throughputs in tokens / s of stopping with the best prefix and of one more step (null when no step is left), and `q`, the predicted acceptance of the positions not drafted yet (from position `steps` on) |
| `la_next` | for every draft step i that a decision took, the predicted acceptance of position i before it was drafted (not the extra step of an extension) |
| `cost_pred` | the predicted verification time of width k (us) |
| `draft_pred` | the draft step time used by the decision (us) |
| `cost_table` | the predicted verification time of every width 0..n-max (us) |
| `draft_us` | the measured time of the draft steps of the round (us) |
| `verify_us` | the measured time from the end of the draft to the next draft (the verification time, see above); null when the round ended with the request or was not verified |
| `accepted` | draft tokens accepted (0 when k = 0); null when the draft was not verified |
| `wait_reset` | true when the time of this round was the observation of a waiting width, whose old mean was cut to its wait confidence (see "Extension"); false otherwise |
| `wr_n` | only with `wait_reset` true: the confidence of the old mean before the cut and after it (the wait confidence), before the observation |
| `end` | `closed`, `dropped` (the request ended, no verification time) or `unverified` (the slot stopped before the verification) |

In an extended round `la`, `draft_pred` and `cost_table` are those of the decision that stopped at `k_nat`;
`cost_pred` is `cost_table[k]`. The predicted values of a decision are those before the round is observed: the
observation of the round moves the estimates after it is logged.

## Limits

- A server that is killed instead of stopped loses the rounds since the last periodic save.
- One controller per server: with `--parallel` > 1 the times include the batch mates of other slots.
- A change of the ratio of the width in use looks like a change of level at first: L takes it and all widths move
  with it; the extensions move the width after it back, the other widths only when a decision takes them.
- A width that is not chosen gets a new ratio only by an extension, and only when it is the width after the one a
  decision stopped at. A width below the one in use, or two or more beyond it, is never verified again: a wrong
  ratio there stays until a decision takes that width. A change of level reaches it without a verification.
- A width no decision reaches is predicted from its neighbours only (the line through them, and never cheaper than
  the nearest narrower width measured by the process).
- A cold start drafts nothing for the first 64 rounds and widens one width per 64 rounds at most: width 7 (the default
  bound) is found after about 450 rounds when every decision goes that far, later when decisions stop short of the frontier.
- The extended rounds cost throughput when the width after the one in use is truly slow: at most one round in 64,
  and one width more than the decision wanted.
- Off (`--no-spec-smart`, a threshold given, or another speculative type), the draft loop and the server loop run
  the single-argument code: every new hook returns at once.
- The round log keeps one line per round in memory until the end of the request (a few hundred bytes a round); a
  server that is killed loses the lines of the current request. The verification time is not split into the target
  batch and the rest.
- The controller cost per step is a few microseconds (tables of at most n-max + 1 widths); not measured in the server.
