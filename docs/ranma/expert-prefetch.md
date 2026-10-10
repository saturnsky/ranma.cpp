# Expert prefetch into the decode pool (experimental)

`RANMA_EXPERT_PREFETCH=1` copies the experts a layer is predicted to route to into the expert cache's decode pool
([expert-cache-ub-pool.md](expert-cache-ub-pool.md)) before the layer runs, with the copy engine, while the layers
before it compute. Off by default. It needs the decode pool (on by default, not `RANMA_EXPERT_UB_POOL=0`), a model that builds the
prediction (GLM-5.3 Flash, `glm5-next`) and decode graphs of at most 8 rows. With it on, the SSD tier's unified host
policy is switched off by its gate ([expert-cache-l2.md](expert-cache-l2.md)).

## How it works

- **Prediction.** For the layers the cache asks for, the graph of layer `il` ends with a prediction of layer `il+1`'s
  experts from its output: the hyper-connection pre-mix and `ffn_norm` of the next FFN block applied to the output, the
  next router, and an argsort of the scores (`ffn_moe_pred-<il+1>`).
- **Publish.** A kernel on the compute stream reserves positions of the next layer at the tail of the pool's LRU for
  the best predicted experts that are admissible and not pooled, and lists them in a mailbox in mapped host memory.
- **Copy.** A host thread copies each listed expert from its host home into its position with `hipMemcpyAsync` on its
  own stream: up and gate first, then down, each followed by a `hipStreamWriteValue32` of a done word.
- **Settle.** The step of the layer settles the finished copies into owned positions without a copy; a settle kernel
  before the down matmul publishes a down part that finished after the step. A copy that has not finished stays
  pending and is not evicted by an ordinary fill. The address tables only ever point at a position or at a home, so
  the output does not depend on the timing.
- **Budget.** The number of copies per layer and publish adapts per layer and row count (1, 2-3, 4+ rows) from the
  measured window between the publish and the step: a cold start of `COLD` publishes without copies, then the budget
  whose up and gate fit the 10th percentile window under `START_US + b * bytes / BW_GBPS`, lowered when fewer than
  90 % of the last copies arrived in time and raised at 98 % when one more fits.
- **Predicted layers.** Per row bucket the layers that predict are none, the MLA layers or every pool layer, kept at
  least `HOLD` graphs, with every layer predicting again every `PROBE` graphs. The profile is part of the graph reuse
  key, so a change picks a stored graph instead of building one.
- Before a reclaim of the pool the thread stops issuing and the compute stream waits for its copies.

## Switches

All are read once when the decode pool starts.

| Environment variable | Default | Effect |
|---|---|---|
| `RANMA_EXPERT_PREFETCH` | 0 | 1 turns the prefetch on (with the decode pool). |
| `RANMA_EXPERT_PREFETCH_MODE` | `inplace` | `inplace` copies into reserved pool positions; `fill` copies into staging slots outside the pool and lets the step's fill copy from them. |
| `RANMA_EXPERT_PREFETCH_SLOTS` | 4 | Most copies per layer and publish (1..8). |
| `RANMA_EXPERT_PREFETCH_RING` | 4 | `fill`: staging sets (2..16). |
| `RANMA_EXPERT_PREFETCH_K` | 8 | Predicted experts per row taken from the argsort (1..16). |
| `RANMA_EXPERT_PREFETCH_COLD` | 8 | Publishes without copies at the start (window samples only). |
| `RANMA_EXPERT_PREFETCH_WINDOW` | 32 | Publishes between budget changes (8..64). |
| `RANMA_EXPERT_PREFETCH_BW_GBPS` | 55 | Copy bandwidth of the transfer model. |
| `RANMA_EXPERT_PREFETCH_START_US` | 70 | Start latency of the transfer model. |
| `RANMA_EXPERT_PREFETCH_MARGIN_US` | 10 | Time kept free in the window. |
| `RANMA_EXPERT_PREFETCH_HOLD` | 512 | Graphs a set of predicting layers is kept (at least 16). |
| `RANMA_EXPERT_PREFETCH_PROBE` | 4096 | Graphs between rounds in which every layer predicts again. |
| `RANMA_EXPERT_PREFETCH_SKIP` | 1 | 0: layers whose budget is 0 keep predicting. |
| `RANMA_EXPERT_PREFETCH_LOG` | 0 | 1: the budgets per layer at the end. |
| `RANMA_EXPERT_PREFETCH_TEST` | 0 | Fault injection for correctness checks: 1 = a done word that never matches, 2 = a wrong prediction (the lowest ranked candidates), 3 = copies that arrive late. The output must stay the same as with the prefetch off; the speed does not matter in these modes. |

`RANMA_EXPERT_PREFETCH_TEST` is how the prefetch is checked: a run in each test mode must give the output of a run
with the prefetch off, which shows that a missing, wrong or late copy only costs time.

## Limits

- Only GLM-5.3 Flash builds the prediction; other models run with the pool alone.
- Copies compete with the decode for the host-to-device link; the adaptive budget limits them to what fits the window.
- Experimental: off by default, and not part of the measured release configuration.
