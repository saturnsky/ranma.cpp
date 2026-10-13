# Slot 0 priority (llama-server)

With more than one slot (`-np 2` or more), `llama-server` gives slot 0 priority over the other slots when it
builds a batch. The fork is aimed at one main conversation per server; a second slot serves side work that may
wait.

## What it does

While slot 0 works on a request (from the start of its prompt until its generation ends):

- The prompt tokens of the other slots are not added to the batch. Their prompt processing waits until slot 0 is
  idle. Slot 0's own prompt is batched as before (up to `-b`).
- The tokens the other slots generate are still added to the same batches as slot 0's tokens (prompt or
  generation, including speculative drafts), as before. With `--slot0-priority strict` they wait while slot 0
  generates; they still ride along while slot 0 processes its prompt.

When slot 0 is idle, the other slots are batched as before. The other completions of a request with `n_cmpl > 1`
belong to slot 0's request and are not held. A batch that slot 0 leaves empty (it has nothing to submit in that
step) is filled with the prompt of another slot, so the server never stalls on the rule.

Holding a prompt only delays it: the held slot keeps its state, prompt cache, context checkpoints and drafter
state, and continues from the same position when slot 0 is idle. A batch that is already being computed is not
interrupted, so a new request on slot 0 can wait for at most one batch of another slot.

## Options

`--slot0-priority {off,prefill,strict}` (env `LLAMA_ARG_SLOT0_PRIORITY`, default `prefill`):

| Value | Other slots while slot 0 processes its prompt | Other slots while slot 0 generates |
|---|---|---|
| `off` | batched as upstream | batched as upstream |
| `prefill` | prompts wait, generation shares the batches | prompts wait, generation shares the batches |
| `strict` | prompts wait, generation shares the batches | prompts and generation wait |

With `-np 1` the option changes nothing. A held slot is logged at debug level (`held by slot 0 priority`).

Requests pick their slot as before (`id_slot`, prompt similarity, least recently used); send the main
conversation with `"id_slot": 0` to make sure it gets the priority.
