# Server context checkpoints

Changes to how `llama-server` keeps the context checkpoints of a slot.

## Reuse of a just-restored checkpoint

### What it is

When a prompt restores a context checkpoint and its first batch starts at the
checkpoint position, the memory still holds exactly the state of that
checkpoint. The server used to serialize a new checkpoint at that position
from the unchanged state and let it supersede the entry that had just been
loaded. It now keeps the restored entry instead and moves it to the back of the
list, where the re-created checkpoint would have been appended, so the list
order, the task ids and the positions are the same as before; only the
serialization is skipped.

### When it applies

- Only for the first prompt batch after the restore. The restored entry is
  remembered for the task that restored it and forgotten once that batch has
  been built, or when a new prompt starts.
- The entry must have survived the evictions that make room for the new
  checkpoint and must cover the same token count and the same positions.
- It must hold target data, draft data exactly when a draft context exists, and
  a speculative state equal to the current one. The speculative state is small,
  so it is compared rather than assumed to round-trip.

In every other case the checkpoint is created as before.

### Options

| Name | Kind | Default | Effect |
| --- | --- | --- | --- |
| `LLAMA_SERVER_CKPT_REUSE` | env | `1` | `1` keeps the restored checkpoint; any other value, e.g. `0`, re-creates it from the memory state every time, as upstream does. Read once when the server loads its model. |

### How to verify it

At trace verbosity the server logs `reusing restored context checkpoint ...`
where it logged `superseding context checkpoint ...` before. With
`LLAMA_SERVER_CKPT_REUSE=0` the old line returns.

## Refused suffix removal

The server no longer aborts when a memory refuses a suffix removal. After a
checkpoint restore it falls back to the next older context checkpoint, as when
the newest one does not fit, and reprocesses the prompt from the start only
when none is left (one warning line per step); during a speculative rollback it
ends the request with an error and clears the slot. When the memory of a
standalone draft model refuses the removal of a draft, the server rebuilds the
draft sequence from the newest prompt checkpoint whose removal it accepts, else
from the start; a draft fed by target hidden states, a prompt with media or with
shifted positions ends the request instead.
