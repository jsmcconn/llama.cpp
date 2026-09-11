# Context checkpoints for Qwen Flash Next

This fork combines upstream Qwen Flash Next support, the Qwen4Exp shared MTP
implementation, and CachyLlama's
persistent conversation cache. The workstation uses one slot. Slot ownership
provides cache affinity; an idle slot can always be reassigned to a new identity.
Compaction often changes the automatic identity because it changes the first
user message. Reassignment clears live state and searches the new owner's RAM
and SSD caches. Both cache tiers scope entries by identity.

## Recurrent state has an exact boundary

A Qwen hybrid checkpoint contains recurrent tensors representing the entire
saved prefix. It cannot be trimmed or rewound by removing attention entries.
`pos_max` is inclusive, so a text checkpoint with N tokens ends at N-1 and resumes
at N. A warm restore requires all N tokens to match the incoming prompt and at
least one genuine suffix token to decode for logits. Exact-length retries use
an earlier matching checkpoint or reprocess the prompt. A middle edit cannot
reuse a later recurrent snapshot, even if its metadata has `pos_min=0`.

`common_prompt_checkpoint::can_resume_recurrent` checks these boundaries.
Checkpoint invalidation removes snapshots beyond the retained prefix, including
deferred finals. Cache safety checks the model architecture, so turning off
bounded recurrent rollback does not accidentally enable partial SSD restores.

## Capture and persistence

`create_checkpoint()` records mid-prompt state before decoding the pending
batch. The token count excludes that batch; position metadata comes from the
live memory. These partial snapshots support warm rollback while attention
entries are still resident.

`deferred_create_final_checkpoint()` normally runs immediately after the first
sampled token is sent, before it is added to the prompt. For recurrent targets
it verifies the live state is at the complete prompt boundary and that only one
token has been sampled. A delayed callback is skipped rather than manufacturing
an earlier recurrent state from position metadata. Exact recurrent snapshots
are copied once without temporarily changing the live target memory.

Checkpoint copying and disk writes run synchronously on the inference loop.
Sending the first token before the final checkpoint can improve first-token
latency; it does not make the checkpoint write asynchronous or free.

Durable SSD checkpoints contain full target and draft state plus speculative
implementation state. This differs from the partial in-memory snapshots. SSD
restore requires a fully matching prefix at an exact saved boundary and a real
suffix to decode. Synthetic RAM metadata after SSD restore also uses inclusive
`pos_max=N-1`. The target-only system-prompt cache is disabled for recurrent
models because it cannot restore the required attention/recurrent/MTP state.

## Capacity and write cadence

Count eviction follows insertion order. Both producers also prune older entries
against `_ckpt_memory_budget()`. Its budget is `max(2 GiB, checkpoint_count *
400 MiB)`; 32 entries give 12.5 GiB. This is a pruning threshold for prior entries,
not an OS memory reservation or a strict allocation ceiling: at least one entry
is retained and the next snapshot is allocated after pruning. For Qwen Flash
Next, partial recurrent snapshots are much smaller than full SSD snapshots.

The separate RAM prompt cache, SSD hot/warm tiers, model weights, PLE table and
in-flight state copies also consume memory. Assess actual resident usage and
available RAM rather than summing configured capacities as committed memory.

The workstation SSD cadence persists when token growth or maximum age is due.
`KV_SSD_STORE_SKIP_CADENCE` defers a write; it does not defer inference. Large
checkpoints that exceed the hot-tier cap remain cold and stream directly into
restore buffers. The filesystem size cap includes loaded and unloaded sessions.

## Validation

- `test-recurrent-checkpoint-boundary`: appended prefixes, edits, retries,
  inclusive metadata, empty checkpoints.
- `test-server-cache-identity`: owner-scoped deduplication, replacement and
  lookup, plus text containers on a media-capable server.
- Existing SSD isolation, prefix, continuation and capacity tests.
- `tools/server/tests/qwen_cache_live.py`: manual model-dependent API checks on
  an isolated server, including automatic identity changes during compaction,
  usage totals, cancellation, tools, vision, warm/cold parity and SSD restart.

The API's OpenAI `prompt_tokens` counts the full request, including cached input.
`prompt_tokens_details.cached_tokens` is a subset. Anthropic separates uncached
`input_tokens` from `cache_read_input_tokens`; add them for full input size.
A client's cumulative uncached-token analytics are not a context-size measure.
