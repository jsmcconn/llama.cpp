# Shared checkpoints for a single Qwen Flash Next user

Use a fixed `llama_user_id` in every request to pool checkpoints across agent
sessions. A proxy can set this field for all model aliases. This deliberately
shares one owner's cache; ordinary server requests with different owners remain
isolated. Prompt tokens still have to match exactly. A new session does not
inherit another conversation's messages.

For the Qwen3.8 Flash Next shared-MTP workstation configuration:

```sh
--cache-ssd /path/to/quant-specific-cache
--cache-ssd-checkpoints 1
--cache-ssd-hot-ram 1024
--cache-ssd-warm-ram 512
--cache-ssd-prefix-checkpoints 3
--cache-ssd-max-cold 32
--cache-ssd-max-conversations 8
--cache-ssd-cold-maxsize 262144
--cache-ssd-durable-min-growth 4096
--cache-ssd-durable-max-age 600
--cache-ssd-system-prompts 0
```

`--cache-ssd-prefix-checkpoints` is opt-in (0..3, default 0). It persists the
boundary before the **first user message**, including the rendered system/tool
preamble, when that prefix is at least 1024 tokens and target/draft positions
agree. It uses full target, draft, and speculative state. The separate
target-only system-prompt cache cannot restore this recurrent model safely.

New anchors bypass the usual near-end and durable-growth suppression; exact
duplicates avoid full serialization. Three bounded retention hints occupy the
previously reserved words in the v4 index header. Existing checkpoint record
and state formats are unchanged; older v4 readers ignore the hints. Cold-count
and global disk eviction prefer ordinary conversation checkpoints over anchors,
but anchors can still be evicted to meet the caps. These hints do not reserve
additional RAM. The 256 GiB disk cap counts every active checkpoint namespace;
`max-cold` counts cold checkpoints per namespace, excluding resident hot/warm
entries. `cache-ssd-checkpoints` sizes the page manager's latest-per-slot map.

Lookup prefers the longest completely matching saved prefix before recency.
With prefix retention enabled, a warm text-only recurrent slot also probes SSD
within its current owner when a checkpoint can save at least 1024 additional
tokens over its reusable RAM state. It clears the old slot only after finding
a full-hash candidate, then uses normal cold restore. A failed disk load falls
back to prefill. Partial recurrent restores and exact-extent restores without
a real suffix remain disallowed. Media and aLoRA use their existing paths.

Closing an agent or restarting the backend retains durable SSD checkpoints.
Changing the system prompt, tool schema/order, template, or earlier tokens may
prevent reuse; neither recurrent state nor cache contents can be trimmed to
invent a boundary. Model loading after a backend unload is separate from prompt
cache latency. Keep different quantizations in different cache directories.
