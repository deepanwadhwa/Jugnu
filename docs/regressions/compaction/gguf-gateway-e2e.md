# Bonsai and Ornith gateway compaction — real-model regression

Date: 2026-07-19  
Branch: `models-gateway-compaction`  
Result: **PASS on the named machine and models below**

This is a real gateway → Prism `llama-server` → GGUF model run. It is not a
mock, build-only check, or direct-backend shortcut.

## Host and model identity

```text
Darwin 25.5.0 arm64 (macOS 26.5.1 build 25F80)
Apple M3 MacBook Air
physical memory: 17179869184 bytes (16 GiB)
llama-server: version 9596 (9fcaed76), AppleClang 21, Darwin arm64

Bonsai-27B-Q1_0.gguf
  3803452480 bytes
  sha256 17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0

Ornith-1.0-9B-Q4_K_M.gguf
  5701067872 bytes
  sha256 641675b356a4463677a02a0d703f7b571b39f0b747fbd270b379c848edc29e84
```

## Method

The gateway ran on `127.0.0.1:18742`, with its backend on `:18743`, an isolated
temporary `SAMOSA_HOME`, and a deliberately reduced 2,048-token context. This
small window forces the real automatic-compaction path without allocating a
large K/V cache on the 16 GiB reference machine:

```sh
SAMOSA_HOME=/tmp/samosa-gateway-e2e.<run> \
SAMOSA_PORT=18742 SAMOSA_BACKEND_PORT=18743 \
SAMOSA_GGUF_CONTEXT_TOKENS=2048 \
python3 tools/samosa_gateway.py

curl http://127.0.0.1:18742/v1/settings \
  -H 'Content-Type: application/json' \
  --data-binary \
  '{"context_tokens":"auto","auto_compact":true,"compact_threshold_percent":50}'
```

For each model, one conversation supplied five exact facts inside a long
multi-turn planning transcript and ended with a request to reply `READY`.
Requests used `temperature: 0`, thinking disabled, and a 48-token answer
ceiling. The 50% threshold forced automatic compaction. The gateway was then
stopped through `/v1/shutdown`, restarted from the same isolated home, and
given only a new recall question under the same `conversation_id`.

An initial Bonsai fixture was correctly rejected because its 2,703-token
summary request could not fit the artificial 2,048-token test window. The
fixture was reduced before the passing run; no truncation was accepted.

## Bonsai result

Automatic compaction and the model's response:

```json
{
  "elapsed_seconds": 35.54,
  "content": "READY",
  "samosa": {
    "compacted": true,
    "compacted_from_tokens": 1075,
    "compacted_to_tokens": 367,
    "retained_recent_tokens": 244
  },
  "usage": {
    "completion_tokens": 2,
    "prompt_tokens": 655,
    "total_tokens": 657
  }
}
```

The atomic ledger contained five messages after the response: continuation
memory, a retained user/assistant tail, the trigger question, and `READY`.
After complete gateway/backend restart, a request containing only the recall
question returned:

```text
Vault Access Code: EMBER-4417
Project Name: Heliotrope Lantern
Delivery City: Valparaiso, Chile
Finish Preference: Matte teal (explicit no-gloss constraint)
Audit Checksum: 7QX-19-MARBLE
```

All five required strings matched.

## Ornith result

Automatic compaction and the model's response:

```json
{
  "elapsed_seconds": 14.53,
  "content": "READY",
  "samosa": {
    "compacted": true,
    "compacted_from_tokens": 1071,
    "compacted_to_tokens": 362,
    "retained_recent_tokens": 242
  },
  "usage": {
    "completion_tokens": 2,
    "prompt_tokens": 650,
    "total_tokens": 652
  }
}
```

After complete gateway/backend restart, a request containing only the recall
question returned:

```text
Observatory key: NOVA-8821
Expedition name: Silver Kestrel
Rendezvous city: Sapporo, Japan
Material preference: Brushed copper (chrome explicitly excluded)
Manifest checksum: 4LM-72-CEDAR
```

All five required strings matched.

## Browser streaming path

The app uses SSE rather than the non-streaming response above, so the restarted
Ornith conversation received this additional streaming turn:

```text
User: Add one new exact detail ... the secondary beacon is BLUE-603.
Assistant: The secondary beacon has been added to the record as BLUE-603.
```

The collected stream and the final ledger contained the identical complete
assistant sentence:

```json
{
  "saved_last_role": "assistant",
  "saved_last_content":
    "The secondary beacon has been added to the record as BLUE-603."
}
```

This checks that streaming persistence saves all chunks, not only the buffered
prefix used to distinguish normal text from a gateway tool call.

## Machine safety

Before model load, after Bonsai recall, after Ornith recall, and after final
shutdown:

```text
vm.swapusage: total = 0.00M  used = 0.00M  free = 0.00M
No thermal warning level has been recorded
No performance warning level has been recorded
```

The isolated gateway and both model-server processes were stopped after the
test. These results qualify this exact 2K forced-compaction workload on the
named 16 GiB M3 machine; they do not claim that every model-authored summary
will preserve every detail or that a 262K window fits this hardware.
