# Models and Internet sources

The Samosa app can run any of these local backends:

- **Bonsai 27B 1-bit** through Prism's Metal-enabled `llama-server`
- **Ornith 9B** (DeepReinforce Ornith-1.0-9B, Q4_K_M GGUF) through the same `llama-server`
- **Qwen3.6 35B A3B** through Samosa's streaming expert engine

Choose the model under **Settings → Model**. Samosa unloads the old backend
before starting the new one, so both models are never resident at once. A switch
starts a new conversation because the models use different tokenizers and
conversation-state formats. Bonsai is text-only unless a compatible vision
projector is installed; Qwen retains the app's existing image support.

The gateway discovers Bonsai at:

```text
~/.samosa/backends/prism-llama.cpp/build/bin/llama-server
~/.samosa/models/bonsai-27b-1bit/Bonsai-27B-Q1_0.gguf
```

These can be overridden with `SAMOSA_BONSAI_SERVER` and
`SAMOSA_BONSAI_MODEL`.

Ornith is discovered at `~/.samosa/models/ornith-9b/Ornith-1.0-9B-Q4_K_M.gguf`
(override with `SAMOSA_ORNITH_MODEL`); it runs through the same Prism
`llama-server` binary as Bonsai. Ornith is a reasoning model — with thinking
on, it can spend most of a small token budget inside its reasoning trace, so
raise the max-tokens setting or switch thinking off for short answers. It is
text-only as installed.

## Context capacity and compaction

All three backends expose the same controls in **Settings → Total context
capacity** and **Auto-compact conversations**. Qwen uses its hardware-aware
context policy. The GGUF gateway keeps its conservative 8,192-token default;
`SAMOSA_GGUF_CONTEXT_TOKENS` can change that startup default, and an explicit
value selected in the app can use any capacity up to the model-declared
262,144-token maximum. Applying a different GGUF capacity restarts only the
active `llama-server`; the gateway and saved conversations stay in place.
Whether a large explicit window fits and performs well still depends on the
model, quantization, and machine.

Bonsai and Ornith conversation state is stored separately under:

```text
~/.samosa/chats/<conversation-id>/bonsai.json
~/.samosa/chats/<conversation-id>/ornith.json
```

These are model-facing ledgers; the complete visible transcript remains in the
browser. The gateway renders each ledger with the active model's own chat
template and calls that model's tokenizer for exact usage. At the configured
threshold (80% by default), it asks the active model to summarize older turns,
keeps a recent message-aligned tail verbatim, and atomically replaces the
ledger. The next inference sends the smaller prompt, so `llama-server` discards
the old slot prefix and rebuilds K/V state. The conversation ID does not
change, and the compacted ledger is reloaded after a gateway or model restart.

**Compact this conversation now** uses the same operation. A failed summary,
non-shrinking result, oversized result, or file-write failure leaves the
previous ledger authoritative. Gateway policy is stored in
`~/.samosa/gateway-settings.json`.

## Internet sources

Paste a public `http://` or `https://` URL under **Settings → Internet source**
to read it and add cleaned text to the next prompt. Fetches are user-initiated;
the model cannot make network requests. Samosa blocks private, loopback,
link-local, transition, multicast, and reserved addresses, pins DNS results,
revalidates redirects, limits responses to 5 MB, and times out after 20 seconds.

The **model decides** when to use the Internet. The gateway tells every model,
in the system prompt, that it has two abilities — `web_search` (search the
public web) and `open_url` (read one public page) — and the model requests one
by replying with a single JSON line. The gateway runs the tool, feeds the
output back, and the model answers from it (up to 3 tool calls per turn; tool
activity is shown in the thinking area). There is no keyword trigger; asking
"find an IMAX theater near Clemson" works the same as "search for …". The
gateway also adds the host's current local date to every chat.

## Connecting a search service

Without configuration, `web_search` uses DuckDuckGo's keyless HTML endpoint
with Bing's RSS as fallback. Those need no account but are **location-blind
and low quality** for local or time-sensitive questions. To give your local
models real search, connect a service you have credentials for in
`~/.samosa/config.json`. Samosa never ships a shared API key, and credentials
never leave your machine except to the service you configured.

Presets exist for `brave`, `tavily`, `serpapi`, `google` (Programmable
Search), and `searxng` — name one and supply only its credentials:

```json
{
  "search": {
    "provider": "brave",
    "providers": {
      "brave":   { "api_key": "YOUR_BRAVE_KEY" },
      "tavily":  { "api_key": "YOUR_TAVILY_KEY" },
      "serpapi": { "api_key": "YOUR_SERPAPI_KEY" },
      "google":  { "api_key": "YOUR_GOOGLE_KEY", "cx": "YOUR_ENGINE_ID" },
      "searxng": { "base_url": "https://your-searxng.example" }
    }
  }
}
```

**Any other HTTP JSON search API** can be described declaratively — no code
changes. `{query}` is the URL-encoded search text; every other `{name}`
placeholder resolves from the provider's own config values; `results` is a
dot-path to the result array in the response; `fields` maps title/url/
description within one result. A `body` object makes the request a POST.

```json
{
  "search": {
    "provider": "my-service",
    "providers": {
      "my-service": {
        "url": "https://api.example.com/v2/search?q={query}",
        "headers": { "Authorization": "Bearer {api_key}" },
        "api_key": "YOUR_KEY",
        "results": "data.hits",
        "fields": { "title": "name", "url": "link", "description": "summary" }
      }
    }
  }
}
```

If a configured provider fails (bad key, quota, outage), Samosa logs it and
falls back to the keyless path; set `"fallback": false` under `search` to
surface the error to the model instead. The preset request/response shapes
follow each service's published API; only providers you have keys for can be
verified on your machine — the generic executor itself is exercised by
`make test` and was verified live against a config-defined provider.

Set `SAMOSA_OFFLINE=1` before starting the app, or set `"offline": true` in
the config file, to disable every outbound Internet path.
