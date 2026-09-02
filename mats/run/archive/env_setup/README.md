# Environment setup — rescued from the VM home directory

These lived in `~` on the box, **not** on the detachable storage, so they would
have died with the instance. They are the environment bring-up and the
inference-engine benchmarking from the first session (2026-09-01), and they
record how the venv was actually built.

| file | what |
|---|---|
| `freeze.txt` | `pip freeze` of the working venv — **215 pinned versions**, incl. vllm `0.28.1rc1.dev248+g55178f2d0`, torch `2.13.0+cu132`, transformers `5.16.1` |
| `install_vllm.sh` | how vLLM was installed: the nightly wheel index, `UV_CACHE_DIR` on the shared mount |
| `verify_cfg.py` | confirms `Qwen3.5-4B` is `Qwen3_5ForConditionalGeneration` with the text stack under `config.text_config` |
| `probe_args.py` | which `EngineArgs` fields this vLLM build actually accepts — several plan-assumed flags do not exist |
| `try_engine.py` | the engine variant harness (baseline, prefix caching, fp8 KV, chunked prefill, MTP spec-decode) |
| `run_engine_sweep.sh` | drives `try_engine.py` across all seven variants |
| `query.sh` | a curl smoke test against a served endpoint |

## Why these matter for reproduction

**Never pin torch.** vLLM resolves it — the plan pinned `torch 2.6.*`, which
would have failed. `freeze.txt` is the record of what actually worked together.

**PyPI stable vLLM cannot load `qwen3_5`.** The nightly index in
`install_vllm.sh` is required, not a preference.

**flash_attn is absent and was never needed for inference.** For *training* the
prebuilt `kernels-community/flash-attn2` is used instead, via the `kernels`
package — no compiler.

The measurements these produced are in `results/FINDINGS.md` (2026-09-01):
throughput 7,745–8,642 output tok/s, prefix caching worth 1.02× rather than the
plan's claimed 2×, MTP speculative decoding 1.07× and inside noise, and the
trace-length distribution that set `max_tokens=32768`.
