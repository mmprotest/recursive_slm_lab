# recursive_slm_lab

A runnable demo for persistent memory plus continuous learning using verification-gated recursion.

This repo evaluates four conditions:
1. Baseline (no memory, no learning)
2. Memory only
3. Semantic memory only (mock backend)
4. Memory plus learning (adapters when available)

The system attempts tasks, verifies candidates with unit tests, stores only passing episodes in SQLite, consolidates memory into rules and procedures, and optionally trains LoRA adapters on verified traces.

## What changed and why (short summary)

- Added heldout + hidden splits with statistical gating so promotions rely on robust evidence, not a single noisy mean.
- Added Docker-based verification mode (optional) for safer isolation.
- Added provenance, activation, and rollback for rules/procedures so memory can be audited and reverted safely.
- Added curriculum mining and run manifests to keep iteration traceable and extensible.

## Claim boundaries

This repository is a research-grade demo of recursive self-improvement for code tasks. It is not:
- A production-ready autonomous agent platform.
- A guarantee of continuous improvement without careful evaluation and human oversight.
- A replacement for full security sandboxing or formal verification.

It is intended to show how to enforce evaluation gates, provenance, and rollback for changes that affect behavior.

## Quickstart

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

Visualization support (plots):

```bash
pip install -e ".[viz]"
```

Run the demo:

```bash
bash scripts/run_demo.sh
```

On Windows PowerShell:

```powershell
scripts/run_demo.ps1
```

Artifacts are written under `./artifacts/`.
Each eval run is also stored in the SQLite DB so plots can show progress across iterations.
Set `RSLM_FAST_VERIFY=1` to use a lightweight assert harness instead of pytest when tasks provide `assert_tests`.

## Real mode (Qwen3 4B reasoning)

### A) LocalHF (recommended, enables LoRA learning)

```bash
pip install -e ".[localhf]"
export RSLM_BACKEND=localhf
export RSLM_HF_MODEL_ID=Qwen/Qwen3-4B-Thinking-2507
bash scripts/run_real_demo.sh
```

Notes:
- Requires `transformers>=4.51.0`.
- The first run will download model weights from Hugging Face.
- Qwen3 thinking output is stripped by removing the first `</think>` block and extracting the first python code fence.

RTX 50-series (sm_120) requires PyTorch nightly cu128/cu129. Example:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
```

Quick smoke command (no training):

```bash
rslm eval --db artifacts_real/memory.sqlite --backend localhf --conditions baseline,memory --heldout-size 20
```

### B) OpenAI compatible server (inference only)

Point the backend at an OpenAI-compatible server:

```bash
export RSLM_BACKEND=openai
export RSLM_BASE_URL=http://localhost:8000/v1
export RSLM_MODEL=your-model-name
export RSLM_API_KEY=optional
rslm eval --db artifacts_real/memory.sqlite --backend openai --conditions baseline,memory --heldout-size 20
```

Adapters cannot be applied in this mode, so the "learned" condition is unavailable and will run as baseline.

### Environment variables

- `RSLM_DB`: SQLite path (default: `artifacts/memory.sqlite`)
- `RSLM_BACKEND`: `localhf`, `openai`, or `mock`
- `RSLM_HF_MODEL_ID`: Hugging Face model id for LocalHF
- `RSLM_BASE_URL`: OpenAI-compatible base URL
- `RSLM_MODEL`: OpenAI-compatible model name
- `RSLM_MAX_TOKENS`: generation length
- `RSLM_TEMPERATURE`: sampling temperature
- `RSLM_TOP_P`: nucleus sampling value
- `RSLM_TOP_K`: top-k sampling value
- `RSLM_FAST_VERIFY`: set to `1` to use assert-based fast verification
- `RSLM_STRICT_VERIFY`: set to `1` to apply tighter verifier limits
- `RSLM_VERIFY_WORKERS`: parallel verification workers (defaults to a small CPU-based value)
- `RSLM_VERIFY_MODE`: `local` (default) or `docker` for containerized verification
- `RSLM_INCLUDE_GENERATED_TASKS`: set to `1` to include generated curriculum tasks in training

## CLI overview

```bash
rslm init-db --db artifacts/memory.sqlite
rslm seed-tasks
rslm seed-hidden-tasks
rslm run-iteration --db artifacts/memory.sqlite --tasks bundled --k 4 --mode trainpool --backend mock --memory-enabled --heldout-size 20 --task-limit 30
rslm consolidate --db artifacts/memory.sqlite --min-evidence 1
rslm consolidate-llm --db artifacts/memory.sqlite --backend openai --heldout-size 20 --task-limit 20 --sample-episodes 80 --max-rules 20 --min-gain 0.01
rslm eval --db artifacts/memory.sqlite --backend mock --conditions all --k 1 --heldout-size 20 --task-limit 20 --output artifacts/results_iter001.json
rslm eval-robust --db artifacts/memory.sqlite --backend mock --conditions all --k 1 --heldout-size 20 --task-limit 20
rslm plot --input artifacts/memory.sqlite --output artifacts/results.png
rslm policy-list --db artifacts/memory.sqlite
```

Note: Typer boolean flags use `--memory-enabled/--no-memory` (no extra `true/false` argument).
The CLI also accepts `--memory-enabled true|false` for backward compatibility.

## Generate tasks

Regenerate the bundled task set with deterministic, hidden pytest tests:

```bash
rslm seed-tasks --regen --regen-families --count 200
rslm seed-hidden-tasks --count 40
```

Then run a short iteration and evaluation:

```bash
rslm run-iteration --db artifacts/memory.sqlite --tasks bundled --k 4 --mode trainpool --backend mock --memory-enabled --heldout-size 20 --task-limit 20
rslm eval --db artifacts/memory.sqlite --backend mock --conditions all --k 1 --heldout-size 20 --task-limit 20 --output artifacts/results_iter001.json
```

Task tests are stored only in the verifier and are not exposed to the model prompts.

## Hidden gating & robust eval

Hidden tasks are stored in `recursive_slm_lab/tasks/hidden_tasks.jsonl` and are used only for promotion gating.
Weak hidden tasks (those that allow trivial cheats) are automatically excluded from gating.
Use the robust evaluator to report heldout + hidden splits:

```bash
rslm eval-robust --db artifacts/memory.sqlite --backend mock --conditions all --heldout-size 20 --task-limit 20
```

## Recursive self-improvement (policy loop)

This repo uses a first-class `Policy` layer that controls prompts, retrieval, sampling, and consolidation settings.
Recursive self-improvement means the system proposes policy updates, evaluates them on heldout/regression gates, and
automatically activates a candidate only if it wins without regressions.

Minimal meta-loop demo (mock backend):

```bash
rslm init-db --db artifacts/memory.sqlite
rslm seed-tasks --regen --regen-families --count 200
rslm run-iteration --db artifacts/memory.sqlite --backend mock --memory-enabled --heldout-size 20 --task-limit 20
rslm policy-run-meta-iteration --db artifacts/memory.sqlite --backend mock --heldout-size 20 --regression-size 25 --repeats 3
```

LocalHF meta-loop (with adapters available):

```bash
pip install -e ".[localhf]"
export RSLM_BACKEND=localhf
export RSLM_HF_MODEL_ID=Qwen/Qwen3-4B-Thinking-2507
rslm policy-run-meta-iteration --db artifacts/memory.sqlite --backend localhf --heldout-size 40 --regression-size 25 --repeats 3
```

## Determinism defaults

Promotion gates run deterministically by default (temperature 0, k=1, repeats>=3) to prevent random wins.
Use `--stochastic` and a higher `--repeats` on `rslm eval`, `rslm train-and-promote`, or `rslm policy-run-meta-iteration`
to enable exploratory sampling.

## OpenAI-compatible backend

Set a local server URL and model name:

```bash
export RSLM_BACKEND=openai
export RSLM_BASE_URL=http://localhost:8000/v1
export RSLM_MODEL=your-model-name
export RSLM_API_KEY=optional
rslm run-iteration --db artifacts/memory.sqlite --tasks bundled --k 4 --mode trainpool
```

The client uses the `chat.completions` API and works with OpenAI-compatible servers.

## Local HF LoRA training

Install optional dependencies and provide a local model id:

```bash
pip install -e ".[localhf,train]"
export RSLM_HF_MODEL_ID=Qwen/Qwen3-4B-Thinking-2507
rslm train-lora --db artifacts/memory.sqlite --out adapters/adapter_v001
```

If optional dependencies are missing, the command exits cleanly with a clear message and no crash.

Train-and-promote (LocalHF only) trains an adapter and activates it only if it passes heldout/regression gates:

```bash
rslm train-and-promote --db artifacts/memory.sqlite --out adapters/adapter_v002 --backend localhf --heldout-size 40 --regression-size 25 --min-improvement 0.02 --max-regression-drop 0.0
```

Promotions now require statistically defensible improvements on heldout and no unacceptable regressions on hidden.

## Docker verifier mode

Set verification mode to run candidate tests inside a locked-down Docker container:

```bash
export RSLM_VERIFY_MODE=docker
rslm eval --db artifacts/memory.sqlite --backend mock --conditions baseline
```

Docker mode fails fast if Docker is missing and runs with no network, read-only filesystem, and strict resource limits.

## Memory provenance and rollback

List/activate/deactivate/rollback rules and procedures:

```bash
rslm rules list --db artifacts/memory.sqlite --all
rslm rules activate --db artifacts/memory.sqlite --rule-id 3
rslm rules rollback --db artifacts/memory.sqlite --to-rule-id 2

rslm procedures list --db artifacts/memory.sqlite --all
rslm procedures deactivate --db artifacts/memory.sqlite --procedure-id 4
```

## Curriculum mining

Mine failures and generate new tasks to expand training:

```bash
rslm mine-curriculum --db artifacts/memory.sqlite --out artifacts/generated_tasks.jsonl --max-new 50
```

Set `RSLM_INCLUDE_GENERATED_TASKS=1` to include generated tasks in the train pool (never in hidden).

## Run manifests

Each `run-iteration` and `eval-robust` writes an artifact under `artifacts/runs/` with config, task ids, metrics, and git hash.

## Notes

- Python sandboxing is not perfectly safe. The verifier applies best-effort isolation but cannot guarantee complete security.
- Memory is only written for passing episodes, and training only uses verified traces.
- For the mock backend, the evaluation label uses "semantic" for rules and procedures instead of "learning".
