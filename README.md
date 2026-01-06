# recursive_slm_lab

A runnable demo for persistent memory plus continuous learning using verification-gated recursion.

This repo evaluates four conditions:
1. Baseline (no memory, no learning)
2. Memory only
3. Learning only
4. Memory plus Learning

The system attempts tasks, verifies candidates with unit tests, stores only passing episodes in SQLite, consolidates memory into rules and procedures, and optionally trains LoRA adapters on verified traces.

## Quickstart

```bash
pip install -e .
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

## CLI overview

```bash
rslm init-db --db artifacts/memory.sqlite
rslm seed-tasks
rslm run-iteration --db artifacts/memory.sqlite --tasks bundled --k 8 --mode trainpool --backend mock
rslm consolidate --db artifacts/memory.sqlite --min-evidence 3 --backend mock
rslm eval --db artifacts/memory.sqlite --backend mock --conditions all --k 1 --heldout-size 40 --output artifacts/results.json
rslm plot --input artifacts/results.json --output artifacts/results.png
```

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

Install optional dependencies and provide a local model path:

```bash
pip install -e .[localhf]
export RSLM_HF_MODEL_PATH=/path/to/local/model
rslm train-lora --db artifacts/memory.sqlite --out adapters/adapter_v001
```

If optional dependencies are missing, the command exits cleanly with a clear message and no crash.

## Notes

- Python sandboxing is not perfectly safe. The verifier applies best-effort isolation but cannot guarantee complete security.
- Memory is only written for passing episodes, and training only uses verified traces.
