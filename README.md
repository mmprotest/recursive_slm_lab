# recursive_slm_lab

A runnable demo for persistent memory plus continuous learning using verification-gated recursion.

This repo evaluates four conditions:
1. Baseline (no memory, no learning)
2. Memory only
3. Semantic memory only (mock backend)
4. Memory plus learning (adapters when available)

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
Each eval run is also stored in the SQLite DB so plots can show progress across iterations.
Set `RSLM_FAST_VERIFY=1` to use a lightweight assert harness instead of pytest when tasks provide `assert_tests`.

## CLI overview

```bash
rslm init-db --db artifacts/memory.sqlite
rslm seed-tasks
rslm run-iteration --db artifacts/memory.sqlite --tasks bundled --k 4 --mode trainpool --backend mock --memory-enabled --heldout-size 20 --task-limit 30
rslm consolidate --db artifacts/memory.sqlite --min-evidence 1 --backend mock
rslm eval --db artifacts/memory.sqlite --backend mock --conditions all --k 1 --heldout-size 20 --task-limit 20 --output artifacts/results_iter001.json
rslm plot --input artifacts/memory.sqlite --output artifacts/results.png
```

## Generate tasks

Regenerate the bundled task set with deterministic, hidden pytest tests:

```bash
rslm seed-tasks --regen --count 200
```

Then run a short iteration and evaluation:

```bash
rslm run-iteration --db artifacts/memory.sqlite --tasks bundled --k 4 --mode trainpool --backend mock --memory-enabled --heldout-size 20 --task-limit 20
rslm eval --db artifacts/memory.sqlite --backend mock --conditions all --k 1 --heldout-size 20 --task-limit 20 --output artifacts/results_iter001.json
```

Task tests are stored only in the verifier and are not exposed to the model prompts.

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
- For the mock backend, the evaluation label uses "semantic" for rules and procedures instead of "learning".
