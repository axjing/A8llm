# XlmPrimer AI Agent Specification

**Never give up on the right solution.**

## 1. Overview

### 1.1 Purpose

Defines mandatory behavioral and coding rules for all AI coding agents operating in this repository. All AI-generated code, file modifications, shell commands, Git operations, and comments must comply.

### 1.2 Scope

Python and documentation. TypeScript is deferred for future Web UI.

### 1.3 Rule Priority (Strict)

1. Explicit user instructions (confirm if conflicting)
2. This document
3. Language official standards (PEP8)
4. Community conventions

### 1.4 Core Principles

- **Correctness**: Resolve issues completely. No skipped validation, deferred fixes, or suboptimal implementations.
- **Session isolation**: No cross-session file modification or overwriting.
- **Confirmation on conflict**: Pause and request confirmation for rule conflicts.

### 1.5 Design Principles

#### Principle 1: Usability over Performance

The project's primary goal is usability; a secondary goal is _reasonable_ performance. We operate in a _usability-first_ manner and avoid _restriction-first_ regimes without a clear-eyed view of the tradeoffs.

#### Principle 2: Simple Over Easy

Borrowing from The Zen of Python:
- _Explicit is better than implicit_
- _Simple is better than complex_

We favor exposing simple and explicit building blocks rather than APIs that are easy-to-use but hard to debug.

#### Principle 3: Primary Language First

The framework is built deeply into Python. We reuse the PyTorch ecosystem rather than reinventing.

---

## 2. Agent Behavior

This file exists because LLMs make predictable mistakes. These are not suggestions. These are rules.

### 1. Read Before You Write

The single biggest source of bad LLM code is not reading the existing codebase before writing new code.

Before writing anything:

- Read the files you're about to modify. Not skim. Read.
- Look at how similar things are done elsewhere in the project. If there's a pattern for API routes, follow that pattern. If there's a utility function that does half of what you need, use it.
- Check the imports at the top of the file. They tell you what libraries this project actually uses. Don't introduce a new library if the project uses something else everywhere.
- Look at the test files. They tell you what the expected behavior actually is.

If you're not sure how something is done in this project, say so.

### 2. Think Before You Code

Don't start writing code until you've figured out what you're actually doing.

**State your assumptions.** If the user says "add KV cache" that could mean static KV cache, paged attention, or sliding window. Don't pick one silently.

**Name the tradeoffs.** Almost every implementation choice has a tradeoff. If you're adding Flash Attention, say "this requires specific GPU architectures and won't work on older CUDA versions."

**If multiple approaches exist, present them briefly.** Two, maybe three. With a recommendation.

**If something is confusing, stop.** Don't fill confusion with plausible-sounding code.

### 3. Simplicity

Write the minimum amount of code that solves the problem.

**Premature abstraction.** Duplication is far cheaper than the wrong abstraction. Copy-paste twice before you abstract.

**Speculative error handling.** Only handle errors that can actually occur. Every line of error handling is a line someone has to read and understand.

**Unnecessary configurability.** Configuration is not free. Hardcode things until there's a real reason not to.

**Dead flexibility.** Interfaces with one implementation. Abstract base classes with one child. These have a cost and zero benefit until a second implementation actually exists.

### 4. Surgical Changes

When you edit existing code, your diff should be as small as possible.

**Don't touch what you weren't asked to touch.** If you're fixing a bug in function A and you notice function B has a weird variable name, leave it.

**Match the existing style.** If the file uses `snake_case`, use `snake_case`. Consistency within a file beats your personal preference.

**Clean up after yourself, not after others.** If your change made an import unused, remove that import. But only if YOUR change caused it. Pre-existing dead code is not your problem.

**Don't reformat.** Don't run a formatter on a file that wasn't formatted before. Reformatting creates massive diffs that hide your actual changes.

### 5. Verification

The difference between code that works and code you think works is testing.

**Write the test first when fixing bugs.** Before you fix anything, write a test that reproduces the bug. Run it. Watch it fail. Then fix the bug. Run the test. Watch it pass.

**Run existing tests before and after your changes.** If tests were already failing before your change, say so. Don't silently ignore pre-existing failures.

**Test behavior, not implementation.** Test the interesting cases, not the trivial ones.

### 6. Goal-Driven Execution

Every task should have a clear success criterion before you start writing code.

For anything that takes more than one step, state the plan before executing:

```
Plan:
1. Read the existing model config to understand the config structure
2. Add the new field to LLMConfig/VLMConfig
3. Modify the forward pass to use the new config
4. Add validation for the new field
5. Write tests for the new behavior
6. Run full test suite to check for regressions
```

### 7. Debugging

When something doesn't work, don't guess. Investigate.

**Read the error message.** The whole thing. Including the stack trace.

**Reproduce first.** Before you change anything, make sure you can reproduce the problem.

**Change one thing at a time.** If you change three things and the bug goes away, you don't know which change fixed it.

**Understand the root cause.** Don't add workarounds without understanding the root cause. The null check might prevent a crash, but the underlying bug is still there.

**If you're stuck, say so.** "I've tried X and Y and neither worked. Here's what I'm seeing."

### 8. Dependencies

Don't add dependencies without thinking about it.

Before adding a package:

- Can you do this with what's already in the project? If the project already has a tokenizer, don't add another one.
- Can you do this with PyTorch built-ins?
- Is this dependency actually maintained? Check the last commit date.

When you do add a dependency, say why.

### 9. Communication

**Say what you did and why.** Don't just dump a code block.

**Flag concerns.** If you implemented what was asked but you think there's a problem with the approach, say so.

**Be precise about what you're uncertain about.** "I'm not sure if this CUDA kernel works on Ampere" is useful.

**Don't explain things the user already knows.** If they asked you to add an attention layer, don't explain what self-attention is.

### 10. Common Failure Modes

**The Kitchen Sink.** Asked to add one feature, restructure half the codebase. Don't.

**The Wrong Abstraction.** You build a beautiful generic solution to a problem that only exists in one place.

**The Invisible Decision.** You make an architectural choice (config shape, API design) without flagging it.

**The Optimistic Path.** You write code that handles the happy path perfectly and ignores everything else. Think about what happens when the GPU OOMs, when the checkpoint doesn't exist, when the batch size is 0.

**The Knowledge Hallucination.** You confidently use a PyTorch API that doesn't exist, a parameter that was removed two versions ago. Check the docs.

**The Style Drift.** You write code in your "preferred" style instead of matching the project.

**The Runaway Refactor.** Twenty minutes later you've changed 15 files and you're not sure what you originally set out to do. If a fix is cascading, stop.

---

## 3. Development Environment

### 3.1 Python

- Virtual environment: `.venv` (project-local, via `uv venv`).
- Activate before running/linting/committing: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix).
- Manage dependencies with `uv sync`, `uv add`, `uv remove`, or `uv pip install`.
- Python 3.10 minimum.
- PyTorch is the core dependency. CUDA required for GPU training.

### 3.2 TypeScript/JavaScript

Not yet in use. When added, depend on `package.json` + `package-lock.json` consistency.

---

## 4. Validation

### 4.1 Lint

```bash
uv run ruff check xlm/ tests/
uv run ruff format --check xlm/ tests/
```

### 4.2 Type Check

```bash
uv run mypy xlm/ tests/ --strict
```

### 4.3 Test

```bash
uv run python -m pytest tests/ -v
```

### 4.4 Line Length

- Python: max **88** characters per line (ruff config).

---

## 5. Python Standards

Ref: <https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents.html>

### 5.1 Formatting

- Indent: 4 spaces. Tabs prohibited.
- No trailing spaces. One blank line at file end.
- Two blank lines between module-level functions/classes.
- One blank line between class methods.
- Line wrapping: implicit parentheses. Backslash continuation prohibited.

### 5.2 Naming

| Element | Convention |
|---------|-----------|
| Files/modules/packages | `snake_case` |
| Functions/variables | `snake_case` |
| Constants | `UPPER_SNAKE_CASE` |
| Classes/Exceptions | `PascalCase` |
| Private members | `_prefix` |

### 5.3 Imports (Hard)

- All at file top. Inline/dynamic/`__import__` prohibited.
- No wildcard (`from X import *`).
- Order: Standard Library → Third-party → Project Internal → Relative.
- No `from __future__` imports (project requires Python 3.10+, which natively supports `X | Y` union syntax via PEP 604).

### 5.4 Docstrings

- **Google-style only**. Sphinx (`:param:`, `:return:`) prohibited.
- Required for: modules, classes, public functions, complex private functions.
- Fields: `Args:`, `Returns:`, `Raises:` (on demand).

### 5.5 Type Annotations

- Full annotations on all variables, parameters, return values.
- `Any` prohibited except for unavoidable third-party dynamic interfaces (comment reason).
- Use native generics: `list[]`, `dict[]`, `set[]`.
- Fix type errors by upgrading deps. Never delete code or suppress checks.

### 5.6 Functions & Classes

- Single-use inline logic stays inline. No trivial single-call extraction.
- No mutable default parameters; use `None` placeholder.
- Class order: Docstring → Class Vars → `__init__` → Public → Static/Class → Private → Magic.
- All instance attributes in `__init__`. No dynamic attribute injection.

### 5.7 Exception Handling

- No bare `except:`. Always catch explicit exception types.
- Use `with` context manager for all resource operations.

### 5.8 Ignore Python 2 Compatibility

This project uses Python 3+. Do not use the `__future__` module.

### 5.9 Platform Support

Tests and features must support Linux, macOS and Windows unless feature is explicitly OS-specific.

### 5.10 Project-Specific Conventions

- **Model config**: `LLMConfig` and `VLMConfig` (dataclasses in `xlm/models/config.py`) hold all hyperparameters with HF-compatible serialization (`from_pretrained`, `save`, `update_from_hf_config`).
- **GPT model**: `xlm/models/gpt.py` — GPT-2 style decoder with `Block` composition (`LayerNorm` → `CausalSelfAttention` → `LayerNorm` → `MLP`). `xlm/models/layers.py` holds reusable primitives (`Linear`, `Conv1D`, `LayerNorm`, `RMSNorm`, `CausalSelfAttention`, `GroupedQueryAttention`, `LlamaMLP`, `MLP`).
- **Language model**: `xlm/models/language_model.py` — `LlamaTransformer` with RoPE, RMSNorm, GQA, KV caching, `from_pretrained()` for SmolLM2 weights.
- **Flash Attention**: `xlm/models/flash_attention.py` — FA3 integration for Hopper GPUs with SDPA fallback.
- **Position embeddings**: `xlm/models/position_embedding.py` — RoPE and learned positional embeddings.
- **Vision-language**: `xlm/models/vision_language_model.py` — `VisionLanguageModel` (ViT + LlamaTransformer + ModalityProjector) with `generate()`, `from_pretrained()`, `save_pretrained()`, `push_to_hub()`. `xlm/models/vision_transformer.py` — SigLIP-based ViT. `xlm/models/modality_projector.py` — cross-modal projection.
- **Tokenizer**: `xlm/common/tokenizer.py` — factory pattern (tiktoken, HuggingFace). `xlm/trainer/train_tokenizer.py` — custom tokenizer training.
- **Training**: `xlm/train_llm.py` — LLM pretraining (DDP, Muon optimizer, cosine LR, scaling laws, FP8 optional, checkpointing, eval, sampling). `xlm/train_vlm.py` — VLM training (image+text, gradient accumulation, DDP, lmms-eval).
- **Trainer utilities**: `xlm/trainer/optim.py`, `xlm/trainer/distributed.py` (DDP), `xlm/trainer/train_fp8.py` (FP8 conversion).
- **Data**: `xlm/data/` — `get_datasets.py`, `text_pretrain_loader.py`, `vqa_datasets.py`, `processors.py`.
- **Inference**: `xlm/engine/engine_inference.py` — `Engine` class with KV cache (FA3-optimized), tool-use state machine (calculator). `xlm/engine/utils_checkpoints.py` — checkpoint utilities.
- **Evaluation**: `xlm/eval.py`, `xlm/evaluator/` — eval core, loss-based evaluation.
- **Common**: `xlm/common/` — `execution.py`, `file_os.py`, `logger.py`, `tokenizer.py`.
- **Config files**: JSON (e.g., `configs/gpt2.json`) and CLI arguments. No Pydantic.
- **Checkpoint format**: safetensors.
- **Distributed**: `torchrun` / DDP multi-GPU training. Scripts designed for Slurm (`srun`) and multi-node clusters.
- **Logging**: SwanLab / DummySwanLab for training metrics.
- **Async**: Not used. All training and inference is synchronous PyTorch.

---

## 6. Engineering Restrictions

### 6.1 No Hardcoding

- Hyperparameters, file paths, model checkpoints — use config classes, JSON files, or CLI arguments.
- No magic numbers.

### 6.2 Modification Rules

- Delete/disable existing features only after user confirmation.
- Large-scale refactoring: read full file/module first.

### 6.3 3rdparty Directory

- `3rdparty/` is read-only reference code (nanoGPT, nanochat). Never modify files inside it.

---

## 7. Git Workflow

### 7.1 Commit Rules

- Stage only files modified by current session.
- Explicit file path staging only. `git add .` / `git add -A` prohibited.
- Verify with `git status` before commit.

### 7.2 Commit Message Format

```
{feat|fix|docs|refactor|perf|test}[(models|trainer|data|engine|evaluator|common|config|scripts)]: concise English description
```

Examples:

- `feat(models): add KV cache to inference engine`
- `fix(trainer): handle empty batches in data loader`
- `perf(engine): enable Flash Attention for A100+ GPUs`
- `docs: update README with quick start`
- `refactor: extract LayerNorm to layers.py`

### 7.3 Forbidden Commands

`git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`, `git add -A`, `git add .`, `git commit --no-verify`, `git push --force`.

### 7.4 Conflict Handling

- Resolve only in self-modified files.
- Abort rebase and notify user for external file conflicts.

---

## 8. Issue & PR Workflow

- No branch switching without user instruction.
- Inspect PR via `gh pr view`, `gh pr diff`, `git show`.
- Auto-close issues: `closes #1`.

---

## 9. Standard Workflow

1. **Analyze**: Clarify requirements. Read full module for large changes.
2. **Implement**: Follow language spec strictly.
3. **Validate**: Activate env + lint + type-check + test.
4. **Commit**: Explicit stage + standardized message.
5. **Finalize**: Link issues, finish review.

---

## 10. Forbidden Checklist

### Hard Prohibitions

- Bypassing validation (ruff, mypy, pytest)
- Dynamic imports, wildcard imports
- Hardcoding configurable values
- Dangerous Git operations and force push
- Python: Sphinx docstrings, bare except, overuse `Any`, tab indent
- Modifying files under `3rdparty/`

### User Confirmation Required

- Delete/disable existing features
- Modify global configs
- Disable validation rules
- Drop backward compatibility

---

## Appendix: Quick Reference

```bash
# Development
.venv\Scripts\activate                # Windows
source .venv/bin/activate             # Unix
uv sync                               # install all deps
uv pip install torch torchvision       # or add specific packages

# Lint & Type Check
uv run ruff check xlm/ tests/
uv run ruff format --check xlm/ tests/
uv run mypy xlm/ tests/ --strict

# Test
uv run python -m pytest tests/ -v

# Training
python -m xlm.train_llm
bash scripts/train_vlm.sh

# Git
git status
git add <file-path>
git commit -m "feat(models): add rotary position embeddings"
git push
```
