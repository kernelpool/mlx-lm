# RLM (Recursive Language Model) Inference

RLM is an inference mode that allows LLMs to process arbitrarily long contexts by treating the input as a variable in a Python REPL environment. Instead of feeding the entire context directly to the model, the LLM can programmatically examine, filter, and decompose the context using code.

Based on ["Recursive Language Models"](https://arxiv.org/abs/2512.24601) by Zhang, Kraska, and Khattab (MIT CSAIL, 2025).

## Quick Start

```bash
# Basic usage - pipe context via stdin
echo "The secret code is 42." | mlx_lm.rlm \
    --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
    --task "Find the secret code"

# From file
mlx_lm.rlm --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
    --context-file document.txt \
    --task "Summarize the key findings" \
    --verbose
```

## How It Works

1. Your context is loaded into a `context` variable in a Python REPL
2. The LLM generates Python code to examine and process the context
3. Code is executed, and results are fed back to the LLM
4. The LLM can call `llm_query()` for recursive sub-queries
5. Loop continues until `FINAL()` or `FINAL_VAR()` is called

```
┌─────────────────────────────────────┐
│ Input Context (any size)            │
└──────────────┬──────────────────────┘
               ↓
    ┌──────────────────────┐
    │ REPL Environment     │
    │ - context variable   │
    │ - llm_query()        │
    │ - Python builtins    │
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │ LLM generates code   │◄────┐
    └──────────┬───────────┘     │
               ↓                 │
    ┌──────────────────────┐     │
    │ Execute in REPL      │─────┘
    └──────────┬───────────┘
               ↓
    ┌──────────────────────┐
    │ FINAL(answer)        │
    └──────────────────────┘
```

## REPL Environment

The LLM has access to:

| Variable/Function | Description |
|-------------------|-------------|
| `context` | The input context as a string |
| `llm_query(prompt)` | Make a recursive LLM call |
| `print()` | Output for inspection |
| `FINAL(answer)` | Return final answer |
| `FINAL_VAR(varname)` | Return a variable as answer |

Pre-imported modules: `re`, `json`, `math`, `collections`, `itertools`, `functools`, `string`

## CLI Options

```
mlx_lm.rlm --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model path or HF repo | Llama-3.2-3B-Instruct-4bit |
| `--task`, `-t` | Task/query to execute | (required) |
| `--context`, `-c` | Context string | - |
| `--context-file`, `-f` | Load context from file | - |
| `--max-iterations` | Max REPL iterations | 20 |
| `--max-tokens`, `-m` | Max tokens per turn | 4096 |
| `--temp` | Sampling temperature | 0.0 |
| `--top-p` | Top-p sampling | 1.0 |
| `--top-k` | Top-k sampling | 0 |
| `--verbose`, `-v` | Stream intermediate steps | False |
| `--output-trajectory`, `-o` | Save trajectory to JSON | - |
| `--seed` | Random seed | - |
| `--max-kv-size` | Max KV cache size | - |

## Examples

### Needle in Haystack

Find a hidden value in a large text:

```bash
cat large_document.txt | mlx_lm.rlm \
    --model mlx-community/Qwen2.5-Coder-32B-Instruct-8bit \
    --task "Find the secret password hidden in this text" \
    --verbose
```

### Document Analysis

Analyze and summarize a document:

```bash
mlx_lm.rlm \
    --model mlx-community/Qwen2.5-Coder-32B-Instruct-8bit \
    --context-file report.txt \
    --task "List all action items mentioned in this document" \
    --verbose \
    --output-trajectory analysis.json
```

### Data Extraction

Extract structured data from text:

```bash
mlx_lm.rlm \
    --model mlx-community/Qwen2.5-Coder-32B-Instruct-8bit \
    --context "$(cat emails.txt)" \
    --task "Extract all email addresses and phone numbers"
```

## Python API

```python
from mlx_lm import load
from mlx_lm.rlm import rlm_generate, RLMConfig

# Load model
model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")

# Configure RLM
config = RLMConfig(
    max_iterations=10,
    max_tokens_per_turn=2048,
    verbose=True,
)

# Run inference
response = rlm_generate(
    model,
    tokenizer,
    context="Your long context here...",
    task="Your task description",
    config=config,
)

print(f"Answer: {response.answer}")
print(f"Iterations: {response.iterations}")
print(f"Tokens used: {response.total_tokens}")
```

## Tips

1. **Use capable models**: RLM requires models that can write Python code. Qwen2.5-Coder or similar code-focused models work best.

2. **Verbose mode**: Use `--verbose` to see what the model is doing and debug issues.

3. **Large contexts**: RLM shines with contexts that exceed the model's context window. The model will use chunking strategies to process them.

4. **Save trajectories**: Use `--output-trajectory` to save the full execution trace for debugging or analysis.

## Limitations

- Requires code-capable models (instruction-tuned coder models recommended)
- Each iteration adds latency; complex tasks may take many iterations
- No sandboxing - code runs with full Python access (trust your inputs)
