# Copyright © 2025 Apple Inc.

"""
RLM (Recursive Language Model) inference for mlx-lm.

Based on "Recursive Language Models" by Zhang, Kraska, and Khattab (MIT CSAIL, 2025).
https://arxiv.org/abs/2512.24601

RLM treats prompts as part of an external Python REPL environment, allowing the
LLM to programmatically examine, decompose, and recursively call itself.
"""

import argparse
import io
import json
import re
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import mlx.core as mx

from .generate import stream_generate
from .sample_utils import make_sampler
from .tokenizer_utils import TokenizerWrapper
from .utils import load

# Default values
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_MAX_TOKENS = 4096
DEFAULT_REPL_OUTPUT_LIMIT = 10000
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0


@dataclass
class RLMConfig:
    """Configuration for RLM inference."""

    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_tokens_per_turn: int = DEFAULT_MAX_TOKENS
    repl_output_limit: int = DEFAULT_REPL_OUTPUT_LIMIT
    temp: float = DEFAULT_TEMP
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    max_kv_size: Optional[int] = None
    verbose: bool = False


@dataclass
class RLMResponse:
    """Response from RLM inference."""

    answer: Optional[str] = None
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    iterations: int = 0
    error: Optional[str] = None


# System prompt template based on the RLM paper
RLM_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs. You will be queried iteratively until you provide a final answer.

Your context is a string with {context_length} total characters.

The REPL environment is initialized with:
1. A `context` variable that contains the input data. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM inside your REPL environment.
3. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use variables as buffers to build up your final answer.

Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, suppose you're trying to answer a question about a document. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer:
```repl
query = "What is the main conclusion of this document?"
chunk_size = len(context) // 5
buffers = []
for i in range(5):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < 4 else len(context)
    chunk = context[start:end]
    summary = llm_query(f"Summarize this section, focusing on conclusions: {{chunk}}")
    buffers.append(summary)
    print(f"Section {{i+1}}: {{summary[:200]}}...")

final_answer = llm_query(f"Based on these section summaries, what is the main conclusion? Summaries: {{buffers}}")
print(f"Final answer: {{final_answer}}")
```

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.

Task: {task}
"""


def extract_code_blocks(text: str) -> List[str]:
    """Extract Python/REPL code blocks from LLM output."""
    # Match ```repl, ```python, or just ``` code blocks
    pattern = r"```(?:repl|python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [m.strip() for m in matches if m.strip()]


def extract_final_answer(text: str, repl_locals: dict) -> Optional[str]:
    """Check for FINAL() or FINAL_VAR() and extract answer."""
    # FINAL_VAR(varname) pattern - check first since it's more specific
    var_match = re.search(r"FINAL_VAR\(([a-zA-Z_][a-zA-Z0-9_]*)\)", text)
    if var_match:
        varname = var_match.group(1)
        if varname in repl_locals:
            return str(repl_locals[varname])
        else:
            return f"[Error: Variable '{varname}' not found in REPL environment]"

    # FINAL(answer) pattern - handle multiline and various content
    final_match = re.search(r"FINAL\(([^)]+)\)", text, re.DOTALL)
    if final_match:
        answer = final_match.group(1).strip()
        # Check if it's a variable name that exists in repl_locals
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", answer) and answer in repl_locals:
            return str(repl_locals[answer])
        # Remove surrounding quotes if present
        if (answer.startswith('"') and answer.endswith('"')) or (
            answer.startswith("'") and answer.endswith("'")
        ):
            answer = answer[1:-1]
        return answer

    return None


def create_repl_environment(
    context: str,
    model,
    tokenizer,
    config: RLMConfig,
    sampler: Callable,
) -> tuple:
    """Create the REPL environment with context and llm_query function."""

    # Track token usage
    token_counter = {"count": 0}

    def llm_query(prompt: str) -> str:
        """Recursive LLM call from within REPL."""
        response_text = ""

        # Apply chat template for the sub-query
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        for resp in stream_generate(
            model,
            tokenizer,
            formatted_prompt,
            max_tokens=config.max_tokens_per_turn,
            sampler=sampler,
            max_kv_size=config.max_kv_size,
        ):
            response_text += resp.text
            token_counter["count"] += 1

        return response_text

    # Build globals dict with builtins and useful modules
    # Include __import__ so the model can import modules if needed
    repl_globals = {
        "__builtins__": {
            "__import__": __import__,  # Allow imports
            "print": print,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "reversed": reversed,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "round": round,
            "any": any,
            "all": all,
            "isinstance": isinstance,
            "type": type,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "open": open,  # Trust user - no sandboxing
            "True": True,
            "False": False,
            "None": None,
        },
        # Context and llm_query
        "context": context,
        "llm_query": llm_query,
        # Pre-imported useful modules
        "re": __import__("re"),
        "json": __import__("json"),
        "math": __import__("math"),
        "collections": __import__("collections"),
        "itertools": __import__("itertools"),
        "functools": __import__("functools"),
        "string": __import__("string"),
    }

    repl_locals = {}

    return repl_globals, repl_locals, token_counter


def execute_in_repl(
    code: str, repl_globals: dict, repl_locals: dict, output_limit: int
) -> Dict[str, Any]:
    """Execute code in the REPL environment and capture output."""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    result = {
        "code": code,
        "stdout": "",
        "stderr": "",
        "error": None,
        "success": True,
    }

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, repl_globals, repl_locals)

        result["stdout"] = stdout_capture.getvalue()
        result["stderr"] = stderr_capture.getvalue()

        # Truncate if needed
        if len(result["stdout"]) > output_limit:
            result["stdout"] = (
                result["stdout"][:output_limit]
                + f"\n... (truncated {len(result['stdout']) - output_limit} chars)"
            )

    except Exception as e:
        result["success"] = False
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["stderr"] = traceback.format_exc()

    return result


def build_continuation_prompt(
    trajectory: List[Dict], last_response: str, task: str
) -> str:
    """Build the continuation prompt with execution results."""
    prompt_parts = []

    # Add the last LLM response
    prompt_parts.append(f"Your previous response:\n{last_response}\n")

    # Add execution results from this iteration
    if trajectory:
        latest = trajectory[-1]
        prompt_parts.append("REPL execution result:")
        if latest.get("stdout"):
            prompt_parts.append(f"Output:\n{latest['stdout']}")
        if latest.get("stderr") and not latest.get("success"):
            prompt_parts.append(f"Error:\n{latest['stderr']}")
        if latest.get("error"):
            prompt_parts.append(f"Exception: {latest['error']}")

    prompt_parts.append(
        "\nContinue working on the task. Remember to use FINAL() or FINAL_VAR() when you have your answer."
    )
    prompt_parts.append(f"\nTask reminder: {task}")

    return "\n\n".join(prompt_parts)


def rlm_generate(
    model,
    tokenizer,
    context: str,
    task: str,
    config: RLMConfig,
) -> RLMResponse:
    """
    Execute RLM (Recursive Language Model) inference.

    Args:
        model: The language model
        tokenizer: The tokenizer
        context: The input context (available as 'context' variable in REPL)
        task: The task/query to execute
        config: RLM configuration

    Returns:
        RLMResponse with answer, trajectory, and metadata
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Create sampler
    sampler = make_sampler(config.temp, config.top_p, top_k=config.top_k)

    # Initialize REPL environment
    repl_globals, repl_locals, token_counter = create_repl_environment(
        context, model, tokenizer, config, sampler
    )

    # Build initial system prompt
    system_prompt = RLM_SYSTEM_PROMPT.format(
        context_length=len(context),
        task=task,
    )

    # Initialize response
    response = RLMResponse()
    trajectory = []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Begin working on the task: {task}"},
    ]

    iteration = 0
    while iteration < config.max_iterations:
        iteration += 1
        response.iterations = iteration

        if config.verbose:
            print(f"\n{'='*60}")
            print(f"[RLM Iteration {iteration}]")
            print(f"{'='*60}")

        # Generate LLM response
        formatted_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        llm_output = ""
        for resp in stream_generate(
            model,
            tokenizer,
            formatted_prompt,
            max_tokens=config.max_tokens_per_turn,
            sampler=sampler,
            max_kv_size=config.max_kv_size,
        ):
            llm_output += resp.text
            token_counter["count"] += 1
            if config.verbose:
                print(resp.text, end="", flush=True)

        if config.verbose:
            print()  # Newline after streaming

        # Extract and execute code blocks FIRST (so variables are available for FINAL)
        code_blocks = extract_code_blocks(llm_output)

        if code_blocks:
            # Execute each code block
            for i, code in enumerate(code_blocks):
                # Remove any FINAL() calls from code before execution
                # (FINAL is a marker, not a real function)
                code_to_exec = re.sub(r"FINAL\([^)]*\)", "pass", code)
                code_to_exec = re.sub(r"FINAL_VAR\([^)]*\)", "pass", code_to_exec)

                if config.verbose:
                    print(f"\n[Executing code block {i+1}/{len(code_blocks)}]")
                    print(f"Code:\n{code}")

                result = execute_in_repl(
                    code_to_exec, repl_globals, repl_locals, config.repl_output_limit
                )
                trajectory.append(result)

                if config.verbose:
                    if result["stdout"]:
                        print(f"\nOutput:\n{result['stdout']}")
                    if result["error"]:
                        print(f"\nError: {result['error']}")

        # Check for FINAL markers (after code execution so variables are available)
        final_answer = extract_final_answer(llm_output, repl_locals)
        if final_answer is not None:
            response.answer = final_answer
            response.total_tokens = token_counter["count"]
            response.trajectory = trajectory
            if config.verbose:
                print(f"\n[RLM] Final answer extracted after {iteration} iterations")
            return response

        if not code_blocks:
            # No code to execute, but also no FINAL - prompt the model to continue
            if config.verbose:
                print("[RLM] No code blocks found, prompting to continue...")

            messages.append({"role": "assistant", "content": llm_output})
            messages.append(
                {
                    "role": "user",
                    "content": "Please write Python code in ```repl blocks to interact with the context, or provide your final answer using FINAL() or FINAL_VAR().",
                }
            )
            continue

        # Build continuation prompt
        continuation = build_continuation_prompt(
            trajectory[-len(code_blocks) :], llm_output, task
        )

        messages.append({"role": "assistant", "content": llm_output})
        messages.append({"role": "user", "content": continuation})

    # Max iterations reached
    response.error = (
        f"Max iterations ({config.max_iterations}) reached without FINAL answer"
    )
    response.total_tokens = token_counter["count"]
    response.trajectory = trajectory

    if config.verbose:
        print(f"\n[RLM] {response.error}")

    return response


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="RLM (Recursive Language Model) inference - treat prompts as part "
        "of an external REPL environment for programmatic context manipulation."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"The path to the local model directory or Hugging Face repo. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=True,
        help="The task/query to execute on the context",
    )
    parser.add_argument(
        "--context",
        "-c",
        type=str,
        help="Context string (available as 'context' variable in REPL)",
    )
    parser.add_argument(
        "--context-file",
        "-f",
        type=str,
        help="Load context from file",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum REPL iterations. Default: {DEFAULT_MAX_ITERATIONS}",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens per generation turn. Default: {DEFAULT_MAX_TOKENS}",
    )
    parser.add_argument(
        "--repl-output-limit",
        type=int,
        default=DEFAULT_REPL_OUTPUT_LIMIT,
        help=f"Truncate REPL output to this many characters. Default: {DEFAULT_REPL_OUTPUT_LIMIT}",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=DEFAULT_TEMP,
        help=f"Sampling temperature. Default: {DEFAULT_TEMP}",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Top-p sampling. Default: {DEFAULT_TOP_P}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Top-k sampling. Default: {DEFAULT_TOP_K} (disabled)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Stream intermediate steps to terminal",
    )
    parser.add_argument(
        "--output-trajectory",
        "-o",
        type=str,
        help="Save full trajectory to JSON file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="PRNG seed for reproducibility",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Set the maximum key-value cache size",
    )

    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        mx.random.seed(args.seed)

    # Load context from various sources
    context = None

    if args.context:
        context = args.context
    elif args.context_file:
        with open(args.context_file, "r", encoding="utf-8") as f:
            context = f.read()
    elif not sys.stdin.isatty():
        # Read from stdin if piped
        context = sys.stdin.read()

    if context is None:
        parser.error(
            "No context provided. Use --context, --context-file, or pipe to stdin."
        )

    # Load model and tokenizer
    print(f"[RLM] Loading model: {args.model}")
    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter_path,
        tokenizer_config={
            "trust_remote_code": True if args.trust_remote_code else None
        },
    )

    # Create config
    config = RLMConfig(
        max_iterations=args.max_iterations,
        max_tokens_per_turn=args.max_tokens,
        repl_output_limit=args.repl_output_limit,
        temp=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        max_kv_size=args.max_kv_size,
        verbose=args.verbose,
    )

    print(f"[RLM] Context length: {len(context)} characters")
    print(f"[RLM] Task: {args.task}")
    print(f"[RLM] Starting inference (max {config.max_iterations} iterations)...")
    print()

    # Run RLM inference
    start_time = time.perf_counter()
    response = rlm_generate(model, tokenizer, context, args.task, config)
    elapsed = time.perf_counter() - start_time

    # Output results
    print("\n" + "=" * 60)
    print("[RLM Result]")
    print("=" * 60)

    if response.answer:
        print(f"\nAnswer:\n{response.answer}")
    elif response.error:
        print(f"\nError: {response.error}")
    else:
        print("\nNo answer produced.")

    print(f"\nStatistics:")
    print(f"  Iterations: {response.iterations}")
    print(f"  Total tokens: {response.total_tokens}")
    print(f"  Time: {elapsed:.2f}s")

    # Save trajectory if requested
    if args.output_trajectory:
        trajectory_data = {
            "task": args.task,
            "context_length": len(context),
            "answer": response.answer,
            "error": response.error,
            "iterations": response.iterations,
            "total_tokens": response.total_tokens,
            "elapsed_time": elapsed,
            "trajectory": response.trajectory,
        }
        with open(args.output_trajectory, "w", encoding="utf-8") as f:
            json.dump(trajectory_data, f, indent=2)
        print(f"\nTrajectory saved to: {args.output_trajectory}")


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_lm.rlm...` directly is deprecated."
        " Use `mlx_lm.rlm...` or `python -m mlx_lm rlm ...` instead."
    )
    main()
