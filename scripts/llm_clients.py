"""
llm_clients.py
==============

Provider-agnostic chat wrapper shared by the PyMETA prompting drivers
(``single_error_prompting.py`` and ``multi_error_prompting.py``).

The 2024 experiments hard-coded one vendor per script. This module lets a single
driver target any of four providers with the same call signature, so a model can
be swapped with ``--provider`` / ``--model`` instead of a code edit.

Supported providers and their API keys
--------------------------------------
    openai      OPENAI_API_KEY
    anthropic   ANTHROPIC_API_KEY
    gemini      GEMINI_API_KEY (or GOOGLE_API_KEY)
    deepseek    DEEPSEEK_API_KEY      (OpenAI-compatible endpoint)

Usage
-----
    from llm_clients import ChatClient

    client = ChatClient(provider="anthropic", model="claude-fable-5",
                        max_output_tokens=2048)
    text, error = client.generate(prompt, system="You are ...")
    print(client.usage)          # {'calls': 1, 'input_tokens': 812, ...}

Notes on current-generation models
----------------------------------
* Reasoning models spend output tokens on internal reasoning *before* the visible
  answer. ``max_output_tokens`` must leave room for both, otherwise the reply is
  truncated to an empty string. The 2024 setting of 50 output tokens no longer
  works; defaults here are 2048 (single-error) / 4096 (multi-error CoT).
* Sampling parameters are rejected (HTTP 400) on current Claude models
  (Fable 5 / Opus 5 / Sonnet 5 and the 4.6+ family), so ``temperature`` is only
  sent when explicitly requested and never for provider ``anthropic``. Use
  ``effort`` to control reasoning depth there instead.
* No cross-model fallback is configured on purpose: a silent substitution would
  attribute another model's predictions to the model under test.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple

# Provider -> environment variable(s) holding the API key.
ENV_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
}

# Fallback model per provider when --model is omitted. Override on the command
# line; these are not pinned snapshots.
DEFAULT_MODELS = {
    "openai": "gpt-5.6",
    "anthropic": "claude-opus-5",
    "gemini": "gemini-3.1-pro",
    "deepseek": "deepseek-v4-pro",
}

# Convenience aliases so runs can be launched with a short name.
MODEL_ALIASES = {
    "fable5": ("anthropic", "claude-fable-5"),
    "fable-5": ("anthropic", "claude-fable-5"),
    "opus5": ("anthropic", "claude-opus-5"),
    "opus-5": ("anthropic", "claude-opus-5"),
    "sonnet5": ("anthropic", "claude-sonnet-5"),
    "v4": ("deepseek", "deepseek-v4-pro"),
    "v4-pro": ("deepseek", "deepseek-v4-pro"),
    "v4-flash": ("deepseek", "deepseek-v4-flash"),
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Model-id prefix -> provider, so a full model id implies its vendor and
# --provider only has to be given for an unrecognised name.
PROVIDER_PREFIXES = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini", "gemini"),
    ("deepseek", "deepseek"),
)

# Errors worth retrying: rate limits, transient server faults, network trouble.
_TRANSIENT_MARKERS = (
    "rate limit", "rate_limit", "429", "500", "502", "503", "504",
    "overloaded", "timeout", "timed out", "connection", "network",
    "temporarily unavailable", "internal server error",
)


def resolve_model(provider: Optional[str], model: Optional[str]) -> Tuple[str, str]:
    """Resolve a (provider, model) pair, expanding aliases.

    ``--model fable5`` alone is enough to pick the provider; an explicit
    ``--provider`` always wins over an alias's implied provider.
    """
    if model and model.lower() in MODEL_ALIASES:
        alias_provider, alias_model = MODEL_ALIASES[model.lower()]
        return (provider or alias_provider), alias_model
    if model and not provider:
        low = model.lower()
        for prefix, inferred in PROVIDER_PREFIXES:
            if low.startswith(prefix):
                return inferred, model
    if not provider:
        raise ValueError(
            f"Cannot infer the provider from model={model!r}: pass --provider "
            f"({'/'.join(sorted(ENV_KEYS))}), or use a model id starting with "
            f"{', '.join(p for p, _ in PROVIDER_PREFIXES)}, or an alias "
            f"({', '.join(sorted(MODEL_ALIASES))})."
        )
    provider = provider.lower()
    if provider not in ENV_KEYS:
        raise ValueError(f"Unknown provider {provider!r}; expected one of {sorted(ENV_KEYS)}.")
    return provider, (model or DEFAULT_MODELS[provider])


def _read_key(provider: str) -> str:
    names = ENV_KEYS[provider]
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        f"No API key for provider {provider!r}. Set "
        + " or ".join(names)
        + f', e.g.  export {names[0]}="..."'
    )


def is_transient(message: str) -> bool:
    """True when an error message looks like something a retry could fix."""
    low = (message or "").lower()
    return any(marker in low for marker in _TRANSIENT_MARKERS)


class ChatClient:
    """Single-turn chat client with retries and token accounting.

    ``generate`` returns ``(text, error)``: exactly one of the two is None.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_output_tokens: int = 2048,
        temperature: Optional[float] = None,
        effort: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 5,
        retry_delay: float = 5.0,
    ):
        self.provider, self.model = resolve_model(provider, model)
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.effort = effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.usage = {
            "calls": 0,
            "failures": 0,
            "retries": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        # generate() may be called from several worker threads at once; the
        # counters below are read-modify-write, so they need a lock.
        self._usage_lock = threading.Lock()
        self._client = self._build_client()

    # ---------------------------------------------------------------- clients

    def _build_client(self):
        key = _read_key(self.provider)

        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(api_key=key, timeout=self.timeout, max_retries=0)

        if self.provider in ("openai", "deepseek"):
            from openai import OpenAI
            base_url = DEEPSEEK_BASE_URL if self.provider == "deepseek" else None
            return OpenAI(api_key=key, base_url=base_url, timeout=self.timeout, max_retries=0)

        if self.provider == "gemini":
            from google import genai
            return genai.Client(api_key=key)

        raise ValueError(self.provider)  # unreachable; resolve_model validates

    # ------------------------------------------------------------ generation

    def generate(self, prompt: str, system: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Send one prompt, retrying transient failures with exponential backoff."""
        delay = self.retry_delay
        last_error = "unknown error"

        for attempt in range(self.max_retries):
            try:
                text, in_tok, out_tok = self._call(prompt, system)
                with self._usage_lock:
                    self.usage["calls"] += 1
                    self.usage["input_tokens"] += in_tok
                    self.usage["output_tokens"] += out_tok
                if not text:
                    # An empty reply is usually the output cap being consumed by
                    # reasoning tokens. Report it rather than silently scoring it.
                    last_error = (
                        "empty response (raise --max-output-tokens; reasoning tokens "
                        f"count against the current cap of {self.max_output_tokens})"
                    )
                    with self._usage_lock:
                        self.usage["failures"] += 1
                    return None, last_error
                return text, None

            except Exception as exc:  # noqa: BLE001 - provider SDKs raise varied types
                last_error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                if attempt < self.max_retries - 1 and is_transient(last_error):
                    with self._usage_lock:
                        self.usage["retries"] += 1
                    print(f"  ⚠️  attempt {attempt + 1} failed ({last_error}); retrying in {delay:.0f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                break

        with self._usage_lock:
            self.usage["failures"] += 1
        return None, last_error

    def _call(self, prompt: str, system: Optional[str]) -> Tuple[str, int, int]:
        if self.provider == "anthropic":
            return self._call_anthropic(prompt, system)
        if self.provider in ("openai", "deepseek"):
            return self._call_openai_compatible(prompt, system)
        return self._call_gemini(prompt, system)

    def _call_anthropic(self, prompt: str, system: Optional[str]) -> Tuple[str, int, int]:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        # temperature is deliberately never sent: current Claude models reject
        # sampling parameters with a 400.

        if self.max_output_tokens > 8192:
            # Streaming keeps a large output cap from tripping the HTTP timeout.
            with self._client.messages.stream(**kwargs) as stream:
                response = stream.get_final_message()
        else:
            response = self._client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RuntimeError(f"model refused the request (category={category})")

        text = "\n".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", "")
        )
        usage = response.usage
        return text.strip(), usage.input_tokens, usage.output_tokens

    def _call_openai_compatible(self, prompt: str, system: Optional[str]) -> Tuple[str, int, int]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": self.model, "messages": messages}
        if self.provider == "openai":
            # Reasoning models count internal reasoning against this cap.
            kwargs["max_completion_tokens"] = self.max_output_tokens
            if self.effort:
                kwargs["reasoning_effort"] = self.effort
        else:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        response = self._client.chat.completions.create(**kwargs)
        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        return text, in_tok, out_tok

    def _call_gemini(self, prompt: str, system: Optional[str]) -> Tuple[str, int, int]:
        config = {"max_output_tokens": self.max_output_tokens}
        if system:
            config["system_instruction"] = system
        if self.temperature is not None:
            config["temperature"] = self.temperature

        response = self._client.models.generate_content(
            model=self.model, contents=prompt, config=config
        )
        text = (getattr(response, "text", None) or "").strip()
        meta = getattr(response, "usage_metadata", None)
        in_tok = getattr(meta, "prompt_token_count", 0) or 0
        out_tok = (getattr(meta, "candidates_token_count", 0) or 0) + (
            getattr(meta, "thoughts_token_count", 0) or 0
        )
        return text, in_tok, out_tok

    # ----------------------------------------------------------------- misc

    def run_tag(self) -> str:
        """Filesystem-safe identifier for this model, used in output filenames."""
        return "".join(ch if ch.isalnum() else "_" for ch in f"{self.provider}-{self.model}").strip("_")

    def describe(self) -> str:
        parts = [f"provider={self.provider}", f"model={self.model}",
                 f"max_output_tokens={self.max_output_tokens}"]
        if self.effort:
            parts.append(f"effort={self.effort}")
        if self.temperature is not None:
            parts.append(f"temperature={self.temperature}")
        return ", ".join(parts)


def add_model_arguments(parser):
    """Attach the shared provider/model flags to an argparse parser."""
    parser.add_argument("--provider", default=None,
                        help="LLM provider: " + "/".join(sorted(ENV_KEYS)))
    parser.add_argument("--model", default=None,
                        help="Model id or alias (e.g. claude-fable-5, deepseek-chat, fable5)")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="Output token cap; must cover reasoning tokens on reasoning models")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature; omit for reasoning models (rejected by Claude)")
    parser.add_argument("--effort", default=None,
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="Reasoning effort (Anthropic output_config.effort / OpenAI reasoning_effort)")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallel in-flight requests (default: %(default)s = sequential). "
                             "Results are still written in sample order, so resume stays valid.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N samples — use for a smoke test before a full run")
    return parser
