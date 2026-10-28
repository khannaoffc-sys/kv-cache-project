"""ask.py - type a prompt, see how the model answers under each KV-cache setup.

It runs the SAME prompt through four setups and prints each answer with time, speed,
KV-cache size and (on a GPU) measured board power and energy:
  baseline  no compression
  snapkv    evict tokens with SnapKV (needs a prompt longer than 64 tokens)
  knorm     evict tokens with the smallest key norms (what the FPGA unit does)
  hqq       keep every token but store the cache in 4 bits

Run from the project root, with the venv active on the laptop:
  python src/ask.py                         # asks you to type a prompt
  python src/ask.py --prompt "Explain the KV cache in two sentences."
  python src/ask.py --demo                  # built-in long prompt, so every setup works
  python src/ask.py --configs baseline,knorm --ratio 0.7
On Kaggle (GPU) use the same flags with  !python src/ask.py --demo
"""
import argparse
import sys
import threading
import time
from contextlib import nullcontext

DEMO_PROMPT = (
    "Read the passage and answer the question in one sentence.\n\n"
    "Passage: When a language model writes text, it produces one token at a time. To decide "
    "the next token, every layer of the model needs the key and value vectors of all earlier "
    "tokens. Recomputing them at every step would be very slow, so the model stores them in "
    "memory. This store is called the KV cache. The cache grows by one entry for every new "
    "token, so a long prompt or a long answer makes it large. On a small device such as an "
    "edge board, this memory is limited, and reading it again at every step also costs time "
    "and energy. Researchers therefore shrink the cache in two main ways. Eviction methods "
    "delete the entries of tokens that seem unimportant, for example the tokens whose key "
    "vectors have a small norm. Quantization methods keep every token but store each number "
    "with fewer bits, for example four instead of sixteen.\n\n"
    "Question: Why do researchers try to shrink the KV cache?"
)


class PowerSampler:
    """Reads GPU board power through NVML in a background thread.
    If NVML or a GPU is missing it quietly does nothing (power shows as n/a)."""

    def __init__(self, interval=0.05):
        self.ok = False
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml.nvmlDeviceGetPowerUsage(self._handle)  # test read
            self.ok = True
        except Exception:
            self.ok = False

    def _run(self):
        while not self._stop.is_set():
            try:
                watts = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self.samples.append((time.perf_counter(), watts))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if not self.ok:
            return
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, t0, t1):
        """Return (average watts, energy in joules) for the window [t0, t1], or (None, None)."""
        if not self.ok or self._thread is None:
            return None, None
        self._stop.set()
        self._thread.join()
        pts = [(t, w) for t, w in self.samples if t0 <= t <= t1]
        if len(pts) < 2:
            return None, None
        energy = sum((pts[i + 1][0] - pts[i][0]) * (pts[i + 1][1] + pts[i][1]) / 2
                     for i in range(len(pts) - 1))
        span = pts[-1][0] - pts[0][0]
        return (energy / span if span > 0 else None), energy


def cache_info(pkv):
    """(tokens kept in the cache, cache size in bytes), or (None, None) if unreadable."""
    try:
        layers = pkv.layers
        tokens = layers[0].keys.shape[-2]
        nbytes = sum(l.keys.numel() * l.keys.element_size()
                     + l.values.numel() * l.values.element_size() for l in layers)
        return tokens, nbytes
    except Exception:
        return None, None


def fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


def run_config(name, model, tok, inputs, args, sampler, device):
    import torch
    kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False,
                  return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
    ctx = nullcontext()
    if name == "snapkv":
        from kvpress import SnapKVPress
        ctx = SnapKVPress(compression_ratio=args.ratio)(model)
    elif name == "knorm":
        from kvpress import KnormPress
        ctx = KnormPress(compression_ratio=args.ratio)(model)
    elif name == "hqq":
        from transformers import QuantizedCache
        kwargs["past_key_values"] = QuantizedCache(backend="hqq", config=model.config, nbits=4)

    if device == "cuda":
        torch.cuda.synchronize()
    sampler.start()
    t0 = time.perf_counter()
    with ctx:
        out = model.generate(**inputs, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    avg_w, energy = sampler.stop(t0, t1)

    new_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(new_ids, skip_special_tokens=True).strip()
    latency = t1 - t0
    if name == "hqq":
        tokens, nbytes = None, None  # quantized storage is not measured here
    else:
        tokens, nbytes = cache_info(getattr(out, "past_key_values", None))
    return dict(config=name, text=text, latency=latency,
                tps=len(new_ids) / latency if latency > 0 else None,
                cache_tokens=tokens, cache_mb=None if nbytes is None else nbytes / 1e6,
                watts=avg_w, joules=energy)


def answer_prompt(prompt, model, tok, args, sampler, device):
    messages = [{"role": "user", "content": prompt}]
    chat_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat_text, return_tensors="pt").to(device)
    n_prompt = inputs["input_ids"].shape[1]
    print(f"\nPrompt length: {n_prompt} tokens (SnapKV needs more than 64).")

    results = []
    for name in args.configs.split(","):
        name = name.strip()
        print(f"  running {name} ...", flush=True)
        try:
            results.append(run_config(name, model, tok, inputs, args, sampler, device))
        except Exception as exc:  # keep going if one setup fails
            print(f"  [{name}] could not run: {type(exc).__name__}: {exc}")
            results.append(dict(config=name, text=f"(failed: {type(exc).__name__}: {exc})",
                                latency=None, tps=None, cache_tokens=None, cache_mb=None,
                                watts=None, joules=None))

    print("\n" + "=" * 78)
    print(f"{'setup':<10}{'time s':>9}{'tok/s':>8}{'cache tok':>11}{'cache MB':>10}"
          f"{'avg W':>8}{'energy J':>10}")
    for r in results:
        print(f"{r['config']:<10}{fmt(r['latency'], '.2f'):>9}{fmt(r['tps'], '.1f'):>8}"
              f"{fmt(r['cache_tokens'], 'd'):>11}{fmt(r['cache_mb'], '.1f'):>10}"
              f"{fmt(r['watts'], '.1f'):>8}{fmt(r['joules'], '.1f'):>10}")
    print("=" * 78)
    for r in results:
        print(f"\n--- {r['config']} ---\n{r['text']}")
    print("\nNote: one run per setup, so small time differences are noise. "
          "Power is cloud GPU board power, not edge energy.")


def main():
    ap = argparse.ArgumentParser(description="Ask the model a prompt under different KV-cache setups")
    ap.add_argument("--prompt", help="the prompt text")
    ap.add_argument("--file", help="read the prompt from a text file")
    ap.add_argument("--demo", action="store_true", help="use the built-in long prompt")
    ap.add_argument("--model", help="model name (default: 1.5B on a GPU, 0.5B on the laptop)")
    ap.add_argument("--configs", default="baseline,snapkv,knorm,hqq",
                    help="comma list from: baseline,snapkv,knorm,hqq")
    ap.add_argument("--ratio", type=float, default=0.5, help="fraction of prompt tokens to evict")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = args.model or ("Qwen/Qwen2.5-1.5B-Instruct" if device == "cuda"
                                else "Qwen/Qwen2.5-0.5B-Instruct")
    if device == "cpu" and any(s in model_name for s in ("1.5B", "3B")):
        sys.exit("Do not run this model on the laptop CPU (it swaps to disk). Use the 0.5B model "
                 "here, or run on Kaggle with a GPU.")
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading {model_name} on {device} ({dtype}) ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()

    warm = tok("Hello", return_tensors="pt").to(device)  # first call is always slower
    model.generate(**warm, max_new_tokens=2, pad_token_id=tok.eos_token_id)
    sampler = PowerSampler()
    print("GPU power reading:", "on" if sampler.ok else "off (n/a)")

    if args.demo:
        answer_prompt(DEMO_PROMPT, model, tok, args, sampler, device)
    elif args.file:
        answer_prompt(open(args.file, encoding="utf-8").read(), model, tok, args, sampler, device)
    elif args.prompt:
        answer_prompt(args.prompt, model, tok, args, sampler, device)
    else:
        while True:
            try:
                prompt = input("\nType a prompt (empty line to quit): ").strip()
            except EOFError:
                break
            if not prompt:
                break
            answer_prompt(prompt, model, tok, args, sampler, device)


if __name__ == "__main__":
    main()
