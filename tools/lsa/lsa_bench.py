"""End-to-end check of Maya's GLM-5.3-Flash with and without Latent Sparse Attention (--lsa).

Starts Maya's server (serve/server.py --engine maya) on one config, waits for the model, then runs:
  1. short prompts (coherence, thinking off, temperature 0) and a decode-speed prompt;
  2. a long-context needle test: Maya's own docs as the haystack (~N tokens), three secret codes planted at 15 / 50 /
     85 % of it, one question per code - a wrong or missing code means the sparse selection lost the needle;
  3. the engine's own log lines (glm lsa / glm vram: prompt / glm stat: decode) from the part of the log this run wrote.
Then stops the server and writes one JSON with everything.  The prompts are Polish (the author's language); the codes
and the checks do not depend on it.

Usage (from the repository root, with Maya's .venv or any Python 3.10+; the server runs under Maya's .venv):
    python tools/lsa/lsa_bench.py --config maya-glm.json --tag native [--port 8084] [--prompt-tokens 40000]
    python tools/lsa/lsa_bench.py --config maya-glm-lsa.json --tag lsa
    python tools/lsa/lsa_bench.py --compare bench_native.json bench_lsa.json
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

MAYA = Path(__file__).resolve().parents[2]
PY = MAYA / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
SERVER = MAYA / "serve" / "server.py"
OUT_DIR = Path.cwd()

SHORT = [
    ("pl", "Wyjaśnij w trzech zdaniach, czym jest fotosynteza."),
    ("math", "Ile to 17 * 23? Odpowiedz samą liczbą, a potem jednym zdaniem pokaż rachunek."),
    ("code", "Napisz funkcję w Pythonie, która zwraca n-tą liczbę Fibonacciego iteracyjnie. Tylko kod."),
    ("en", "List four differences between TCP and UDP, one line each."),
]
STORY = "Napisz krótkie opowiadanie (około 250 słów) o latarniku, który pewnej nocy zobaczył na morzu światło, " \
        "którego nie było na żadnej mapie."

WORDS = ["malachit", "kompas", "żuraw", "bursztyn", "sekstant", "latarnia", "granit", "modrzew", "szafir", "wrzos",
         "kotwica", "sopel", "agrest", "fiord", "lampion", "cynamon"]


def http(port: int, path: str, body=None, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def chat(port: int, prompt: str, max_tokens: int, thinking=False, temperature=0.0) -> dict:
    t0 = time.time()
    out = http(port, "/v1/chat/completions", {
        "model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "stream": False,
        "chat_template_kwargs": {"enable_thinking": bool(thinking)}})
    el = time.time() - t0
    msg = out["choices"][0]["message"]
    return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "finish": out["choices"][0].get("finish_reason"), "usage": out.get("usage"), "timings": out.get("timings"),
            "wall_s": round(el, 1)}


def n_tokens(port: int, text: str) -> int:
    r = http(port, "/tokenize", {"content": text})
    toks = r.get("tokens") or r.get("ids") or []
    return len(toks)


def haystack(port: int, target_tokens: int) -> str:
    """Maya's own docs (README + docs/*.md, a fixed order) cut to about target_tokens tokens."""
    files = sorted(MAYA.glob("README*.md")) + sorted((MAYA / "docs").glob("*.md"))
    text = ""
    for f in files:
        try:
            text += f"\n\n----- {f.name} -----\n\n" + f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > target_tokens * 6:
            break
    # binary search on characters for the token target (one /tokenize call per step)
    lo, hi = 0, len(text)
    while hi - lo > 2000:
        mid = (lo + hi) // 2
        if n_tokens(port, text[:mid]) < target_tokens:
            lo = mid
        else:
            hi = mid
    return text[:hi]


def plant(text: str, needles: list) -> str:
    """Each needle (frac, sentence) goes in at the paragraph break nearest frac of the text."""
    out = text
    for frac, sentence in sorted(needles, reverse=True):
        at = int(len(out) * frac)
        cut = out.rfind("\n\n", 0, at)
        if cut < 0:
            cut = at
        out = out[:cut] + "\n\n" + sentence + "\n\n" + out[cut:]
    return out


def start_server(cfg: Path, port: int, lines: list) -> subprocess.Popen:
    p = subprocess.Popen([str(PY), str(SERVER), "--engine", "maya", "--config", str(cfg), "--port", str(port)],
                         cwd=str(MAYA), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)

    def pump():
        for line in p.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            print("  | " + line, flush=True)
    threading.Thread(target=pump, daemon=True).start()
    return p


def stop_server(p: subprocess.Popen):
    try:
        p.stdin.close()
    except OSError:
        pass
    try:
        p.terminate()
        p.wait(20)
    except (OSError, subprocess.TimeoutExpired):
        pass
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    else:
        try:
            p.kill()
        except OSError:
            pass
    time.sleep(2)


def wait_ready(port: int, p: subprocess.Popen, timeout_s: float) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:
            raise RuntimeError(f"server exited with {p.returncode} while loading")
        try:
            h = http(port, "/health", timeout=5)
            if h.get("status") == "ok":
                return time.time() - t0
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(3)
    raise RuntimeError("model not ready in time")


def log_tail(log: Path, offset: int) -> list:
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []


def run(a) -> int:
    cfg = Path(a.config).resolve()
    c = json.loads(cfg.read_text(encoding="utf-8-sig"))
    log = Path(c.get("log") or cfg.with_suffix(".log"))
    off = log.stat().st_size if log.exists() else 0
    lines = []
    print(f"== {a.tag}: starting {cfg.name} on port {a.port} (engine log {log})", flush=True)
    p = start_server(cfg, a.port, lines)
    res = {"tag": a.tag, "config": cfg.name, "args": c.get("args"), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        res["load_s"] = round(wait_ready(a.port, p, a.load_timeout), 1)
        print(f"== ready after {res['load_s']} s", flush=True)
        res["short"] = []
        for name, q in SHORT:
            r = chat(a.port, q, 200)
            res["short"].append({"name": name, "prompt": q, **r})
            print(f"-- {name}: {r['wall_s']} s, {(r['timings'] or {}).get('predicted_per_second')} tok/s\n{r['text'][:400]}\n",
                  flush=True)
        r = chat(a.port, STORY, 384)
        res["story"] = r
        print(f"-- story: {(r['timings'] or {}).get('predicted_per_second')} tok/s decode, "
              f"{r['usage'] and r['usage'].get('completion_tokens')} tokens\n{r['text'][:600]}\n", flush=True)
        if not a.skip_long:
            rnd = random.Random(a.seed)
            codes = {k: rnd.choice(WORDS) + "-" + str(rnd.randint(100, 999)) for k in ("alfa", "beta", "gamma")}
            needles = [(0.15, f"Zapamiętaj: tajny kod procedury ALFA to {codes['alfa']}."),
                       (0.50, f"Zapamiętaj: tajny kod procedury BETA to {codes['beta']}."),
                       (0.85, f"Zapamiętaj: tajny kod procedury GAMMA to {codes['gamma']}.")]
            hay = plant(haystack(a.port, a.prompt_tokens), needles)
            res["needle"] = {"codes": codes, "prompt_tokens_est": n_tokens(a.port, hay), "answers": []}
            print(f"== needle: haystack ~{res['needle']['prompt_tokens_est']} tokens, codes {codes}", flush=True)
            for i, (name, code) in enumerate(codes.items()):
                q = hay + f"\n\nPytanie: jaki jest tajny kod procedury {name.upper()}? Odpowiedz samym kodem."
                r = chat(a.port, q, 48)
                ok = code.lower() in (r["text"] or "").lower()
                res["needle"]["answers"].append({"name": name, "code": code, "ok": ok, **r})
                t = r["timings"] or {}
                print(f"-- {name}: {'OK' if ok else 'MISS'} -> {r['text'].strip()[:80]!r}  "
                      f"prompt {t.get('prompt_n')} tok @ {t.get('prompt_per_second')} tok/s, cache {t.get('cache_n')}, "
                      f"decode {t.get('predicted_per_second')} tok/s, wall {r['wall_s']} s", flush=True)
            # decode speed at the long context: a short essay after the haystack (the cache holds the haystack)
            q = hay + "\n\nPytanie: streść w pięciu zdaniach, o czym są te dokumenty."
            r = chat(a.port, q, 256)
            res["needle"]["long_decode"] = r
            t = r["timings"] or {}
            print(f"-- long decode: {t.get('predicted_per_second')} tok/s ({t.get('predicted_n')} tokens)\n{r['text'][:500]}\n",
                  flush=True)
        try:
            res["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                                        capture_output=True, text=True).stdout.strip()
        except OSError:
            res["gpu"] = None
    except Exception as e:                       # noqa: BLE001 - report whatever happened, then stop the server
        res["error"] = f"{type(e).__name__}: {e}"
        print(f"!! {res['error']}", flush=True)
    finally:
        stop_server(p)
    tail = log_tail(log, off)
    res["engine_log"] = {
        "lsa": [l for l in tail if "glm lsa" in l],
        "prompt": [l for l in tail if "glm vram: prompt" in l],
        "decode": [l for l in tail if "glm stat: decode" in l][-6:],
        "errors": [l for l in tail if re.search(r"error|fail|abort|assert|illegal|invalid", l, re.I)
                   and "no error" not in l and "glm lsa" not in l][:20],
        "lines": len(tail)}
    res["server_out"] = lines[-40:]
    out = OUT_DIR / f"bench_{a.tag}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"== wrote {out}")
    for k in ("lsa", "prompt", "decode"):
        for l in res["engine_log"][k]:
            print("   " + l)
    if res["engine_log"]["errors"]:
        print("   engine log lines mentioning errors:")
        for l in res["engine_log"]["errors"]:
            print("   " + l)
    return 1 if res.get("error") else 0


def compare(paths: list) -> int:
    rs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    print(f"{'':28}" + "".join(f"{r['tag']:>16}" for r in rs))

    def row(label, f):
        print(f"{label:28}" + "".join(f"{str(f(r)):>16}" for r in rs))
    row("load s", lambda r: r.get("load_s"))
    row("short decode tok/s (avg)", lambda r: round(sum((x["timings"] or {}).get("predicted_per_second") or 0
                                                       for x in r.get("short", [])) / max(1, len(r.get("short", []))), 1))
    row("story decode tok/s", lambda r: ((r.get("story") or {}).get("timings") or {}).get("predicted_per_second"))
    row("needle prompt tokens", lambda r: (r.get("needle") or {}).get("prompt_tokens_est"))
    row("needle prefill tok/s", lambda r: ((r.get("needle") or {}).get("answers") or [{}])[0].get("timings", {}) and
        ((r.get("needle") or {}).get("answers") or [{}])[0]["timings"].get("prompt_per_second"))
    row("needle hits", lambda r: f"{sum(x['ok'] for x in (r.get('needle') or {}).get('answers', []))}/"
                                 f"{len((r.get('needle') or {}).get('answers', []))}")
    row("long decode tok/s", lambda r: (((r.get("needle") or {}).get("long_decode") or {}).get("timings") or {})
        .get("predicted_per_second"))
    row("error", lambda r: r.get("error"))
    for r in rs:
        for x in r.get("short", []):
            print(f"\n[{r['tag']}] {x['name']}: {x['text'].strip()[:300]!r}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--port", type=int, default=8084)
    ap.add_argument("--prompt-tokens", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--load-timeout", type=float, default=1800)
    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--compare", nargs="+")
    a = ap.parse_args()
    if a.compare:
        return compare(a.compare)
    if not a.config:
        ap.error("--config or --compare")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
