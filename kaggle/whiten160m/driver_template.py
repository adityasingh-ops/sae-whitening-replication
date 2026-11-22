"""Pythia-160M layer 8, TopK k=32 x8: plain vs whitened, seeds 1-3, 30M tokens; then eval.
Plain runs on GPU 0 and whitened runs on GPU 1 in parallel."""
import base64, os, subprocess, sys, threading, time
from pathlib import Path

W = Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sae-lens", "zstandard"], check=True)
FILES = __FILES__
sd = W / "whitening" / "scripts"
sd.mkdir(parents=True, exist_ok=True)
for name, b in FILES.items():
    (sd / name).write_bytes(base64.b64decode(b))

COMMON = ["--model", "EleutherAI/pythia-160m-deduped", "--layer", "8", "--d-in", "768",
          "--arch", "topk", "--k", "32", "--tokens", "__TOKENS__"]
log_lock = threading.Lock()

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with log_lock:
        print(line, flush=True)
        with open(W / "progress.log", "a") as f:
            f.write(line + "\n")

def run(cmd, gpu, tag, attempts=3):
    """Run cmd; on failure retry (train_sae.py resumes from its latest checkpoint)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for attempt in range(1, attempts + 1):
        log(f"start {tag} on gpu {gpu} (attempt {attempt})")
        with open(W / f"{tag}.log", "a") as f:
            r = subprocess.run(cmd, cwd=W, env=env, stdout=f, stderr=subprocess.STDOUT)
        tail = (W / f"{tag}.log").read_text().strip().splitlines()[-1:] or [""]
        log(f"end {tag} rc={r.returncode} :: {tail[0][:200]}")
        if r.returncode == 0:
            return
        time.sleep(60)
    log(f"gave up on {tag}")

def worker(gpu, whiten):
    for seed in __SEEDS__:
        tag = f"{'white' if whiten else 'plain'}_s{seed}"
        cmd = [sys.executable, "whitening/scripts/train_sae.py", *COMMON, "--seed", str(seed), "--device", "cuda",
               "--n-checkpoints", "3"]
        if whiten:
            cmd.append("--whiten")
        run(cmd, gpu, tag)

threads = [threading.Thread(target=worker, args=(0, False)), threading.Thread(target=worker, args=(1, True))]
for t in threads: t.start()
for t in threads: t.join()

runs = sorted(str(p) for p in (W / "whitening" / "results").glob("pythia-160m-deduped_L8_*_s*"))
run([sys.executable, "whitening/scripts/eval_sae.py", *runs, "--batches", "20", "--device", "cuda"], 0, "eval")
import shutil
for ckpt in (W / "whitening" / "results").glob("*/checkpoints"):
    shutil.rmtree(ckpt)  # keep the downloadable output small; final SAEs are kept
log("all done")
