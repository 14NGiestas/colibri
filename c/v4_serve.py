import http.server, json, subprocess, os, signal, sys, time, re

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "deepseek_v4")
MODEL = os.environ.get("COLI_MODEL", os.path.expanduser("~/cheesyham/output/deepseek-v4-flash-dspark"))
OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS", "8")
MEMORY_GB = os.environ.get("MEMORY_GB", "14")

class V4Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try: req = json.loads(body)
        except: self.send_error(400, "bad json"); return
        prompt = req.get("prompt", req.get("messages", [{}])[-1].get("content", ""))
        if not prompt: self.send_error(400, "no prompt"); return
        ngen = min(req.get("max_tokens", 64), 512)
        cmd = [ENGINE, MODEL, prompt, "--max-tokens", str(ngen), "--memory-gb", MEMORY_GB]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = OMP_NUM_THREADS
        for k in ["NIX_ENFORCE_NO_NATIVE", "COLI_MODEL"]: env.pop(k, None)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
        elapsed = time.time() - t0
        text = r.stdout.strip()
        m = re.search(r'generated_text=(.*)', text)
        if m: text = m.group(1).strip()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        tokens = 0
        for line in r.stderr.split("\n"):
            if line.startswith("v4_tokens"):
                for part in line.split():
                    if part.startswith("generated="): tokens = int(part.split("=")[1])
        resp = {"response": text, "time_s": round(elapsed, 2)}
        if tokens: resp["tokens"] = tokens; resp["tok_s"] = round(tokens/elapsed, 3)
        self.wfile.write(json.dumps(resp).encode())
    def log_message(self, *a): pass

PORT = 11436
s = http.server.HTTPServer(("0.0.0.0", PORT), V4Handler)
print(f"[v4-serve] http://0.0.0.0:{PORT} | model={MODEL}", flush=True)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
s.serve_forever()
