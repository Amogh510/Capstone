#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import json
import shutil
import threading
from dataclasses import dataclass
from typing import Optional

# Lightweight progress bar without extra deps
class ProgressBar:
    def __init__(self, title: str, total: int = 100):
        self.title = title
        self.total = total
        self.current = 0
        self._lock = threading.Lock()
        self._last_print_len = 0

    def update(self, value: int):
        with self._lock:
            self.current = max(0, min(self.total, value))
            self._print()

    def tick(self, inc: int = 1):
        self.update(self.current + inc)

    def done(self):
        self.update(self.total)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _print(self):
        width = 40
        filled = int((self.current / self.total) * width)
        bar = "#" * filled + "-" * (width - filled)
        line = f"{self.title} [{bar}] {self.current}/{self.total}\r"
        sys.stdout.write(line)
        pad = max(0, self._last_print_len - len(line))
        if pad:
            sys.stdout.write(" " * pad)
        self._last_print_len = len(line)
        sys.stdout.flush()

@dataclass
class Neo4jConfig:
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "admin123")

ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(ROOT, "src")
NEO4J_IMPORTER_DIR = os.path.join(ROOT, "neo4j-importer")
RETRIEVAL_DIR = os.path.join(ROOT, "retrieval-service")
CONFIG_JS_PATH = os.path.join(SRC_DIR, "config.js")
KG_JSON_PATH = os.path.join(SRC_DIR, "output", "kg.json")


def check_cmd(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: list[str], env: Optional[dict] = None, cwd: Optional[str] = None) -> int:
    proc = subprocess.Popen(cmd, cwd=cwd, env=env or os.environ.copy())
    return proc.wait()


def run_capture(cmd: list[str], env: Optional[dict] = None, cwd: Optional[str] = None) -> tuple[int, str, str]:
    proc = subprocess.Popen(cmd, cwd=cwd, env=env or os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def ensure_dependencies():
    steps = [
        ("node", "Node.js"),
        ("npm", "npm"),
        ("go", "Go"),
        ("python3", "Python 3"),
    ]
    missing = [name for cmd, name in steps if not check_cmd(cmd)]
    if missing:
        raise RuntimeError(f"Missing required tools: {', '.join(missing)}")

    # Create/ensure venv for retrieval-service and install deps inside
    venv_dir = os.path.join(RETRIEVAL_DIR, ".venv")
    venv_bin = os.path.join(venv_dir, "bin")
    venv_py = os.path.join(venv_bin, "python")
    venv_pip = os.path.join(venv_bin, "pip")
    if sys.platform.startswith("win"):
        venv_bin = os.path.join(venv_dir, "Scripts")
        venv_py = os.path.join(venv_bin, "python.exe")
        venv_pip = os.path.join(venv_bin, "pip.exe")

    if not os.path.exists(venv_dir):
        rc, _, err = run_capture(["python3", "-m", "venv", ".venv"], cwd=RETRIEVAL_DIR)
        if rc != 0:
            raise RuntimeError(f"Failed to create virtualenv: {err}")

    rc, _, err = run_capture([venv_py, "-m", "pip", "install", "-r", "requirements.txt", "--disable-pip-version-check"], cwd=RETRIEVAL_DIR)
    if rc != 0:
        raise RuntimeError(f"pip install in venv failed: {err}")

    # Go deps for neo4j-importer
    rc, _, err = run_capture(["go", "mod", "download"], cwd=NEO4J_IMPORTER_DIR)
    if rc != 0:
        raise RuntimeError(f"go mod download failed: {err}")

    # Node deps for src (none declared, but ensure package.json at root)
    if os.path.exists(os.path.join(ROOT, "package.json")):
        rc, _, err = run_capture(["npm", "ci"], cwd=ROOT)
        if rc != 0:
            # Fallback to npm install
            rc2, _, err2 = run_capture(["npm", "install"], cwd=ROOT)
            if rc2 != 0:
                raise RuntimeError(f"npm install failed: {err}\n{err2}")


def update_config_project_root(project_root: str):
    if not os.path.exists(CONFIG_JS_PATH):
        raise FileNotFoundError(f"config.js not found at {CONFIG_JS_PATH}")
    with open(CONFIG_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    # Replace the projectRoot line conservatively
    import re
    new_line = f"  projectRoot: path.resolve(__dirname, '..', '{project_root}'), // IMPORTANT: Change this to the actual project root"
    
    # Look for existing projectRoot line (proper property)
    if re.search(r"^\s*projectRoot\s*:", content, flags=re.M):
        # Replace the existing projectRoot line
        content2 = re.sub(r"^\s*projectRoot\s*:.*$", new_line, content, flags=re.M)
    else:
        # No proper projectRoot found, insert after the comment line
        comment_pattern = r"(\s*// Define the root directory of the project to be analyzed\s*\n)"
        if re.search(comment_pattern, content):
            content2 = re.sub(comment_pattern, f"\\1{new_line}\n", content)
        else:
            # Fallback: insert after opening brace
            content2 = content.replace('module.exports = {', f"module.exports = {{\n  // Define the root directory of the project to be analyzed\n{new_line}")
    
    with open(CONFIG_JS_PATH, "w", encoding="utf-8") as f:
        f.write(content2)


def run_node_analyzer() -> None:
    bar = ProgressBar("Analyzing source (Node)", total=100)
    bar.update(5)
    # src/index.js is an executable script that writes output/*.json
    # We stream logs to parse coarse progress
    proc = subprocess.Popen(["node", "index.js"], cwd=SRC_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    progressed = 5
    try:
        for line in proc.stdout:  # type: ignore[arg-type]
            sys.stdout.write(line)
            sys.stdout.flush()
            # Heuristic progress bumps
            if "Components found" in line:
                progressed = min(70, progressed + 1)
                bar.update(progressed)
            elif "Style file (CSS analysis)" in line:
                progressed = min(80, progressed + 1)
                bar.update(progressed)
            elif "Analysis complete" in line:
                bar.update(95)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("Node analysis failed")
        bar.done()
    finally:
        try:
            proc.stdout.close()  # type: ignore[union-attr]
        except Exception:
            pass

    if not os.path.exists(KG_JSON_PATH):
        raise FileNotFoundError(f"Expected KG file not found at {KG_JSON_PATH}")


def clean_neo4j(cfg: Neo4jConfig):
    bar = ProgressBar("Cleaning Neo4j", total=3)
    bar.update(1)
    # Use cypher-shell if available; otherwise, use a tiny Go one-shot? Prefer cypher-shell.
    if not check_cmd("cypher-shell"):
        # Fallback: use the retrieval-service venv python to run cleanup via neo4j driver
        venv_dir = os.path.join(RETRIEVAL_DIR, ".venv")
        venv_py = os.path.join(venv_dir, "bin", "python")
        if sys.platform.startswith("win"):
            venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
        if not os.path.exists(venv_py):
            raise RuntimeError("cypher-shell missing and retrieval-service venv python not found.")
        script = (
            "from neo4j import GraphDatabase; import os; "
            "uri=os.environ['NEO4J_URI']; u=os.environ['NEO4J_USER']; p=os.environ['NEO4J_PASSWORD']; "
            "drv=GraphDatabase.driver(uri, auth=(u,p)); "
            "with drv.session() as s: s.run('MATCH (n) DETACH DELETE n'); "
            "try:\n    s.run(\"CALL db.indexes() YIELD name WHERE name STARTS WITH 'kg_' CALL db.index.drop(name) YIELD name as dropped RETURN dropped\");\n"
            "except Exception: pass; drv.close(); print('DB cleaned')"
        )
        env = os.environ.copy()
        env.update({"NEO4J_URI": cfg.uri, "NEO4J_USER": cfg.user, "NEO4J_PASSWORD": cfg.password})
        rc, out, err = run_capture([venv_py, "-c", script], env=env)
        if rc != 0:
            raise RuntimeError(f"Neo4j clean failed: {err or out}")
        bar.update(3)
        bar.done()
        return

    env = os.environ.copy()
    env.update({"NEO4J_URI": cfg.uri, "NEO4J_USER": cfg.user, "NEO4J_PASSWORD": cfg.password})
    # cypher-shell uses NEO4J_USER and NEO4J_PASSWORD; URI via --address
    rc, out, err = run_capture([
        "cypher-shell",
        "--address", cfg.uri.replace("bolt://", "neo4j://"),
        "-u", cfg.user,
        "-p", cfg.password,
        "MATCH (n) DETACH DELETE n;"
    ], env=env)
    if rc != 0:
        raise RuntimeError(f"Neo4j clean failed: {err or out}")
    bar.update(3)
    bar.done()


def run_go_importer(cfg: Neo4jConfig):
    bar = ProgressBar("Importing KG to Neo4j", total=100)
    env = os.environ.copy()
    env.update({
        "NEO4J_URI": cfg.uri,
        "NEO4J_USER": cfg.user,
        "NEO4J_PASSWORD": cfg.password,
        "KG_FILE": KG_JSON_PATH,
    })
    bar.update(5)
    proc = subprocess.Popen(["go", "run", "main.go"], cwd=NEO4J_IMPORTER_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    progressed = 5
    try:
        for line in proc.stdout:  # type: ignore[arg-type]
            sys.stdout.write(line)
            sys.stdout.flush()
            if "Processing nodes" in line:
                progressed = max(progressed, 10); bar.update(progressed)
            elif "Imported" in line and "nodes" in line:
                progressed = min(70, progressed + 5); bar.update(progressed)
            elif "Processing edges" in line:
                progressed = max(progressed, 75); bar.update(progressed)
            elif "Imported" in line and "edges" in line:
                progressed = min(95, progressed + 3); bar.update(progressed)
            elif "Import completed successfully" in line:
                bar.update(100)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError("Go importer failed")
        bar.done()
    finally:
        try:
            proc.stdout.close()  # type: ignore[union-attr]
        except Exception:
            pass


def start_retrieval_service(cfg: Neo4jConfig):
    bar = ProgressBar("Starting retrieval service", total=100)
    env = os.environ.copy()
    env.update({
        "NEO4J_URI": cfg.uri,
        "NEO4J_USER": cfg.user,
        "NEO4J_PASSWORD": cfg.password,
        "API_HOST": "0.0.0.0",
        "API_PORT": "8000",
    })
    # Run in background
    venv_dir = os.path.join(RETRIEVAL_DIR, ".venv")
    venv_py = os.path.join(venv_dir, "bin", "python")
    if sys.platform.startswith("win"):
        venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
    proc = subprocess.Popen([venv_py, "main.py"], cwd=RETRIEVAL_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def _log_stream():
        for ln in proc.stdout:  # type: ignore[arg-type]
            # Print but don't flood; user can see logs if needed
            if "Uvicorn" in ln or "INFO" in ln:
                pass
            sys.stdout.write(ln)
            sys.stdout.flush()

    threading.Thread(target=_log_stream, daemon=True).start()

    # poll health
    import urllib.request
    import urllib.error
    base = f"http://127.0.0.1:8000"
    started = False
    for i in range(100):
        bar.update(i)
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    started = True
                    break
        except Exception:
            time.sleep(0.3)
    bar.done()
    if not started:
        raise RuntimeError("Retrieval service failed to become healthy in time")
    return proc


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run_pipeline.py /absolute/path/to/your/codebase/src")
        sys.exit(1)

    project_root = sys.argv[1]
    if not os.path.isabs(project_root):
        project_root = os.path.abspath(project_root)
    if not os.path.exists(project_root):
        print(f"Provided path does not exist: {project_root}")
        sys.exit(2)

    cfg = Neo4jConfig()

    print("Checking dependencies and installing as needed...")
    ensure_dependencies()

    print(f"Updating src/config.js projectRoot -> {project_root}")
    update_config_project_root(project_root)

    print("Running Node analyzer to generate KG JSON...")
    run_node_analyzer()

    print("Cleaning Neo4j database (fresh import)...")
    clean_neo4j(cfg)

    print("Importing KG into Neo4j (Go importer)...")
    run_go_importer(cfg)

    print("Starting retrieval service and warming up...")
    proc = start_retrieval_service(cfg)

    print("All set! Retrieval service is running at http://localhost:8000")
    print("Press Ctrl+C to stop the retrieval service.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping retrieval service...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
