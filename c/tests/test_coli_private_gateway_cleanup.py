"""Private CLI gateways own loading engines as well as ready engines."""
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

C_DIR = Path(__file__).resolve().parents[1]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@unittest.skipIf(os.name == "nt", "POSIX session and signal ownership")
class PrivateGatewayCleanupTests(unittest.TestCase):
    def exercise(self, command="run", ready=False, ignore_term=False):
        if command == "chat":
            # --no-attach must never connect the test to an existing server.
            with socket.socket() as probe:
                try:
                    probe.bind(("127.0.0.1", 8000))
                except OSError:
                    self.skipTest("private chat port 8000 already in use")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "qwen4_exp",
                                                         "max_position_embeddings": 1024}))
            (model / "tokenizer.json").write_text("{}")
            pidfile, release, drained = (root / name for name in ("engine.json", "release", "drained"))
            engine = root / "fixture-engine"
            code = ("#!" + sys.executable + "\nimport os,time,pathlib,json,signal,sys\n"
                    "pid=pathlib.Path(os.environ['OWN_PID']);pending=pid.with_suffix('.tmp')\n"
                    "pending.write_text(json.dumps({'pid':os.getpid(),'group':os.getpgrp()}));pending.replace(pid)\n")
            if ignore_term:
                code += "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            if ready:
                code += ("sys.stdout.buffer.write(b'\\x01\\x01READY\\x01\\x01\\nSTAT 0 0 0 0\\n');sys.stdout.buffer.flush()\n"
                         "sys.stdin.buffer.read()\npathlib.Path(os.environ['OWN_DRAIN']).write_text('stdin EOF')\n")
            else:
                code += ("release=pathlib.Path(os.environ['OWN_RELEASE']);deadline=time.monotonic()+25\n"
                         "while not release.exists() and time.monotonic()<deadline:time.sleep(.02)\n")
            engine.write_text(code)
            engine.chmod(0o700)
            environment = {name: os.environ[name] for name in ("PATH", "TMPDIR", "TMP", "TEMP",
                                                               "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
                           if name in os.environ}
            environment.update(COLI_ENGINE=str(engine), COLI_NO_OMP_TUNE="1",
                               HOME=temporary, USERPROFILE=temporary, OWN_PID=str(pidfile),
                               OWN_RELEASE=str(release), OWN_DRAIN=str(drained),
                               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            args = [sys.executable, str(C_DIR / "coli"), command, "--model", str(model)]
            args += ["--no-attach"] if command == "chat" else ["fixture"]
            outside = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(40)"],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            child = None
            with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
                process = subprocess.Popen(args, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr)
                try:
                    deadline = time.monotonic() + 10
                    while not pidfile.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(pidfile.exists(), (root / "stderr.log").read_text())
                    info = json.loads(pidfile.read_text())
                    child = info["pid"]
                    if ready:
                        deadline = time.monotonic() + 10
                        healthy = False
                        while time.monotonic() < deadline:
                            match = re.search(r"listening on (http://127\.0\.0\.1:\d+)/v1",
                                              (root / "stderr.log").read_text())
                            if match:
                                try:
                                    with urllib.request.urlopen(match[1] + "/health", timeout=.2) as response:
                                        healthy = response.status == 200
                                except OSError:
                                    pass
                            if healthy:
                                break
                            time.sleep(.02)
                        self.assertTrue(healthy, "actual gateway did not finish its READY handshake: " + (root / "stderr.log").read_text())
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=22)
                    self.assertIsNone(outside.poll(), "cleanup signalled a process outside its owned gateway")
                    deadline = time.monotonic() + 1
                    while alive(child) and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertFalse(alive(child), "cancelled CLI left its loading engine running")
                    if ready:
                        self.assertEqual(drained.read_text(), "stdin EOF",
                                         "ready engines must retain their graceful EOF teardown")
                finally:
                    release.touch()
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                    outside.terminate()
                    outside.wait(timeout=5)
                    if child:
                        deadline = time.monotonic() + 3
                        while alive(child) and time.monotonic() < deadline:
                            time.sleep(.02)
                        if alive(child):
                            os.kill(child, signal.SIGKILL)

    def test_cancelled_run_stops_loading_engine(self):
        self.exercise()

    def test_cancelled_chat_stops_loading_engine(self):
        self.exercise("chat")

    def test_ready_gateway_retains_engine_eof_drain(self):
        self.exercise(ready=True)

    def test_loading_engine_ignoring_term_is_stopped(self):
        self.exercise(ignore_term=True)
