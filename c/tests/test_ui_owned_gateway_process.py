"""A UI-owned exited gateway must not consume the whole stop grace period.

Linux-only native /proc witness. No model, existing gateway or /proc mock is used.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C))
spec = importlib.util.spec_from_file_location('owned_gateway_models', C.parent / 'ui/models.py')
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


@unittest.skipUnless(sys.platform.startswith('linux'), 'native Linux /proc witness')
class OwnedGatewayProcessTests(unittest.TestCase):
    def test_exited_owned_gateway_does_not_wait_for_its_zombie(self):
        with tempfile.TemporaryDirectory(prefix='coli-owned-gateway-') as raw:
            root = Path(raw)
            log = root / 'gateway.log'
            process = models.launch(
                [sys.executable, '-u', '-c',
                 'import time; print("READY", flush=True); time.sleep(120)'],
                str(root), {}, str(log),
            )
            pid = process.pid
            # Match the successful switch worker: it leaves no Popen in a registry.
            del process
            completed = threading.Event()
            errors, signals = [], []
            real_kill = os.kill
            stopper = None

            def state():
                try:
                    return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[0]
                except FileNotFoundError:
                    return None

            def signal_owned(target, sig):
                self.assertEqual(target, pid, 'refusing a signal to an unowned process')
                signals.append(int(sig))
                real_kill(target, sig)

            def stop():
                try:
                    models.stop_gateway({'pid': pid}, {})
                except BaseException as error:
                    errors.append(repr(error))
                finally:
                    completed.set()

            def reap_owned():
                # waitpid itself verifies parent ownership before any cleanup signal.
                try:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    return
                if reaped:
                    return
                real_kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    reaped, _ = os.waitpid(pid, os.WNOHANG)
                    if reaped:
                        return
                    time.sleep(0.01)
                real_kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)

            patcher = patch.object(models.os, 'kill', side_effect=signal_owned)
            patcher.start()
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if log.exists() and 'READY' in log.read_text():
                        break
                    time.sleep(0.01)
                else:
                    self.fail('owned stand-in did not become ready')
                stopper = threading.Thread(target=stop, daemon=True)
                began = time.monotonic()
                stopper.start()
                finished = completed.wait(3)
                witness = {'pid': pid, 'state': state(), 'stop_completed': finished,
                           'elapsed_seconds': time.monotonic() - began,
                           'errors': list(errors), 'signals': list(signals)}
                print('OWNED_GATEWAY_WITNESS ' + json.dumps(witness, sort_keys=True), flush=True)
                self.assertEqual(errors, [])
                self.assertEqual(signals, [int(signal.SIGTERM)])
                self.assertTrue(finished, 'stop_gateway still waits although its owned child is a native zombie')
            finally:
                try:
                    reap_owned()
                    if stopper is not None:
                        stopper.join(2)
                        self.assertFalse(stopper.is_alive(), 'owned stop worker outlived cleanup')
                    print('OWNED_GATEWAY_CLEANUP complete', flush=True)
                finally:
                    patcher.stop()


if __name__ == '__main__':
    unittest.main()
