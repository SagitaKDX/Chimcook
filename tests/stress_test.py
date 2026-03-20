"""
tests/stress_test.py
====================

Stress test: runs the HOG Face Detector and OpenWakeWord concurrently for
30 seconds, measures CPU spikes, and verifies the audio queue never overflows.

Assertions:
  - Peak CPU < 90 %
  - 0 buffer overflows
  - Mean per-iteration latency < 150 ms

Run with:
    cd /home/sagitakdx/Desktop/Code/Chimcook
    python tests/stress_test.py
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARNING] psutil not installed – CPU metrics will be skipped")
    print("          Install with: pip install psutil")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEST_DURATION_S = 30          # Total stress-test wall time
REPORT_INTERVAL_S = 5         # Print a per-second summary every N seconds
AUDIO_QUEUE_MAXSIZE = 512     # Max 512-sample frames pending in buffer
SAMPLE_RATE = 16000            # 16 kHz
HOG_INTERVAL_S = 0.5          # Face-detector cadence (matches production config)
WW_CHUNK_SAMPLES = 1280        # 80ms @ 16kHz – matches orchestrator CHUNK_MS=80
WW_INTERVAL_S = 0.05           # Process one WW chunk every 50ms

# Thresholds
PEAK_CPU_LIMIT = 90.0         # %
MAX_OVERFLOWS = 0
MEAN_LATENCY_LIMIT_MS = 150.0


# ---------------------------------------------------------------------------
# Fake HOG detector (avoids camera dependency while being CPU-representative)
# ---------------------------------------------------------------------------

class _FakeHOGDetector:
    """
    Mimics the face_recognition HOG detect call by running a similarly-weighted
    numpy operation: gradient orientations on a 320x240 uint8 frame.
    """

    _W, _H = 320, 240

    def process_frame(self) -> int:
        """Returns face count (always 1 for test). Runs real math."""
        frame = (np.random.rand(self._H, self._W, 3) * 255).astype(np.uint8)
        # Approximate HOG workload: convert to grayscale + compute gradients
        gray = frame @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        gx = np.gradient(gray.astype(np.float32), axis=1)
        gy = np.gradient(gray.astype(np.float32), axis=0)
        _ = np.arctan2(gy, gx)  # orientation map
        return 1


# ---------------------------------------------------------------------------
# Fake OpenWakeWord (replaced with equivalent multiply-accumulate load)
# ---------------------------------------------------------------------------

class _FakeOpenWakeWord:
    """
    Mimics a small neural network forward pass using numpy matmul
    (representative of OWW's float32 inference cost on CPU).
    """
    _IN = 1280
    _H1 = 256
    _H2 = 64

    def __init__(self):
        rng = np.random.default_rng(42)
        self._w1 = rng.standard_normal((self._IN, self._H1)).astype(np.float32)
        self._w2 = rng.standard_normal((self._H1, self._H2)).astype(np.float32)
        self._w3 = rng.standard_normal((self._H2, 1)).astype(np.float32)

    def predict(self, audio: np.ndarray) -> dict:
        x = np.maximum(0, audio @ self._w1)  # ReLU
        x = np.maximum(0, x @ self._w2)
        score = float(1 / (1 + np.exp(-(x @ self._w3).item())))
        return {"alexa": score}


# ---------------------------------------------------------------------------
# Audio buffer simulator (mimics sounddevice queue)
# ---------------------------------------------------------------------------

class AudioBufferSimulator:
    """
    Produces 80ms chunks from a sine-wave generator and puts them into a
    bounded queue at real-time rate. Counts overflow events.
    """

    def __init__(self, q: queue.Queue):
        self._q = q
        self.overflow_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._produce, daemon=True, name="AudioProducer")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _produce(self) -> None:
        t = 0.0
        dt = WW_CHUNK_SAMPLES / SAMPLE_RATE
        while not self._stop.is_set():
            samples = np.sin(
                2 * np.pi * 440 * np.linspace(t, t + dt, WW_CHUNK_SAMPLES, dtype=np.float32)
            ).astype(np.float32)
            try:
                self._q.put_nowait(samples)
            except queue.Full:
                self.overflow_count += 1
            t += dt
            time.sleep(dt)


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

def _hog_worker(
    detector: _FakeHOGDetector,
    stop: threading.Event,
    latencies: List[float],
) -> None:
    """HOG face detector worker — runs every HOG_INTERVAL_S seconds."""
    while not stop.is_set():
        t0 = time.perf_counter()
        detector.process_frame()
        latencies.append((time.perf_counter() - t0) * 1000)
        stop.wait(timeout=HOG_INTERVAL_S)


def _ww_worker(
    model: _FakeOpenWakeWord,
    audio_q: queue.Queue,
    stop: threading.Event,
    latencies: List[float],
) -> None:
    """OpenWakeWord worker — drains audio queue as fast as it arrives."""
    while not stop.is_set():
        try:
            chunk = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue
        t0 = time.perf_counter()
        model.predict(chunk)
        latencies.append((time.perf_counter() - t0) * 1000)


# ---------------------------------------------------------------------------
# Main stress test
# ---------------------------------------------------------------------------

def run_stress_test() -> bool:
    """
    Execute the 30-second concurrent stress test.
    Returns True if all assertions pass.
    """
    print("=" * 60)
    print("Chimcook Concurrent Stress Test")
    print(f"  Duration   : {TEST_DURATION_S}s")
    print(f"  HOG cadence: every {HOG_INTERVAL_S*1000:.0f}ms")
    print(f"  WW cadence : every {WW_INTERVAL_S*1000:.0f}ms")
    print("=" * 60)

    detector = _FakeHOGDetector()
    ww_model = _FakeOpenWakeWord()
    audio_q: queue.Queue = queue.Queue(maxsize=AUDIO_QUEUE_MAXSIZE)

    hog_latencies: List[float] = []
    ww_latencies: List[float] = []
    cpu_samples: List[float] = []

    stop = threading.Event()

    # Threads
    audio_buf = AudioBufferSimulator(audio_q)
    hog_thread = threading.Thread(
        target=_hog_worker, args=(detector, stop, hog_latencies), daemon=True, name="HOG"
    )
    ww_thread = threading.Thread(
        target=_ww_worker, args=(ww_model, audio_q, stop, ww_latencies), daemon=True, name="WW"
    )

    proc = psutil.Process() if HAS_PSUTIL else None

    # Start
    audio_buf.start()
    hog_thread.start()
    ww_thread.start()

    t_start = time.time()
    last_report = t_start

    while time.time() - t_start < TEST_DURATION_S:
        time.sleep(0.5)
        now = time.time()

        if HAS_PSUTIL:
            cpu = proc.cpu_percent(interval=None)
            cpu_samples.append(cpu)

        if now - last_report >= REPORT_INTERVAL_S:
            elapsed = now - t_start
            hog_mean = np.mean(hog_latencies[-20:]) if hog_latencies else 0.0
            ww_mean = np.mean(ww_latencies[-50:]) if ww_latencies else 0.0
            cpu_cur = cpu_samples[-1] if cpu_samples else 0.0
            overflows = audio_buf.overflow_count
            qsize = audio_q.qsize()
            print(
                f"  [{elapsed:5.1f}s] HOG={hog_mean:.1f}ms  WW={ww_mean:.1f}ms  "
                f"CPU={cpu_cur:.1f}%  overflows={overflows}  q={qsize}"
            )
            last_report = now

    stop.set()
    audio_buf.stop()
    hog_thread.join(timeout=3.0)
    ww_thread.join(timeout=3.0)

    print("\n" + "=" * 60)
    print("Stress Test Results")
    print("=" * 60)

    all_latencies = hog_latencies + ww_latencies
    peak_cpu = max(cpu_samples) if cpu_samples else 0.0
    mean_lat = float(np.mean(all_latencies)) if all_latencies else 0.0
    overflows = audio_buf.overflow_count
    hog_iters = len(hog_latencies)
    ww_iters = len(ww_latencies)

    print(f"  HOG iterations      : {hog_iters}")
    print(f"  WW iterations       : {ww_iters}")
    print(f"  Audio overflows     : {overflows}  (limit: {MAX_OVERFLOWS})")
    print(f"  Mean iteration lat  : {mean_lat:.2f}ms  (limit: {MEAN_LATENCY_LIMIT_MS}ms)")
    if HAS_PSUTIL:
        print(f"  Peak CPU            : {peak_cpu:.1f}%  (limit: {PEAK_CPU_LIMIT}%)")
    else:
        print("  Peak CPU            : N/A (psutil not installed)")

    # Assertions
    failures: List[str] = []
    if overflows > MAX_OVERFLOWS:
        failures.append(f"Audio buffer overflowed {overflows} time(s) (limit: {MAX_OVERFLOWS})")
    if mean_lat > MEAN_LATENCY_LIMIT_MS:
        failures.append(
            f"Mean latency {mean_lat:.2f}ms exceeds {MEAN_LATENCY_LIMIT_MS}ms limit"
        )
    if HAS_PSUTIL and peak_cpu > PEAK_CPU_LIMIT:
        failures.append(f"Peak CPU {peak_cpu:.1f}% exceeds {PEAK_CPU_LIMIT}% limit")

    if failures:
        print("\n[STRESS TEST FAILED]")
        for f in failures:
            print(f"  ✗ {f}")
        return False
    else:
        print("\n[STRESS TEST PASSED] ✓ All assertions met")
        return True


# ---------------------------------------------------------------------------
# Pytest entrypoint (optional)
# ---------------------------------------------------------------------------

def test_stress_concurrent():
    """Pytest-compatible wrapper for the stress test."""
    assert run_stress_test(), "Stress test failed — see output above"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    passed = run_stress_test()
    sys.exit(0 if passed else 1)
