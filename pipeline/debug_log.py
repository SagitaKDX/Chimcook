"""
pipeline/debug_log.py
=====================
NDJSON debug logging for the VAD visualiser and pipeline tracing.

Usage:
    from pipeline.debug_log import agent_log

    agent_log("VAD_LEVEL", "orchestrator:VAD_LOOP", "vad_level",
              {"rms": 0.01, "speech_prob": 0.82, "is_speech": True})
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# Log file written under .cursor/ so the IDE live-log panel can pick it up
_LOG_FILE = Path(__file__).parent.parent / ".cursor" / "debug-b7d16a.log"


def agent_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
    run_id: str = "vad-meter",
) -> None:
    """
    Append one NDJSON line to the debug log.

    Parameters
    ----------
    hypothesis_id : str
        Short tag for the metric family, e.g. "VAD_LEVEL".
    location : str
        Source location, e.g. "pipeline/orchestrator_v2.py:VAD_LOOP".
    message : str
        Event name.
    data : dict
        Payload (must be JSON-serialisable).
    run_id : str
        Groups log lines for the IDE panel.
    """
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "sessionId": "b7d16a",
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with _LOG_FILE.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        # Logging must NEVER crash the main pipeline
        pass
