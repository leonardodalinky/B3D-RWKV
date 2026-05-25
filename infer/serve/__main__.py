"""Entry point: ``python -m infer.serve``.

Reads HOST / PORT from env (defaults 0.0.0.0:8000) and hands control to
uvicorn. Everything else (CKPT, MODEL_NAME, architecture, sampling
defaults) is read inside the lifespan handler in ``infer.serve.app`` so
the config knobs all live next to where they're consumed.

Critical: ``workers=1`` is non-negotiable. The semaphore + loaded model
are per-process state; spawning multiple workers would each load their
own GPU model (OOM) and break the "max-1-concurrent" guarantee.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("infer.serve.app:app", host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
