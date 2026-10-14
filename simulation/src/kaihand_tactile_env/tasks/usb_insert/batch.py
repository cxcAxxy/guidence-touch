"""USB-only bounded spawned capture, independent of the poker batch runner."""

from __future__ import annotations

import multiprocessing
import os
import signal
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

_STOP = None
THREAD_VARIABLES = (
  "OPENBLAS_NUM_THREADS",
  "OMP_NUM_THREADS",
  "MKL_NUM_THREADS",
  "NUMEXPR_NUM_THREADS",
  "LP_NUM_THREADS",
)


def configure_environment():
  for name in THREAD_VARIABLES:
    os.environ[name] = "1"
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


def _initialize_worker(stop):
  global _STOP
  _STOP = stop
  configure_environment()
  # The parent receives terminal Ctrl+C and signals all workers. Let each
  # recorder close its HDF5 and renderer rather than interrupting an HDF5 write.
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  signal.signal(signal.SIGTERM, lambda *_: stop.set())


def _record_job(job):
  from .recording import record_episode

  return record_episode(job, _STOP.is_set)


def _worker_failure(job):
  from .recording import episode_seeds, write_json_new

  object_seed, noise_seed = episode_seeds(job.root_seed, job.episode_index)
  report = {
    "episode_index": job.episode_index,
    "status": "exception",
    "success": False,
    "raw_path": str(job.output.with_suffix(".h5.partial")),
    "object_seed": object_seed,
    "noise_seed": noise_seed,
    "motion_profile": job.motion_profile,
    "failure_reason": traceback.format_exc(),
    "wall_duration_s": None,
    "outcome": {},
  }
  path = job.output.with_suffix(".worker_failure.json")
  write_json_new(path, report)
  return report


def run_jobs(jobs, workers=1, should_stop=lambda: False):
  """Yield completed reports; at most `workers` jobs are ever submitted.

  Cancellation stops submission, signals active recorders and drains their
  reports before returning. No HDF5/model/GL context crosses a process boundary.
  """
  if isinstance(workers, bool) or not isinstance(workers, int) or workers not in (1, 2):
    raise ValueError("USB workers must be 1 or 2")
  jobs = tuple(jobs)
  if len({job.output.resolve() for job in jobs}) != len(jobs):
    raise ValueError("USB jobs must have distinct output files")
  if workers == 1:
    from .recording import record_episode

    for job in jobs:
      if should_stop():
        break
      yield record_episode(job, should_stop)
    return
  if not jobs or should_stop():
    return
  configure_environment()
  context = multiprocessing.get_context("spawn")
  stop = context.Event()
  pool = ProcessPoolExecutor(
    max_workers=min(workers, len(jobs)),
    mp_context=context,
    initializer=_initialize_worker,
    initargs=(stop,),
  )
  pending = {}
  remaining = iter(jobs)

  def submit_next():
    job = next(remaining, None)
    if job is not None:
      pending[pool.submit(_record_job, job)] = job

  try:
    for _ in range(min(workers, len(jobs))):
      submit_next()
    while pending:
      if should_stop():
        stop.set()
      done, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
      for future in done:
        job = pending.pop(future)
        try:
          report = future.result()
        except Exception:
          # Infrastructure failures invalidate the batch. Preserve any raw
          # bytes, retain the traceback and stop launching additional episodes.
          stop.set()
          report = _worker_failure(job)
        if report["status"] == "cancelled":
          stop.set()
        yield report
      # Refill only after all newly completed reports have been consumed.
      if should_stop():
        stop.set()
      if not stop.is_set():
        for _ in range(len(done)):
          submit_next()
  finally:
    stop.set()
    pool.shutdown(wait=True, cancel_futures=True)
