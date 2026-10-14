"""Backward-compatible poker adapter for the common evaluation video."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .evaluation_video import EvaluationVideo


class PokerPolicyVideo(EvaluationVideo):
  """Poker-named compatibility wrapper; tactile output is bilateral in v2."""

  def __init__(
    self,
    sim: Any,
    output: Path,
    *,
    fps: int = 10,
    width: int = 1920,
    height: int = 1080,
    render_width: int = 640,
    render_height: int = 480,
    second_camera: str = "global",
    include_model_wrist: bool = False,
    include_review_wrist: bool = False,
    comparison: bool = False,
    reference_dataset: Path | None = None,
    reference_episode_index: int = 0,
    metrics: Any | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> None:
    super().__init__(
      sim,
      output,
      fps=fps,
      width=width,
      height=height,
      render_width=render_width,
      render_height=render_height,
      second_camera=second_camera,
      include_model_wrist=include_model_wrist,
      include_review_wrist=include_review_wrist,
      comparison=comparison,
      reference_dataset=reference_dataset,
      reference_episode_index=reference_episode_index,
      metrics=metrics,
      heading="POKER MODEL EVALUATION",
      metadata=metadata,
    )
