#!/usr/bin/env python3
"""Serve frozen pi0.5 and pi0.5+TouchGuide with the existing OpenPI WebSocket protocol."""

import argparse
import json
import logging
from pathlib import Path

import torch

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.touchguide.inference import load_touchguide
from openpi.touchguide.policy import PokerTouchGuidePolicy
from openpi.touchguide.sampler import GuidanceConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--cpm-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--execute-steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-steps", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18786)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.execute_steps <= 30:
        parser.error("execute-steps must be in 1..30")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    torch.set_num_threads(8)
    loaded = load_touchguide(
        base_checkpoint=args.base_checkpoint,
        cpm_checkpoint=args.cpm_checkpoint,
        data_root=args.data_root,
        guidance=GuidanceConfig(guidance_steps=args.guidance_steps, scale=args.guidance_scale),
    )
    policy = PokerTouchGuidePolicy(loaded, tokenizer_model=args.tokenizer_model, execute_steps=args.execute_steps)
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(policy.metadata, indent=2) + "\n")
    logging.info("Loaded frozen poker policy, listening on %s:%s", args.host, args.port)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
