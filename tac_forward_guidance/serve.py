"""Serve a baseline and full tactile forward guidance with independently labelled metadata."""

import argparse
import json
import logging
from pathlib import Path

import torch

from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from tac_forward_guidance.config import add_guidance_arguments
from tac_forward_guidance.config import config_from_args
from tac_forward_guidance.runtime import PokerForwardGuidancePolicy
from tac_forward_guidance.runtime import load_forward_guidance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-checkpoint", "cpm-checkpoint", "data-root", "tokenizer-model"):
        parser.add_argument("--" + name, type=Path, required=True)
    add_guidance_arguments(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18787)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    torch.set_num_threads(8)
    loaded = load_forward_guidance(
        base_checkpoint=args.base_checkpoint,
        cpm_checkpoint=args.cpm_checkpoint,
        data_root=args.data_root,
        guidance=config_from_args(args),
        precision=args.precision,
    )
    policy = PokerForwardGuidancePolicy(loaded, tokenizer_model=args.tokenizer_model)
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(policy.metadata, indent=2) + "\n")
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
