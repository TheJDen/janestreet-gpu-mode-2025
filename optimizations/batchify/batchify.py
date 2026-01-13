#!/usr/bin/env python3
"""
Example model implementation for GPU inference game.
Shows how to implement the BaseInferenceClient.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import argparse
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
import itertools


from huggingface_hub import hf_hub_download

from client import BaseInferenceClient, PendingRequest, InferenceResponse
from model.inference_model import MultiTowerModel, ModelConfig


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def recursive_clone(state):
    if isinstance(state, torch.Tensor):
        return state.clone()
    else:
        return [recursive_clone(s) for s in state]

class RoundRobinDataset(Dataset):

    def __init__(self, requests_by_symbol):
        self.requests_by_symbol = requests_by_symbol
        self.n = max(len(reqs) for reqs in requests_by_symbol.values())

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        uids, syms, features = [], [], []
        for sym, reqs in self.requests_by_symbol.items():
            if i >= len(reqs):
                continue
            req = reqs[i]
            syms.append(sym)
            uids.append(req.unique_id)
            features.append(req.features)
        return uids, syms, torch.tensor(features)

def batchify_states(states):
    if isinstance(states[0], torch.Tensor):
        batched = torch.cat(states, dim=0)
        torch._dynamo.mark_dynamic(batched, 0)
        return batched
    return [batchify_states(list(s)) for s in zip(*states)]


def unbatchify_state(batched_state):
    if isinstance(batched_state, torch.Tensor):
        B = batched_state.size(0)
        return [batched_state[i:i+1].clone() for i in range(B)] # remember individual states had batch dim 1, not not batch
    return list(zip(*[unbatchify_state(child) for child in batched_state]))


class NnInferenceClient(BaseInferenceClient):
    def __init__(
        self,
        num_symbols: int,
        server_host: str = "localhost",
        server_port: int = 8080,
        device: str | None = None,
        token: str | None = None,
    ):
        super().__init__(num_symbols, server_host, server_port)

        self.device = device or get_default_device()

        config = ModelConfig(
            hidden_size=2048,
            proj_size=4096,
            tower_depth=12,
            num_heads=8,
            num_features=79,
        )
        self.model = MultiTowerModel(config).to(self.device)

        nparams = sum(p.numel() for p in self.model.parameters())
        print(f"{nparams = }")

        

        weights_file = hf_hub_download(
            repo_id="jane-street-gpu-mode/hackathon",
            filename="state_dict.pt",
            token=token,
        )
        weights = torch.load(weights_file, weights_only=True)
        self.model.load_state_dict(weights)

        self.model = torch.compile(
            self.model,
            mode="reduce-overhead",
            fullgraph=True
        )

        self.states = {
            f"SYM_{num:03d}": self.model.init_state(1, self.device)
            for num in range(self.num_symbols)
        }

    def process_batch(
        self, requests_by_symbol: Dict[str, List[PendingRequest]]
    ) -> InferenceResponse:
        unique_ids = []

        start = time.time()

        total_preds = sum(len(v) for v in requests_by_symbol.values())
        pinned_preds_buffer = torch.empty((total_preds, 4), pin_memory=True) # blocking
        row = 0

        batches = RoundRobinDataset(requests_by_symbol)
        for b in range(len(batches)):
            uids, syms, features = batches[b]
            features = features.to(device=self.device)
            batched_state = batchify_states([self.states[sym] for sym in syms])
            with torch.inference_mode():
                torch.compiler.cudagraph_mark_step_begin()
                torch._dynamo.mark_dynamic(features, 0)
                pred, batched_state = self.model(features, batched_state)
                unique_ids.extend(uids)
                pinned_preds_buffer[row: row + len(pred)].copy_(pred, non_blocking=True) # D2H
                row += len(pred)
            for sym, state in zip(syms, unbatchify_state(batched_state)):
                self.states[sym] = state 

        torch.cuda.synchronize(self.device) # make sure all pred copies are done, 1 sync instead of len(preds) syncs
        preds = pinned_preds_buffer.numpy().astype(float).tolist()

        end = time.time()
        elapsed = end - start

        # May not be a bad idea to print less often!
        print(f"{len(preds) = }, {elapsed = }, {elapsed / len(preds) = }")

        return InferenceResponse(
            unique_ids=unique_ids, predictions=preds, client_timestamp=time.time()
        )


def main():
    parser = argparse.ArgumentParser(description="Example inference client")
    parser.add_argument("--host", type=str, default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=20,
        help="Number of symbols in the tradeable universe",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token to download the model (for testing before the hackathon)",
    )

    args = parser.parse_args()
    client = NnInferenceClient(
        num_symbols=args.num_symbols,
        server_host=args.host,
        server_port=args.port,
        token=args.token,
    )

    client.run()


if __name__ == "__main__":
    main()
