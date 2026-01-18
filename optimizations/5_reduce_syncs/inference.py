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

from huggingface_hub import hf_hub_download

from client import BaseInferenceClient, PendingRequest, InferenceResponse
from model.inference_model import MultiTowerModel, ModelConfig

# allow TF32 tensor cores at cost of precision
torch.set_float32_matmul_precision('high')


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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

        self.config = ModelConfig(
            hidden_size=2048,
            proj_size=4096,
            tower_depth=12,
            num_heads=8,
            num_features=79,
        )
        self.model = MultiTowerModel(self.config).to(self.device)

        nparams = sum(p.numel() for p in self.model.parameters())
        print(f"{nparams = }")

        self.symbol_to_index = {f"SYM_{i:03d}": i for i in range(self.num_symbols)}

        self.state = self.model.init_state(self.num_symbols, self.device)

        weights_file = hf_hub_download(
            repo_id="jane-street-gpu-mode/hackathon",
            filename="state_dict.pt",
            token=token,
        )
        weights = torch.load(weights_file, weights_only=True)
        self.model.load_state_dict(weights)

        self.model = torch.compile(
            self.model,
            fullgraph=True,
            mode="reduce-overhead"
        )

    def interleave_by_symbol(self, requests_by_symbol):
        n = max(len(reqs) for reqs in requests_by_symbol.values())
        for i in range(n):
            padded_features = torch.empty((self.num_symbols, self.config.num_features), pin_memory=True)
            uids, symbol_indices = [], []
            for symbol, reqs in requests_by_symbol.items():
                if i >= len(reqs):
                    continue
                req = reqs[i]
                uids.append(req.unique_id)
                symbol_index = self.symbol_to_index[symbol]
                symbol_indices.append(symbol_index)
                padded_features[symbol_index].copy_(torch.tensor(req.features))
            yield uids, torch.tensor(symbol_indices, pin_memory=True), padded_features

    def update_state_at_indices(self, indices, src_state, dst_state=None):
        dst_state = dst_state if dst_state is not None else self.state
        if isinstance(src_state, torch.Tensor):
            dst_state.index_copy_(0, indices, src_state.index_select(0, indices))
            return
        for src_s, dst_s in zip(src_state, dst_state):
            self.update_state_at_indices(indices, src_s, dst_s)

    def process_batch(
        self, requests_by_symbol: Dict[str, List[PendingRequest]]
    ) -> InferenceResponse:
        unique_ids, preds_gpu = [], []

        start = time.time()

        batches = self.interleave_by_symbol(requests_by_symbol)

        for uids, symbol_indices, batched_features in batches:
            batched_features = batched_features.to(device=self.device, non_blocking=True)
            symbol_indices = symbol_indices.to(device=self.device, non_blocking=True)
            
            with torch.inference_mode():
                batched_preds, batched_state = self.model(batched_features, self.state)
            self.update_state_at_indices(symbol_indices, batched_state)

            unique_ids.extend(uids)
            preds_gpu.append(batched_preds.index_select(0, symbol_indices))
            
        preds = torch.cat(preds_gpu, dim=0).float().cpu().tolist()    

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