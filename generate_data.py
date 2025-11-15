#!/usr/bin/env python3
"""
Reference data generator for GPU inference game.

Generates synthetic data by running the model on random features, maintaining
per-symbol hidden state. Saves to parquet for use by server.py

Usage:
    python generate_data.py --output reference.parquet --num-symbols 20 --rows-per-symbol 1000
"""

import torch
import polars as pl
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

from model.inference_model import MultiTowerModel, ModelConfig


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class DataGenerator:
    """Generates reference data with random features and model-generated targets."""

    def __init__(
        self,
        rows_per_symbol: int = 100,
        num_symbols: int = 20,
        num_features: int = 79,
        device: Optional[torch.device] = None,
    ):
        self.rows_per_symbol = rows_per_symbol
        self.num_symbols = num_symbols
        self.num_features = num_features
        self.device = device or get_default_device()

        print(f"Using device: {self.device}")

        # Initialize model
        config = ModelConfig(
            hidden_size=2048,
            proj_size=4096,
            tower_depth=12,
            num_heads=8,
            num_features=num_features,
        )
        self.model = MultiTowerModel(config).to(self.device)
        self.model.eval()

        nparams = sum(p.numel() for p in self.model.parameters())
        print(f"Created model with {nparams:,} parameters")

    def generate(self) -> "pl.DataFrame":
        """Generate reference data: rows_per_symbol rows per symbol.
        
        Each symbol maintains its own hidden state sequence, ensuring that
        consecutive rows for the same symbol have proper state continuity.
        
        Returns:
            DataFrame with columns: unique_id, symbol, feature_0..feature_78, target_0..target_3
        """
        rows = []
        unique_id_counter = 0

        total_rows = self.rows_per_symbol * self.num_symbols
        print(f"Generating {total_rows:,} reference data points ({self.rows_per_symbol:,} per symbol)...")

        with torch.inference_mode():
            # Generate rows for each symbol
            for symbol_idx in range(self.num_symbols):
                symbol = f"SYM_{symbol_idx:03d}"

                # Initialize state for this symbol
                state = self.model.init_state(1, self.device)

                # Generate rows_per_symbol rows for this symbol
                for row_in_symbol in range(self.rows_per_symbol):
                    if (unique_id_counter + 1) % max(1, total_rows // 20) == 0:
                        print(f"  {unique_id_counter + 1:,} / {total_rows:,}")

                    # Random features
                    features = torch.randn(1, self.num_features, device=self.device)

                    # Get predictions from model (with state)
                    predictions, state = self.model(features, state)

                    # Extract predictions (one per tower)
                    preds = predictions.cpu().squeeze(0).numpy().astype(float)

                    # Create row
                    feature_cols = {f"feature_{i}": float(features[0, i]) for i in range(self.num_features)}
                    target_cols = {f"target_{i}": float(preds[i]) for i in range(4)}  # 4 towers

                    row = {"unique_id": unique_id_counter, "symbol": symbol, **feature_cols, **target_cols}
                    rows.append(row)
                    unique_id_counter += 1

        # Build a Polars DataFrame from list of dicts for fast parquet IO
        df = pl.from_dicts(rows)
        print(f"Generated {df.height:,} rows with {len(df.columns)} columns")
        return df

    def save(self, df: pl.DataFrame, output_path: str):
        """Save dataframe to parquet file using Polars."""
        df.write_parquet(output_path)
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"Saved {df.height:,} rows to {output_path} ({file_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference data for GPU inference game server"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reference_data.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=20,
        help="Number of symbols to generate data for",
    )
    parser.add_argument(
        "--rows-per-symbol",
        type=int,
        default=1000,
        help="Number of rows to generate per symbol",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu, cuda, mps). Auto-detected if not specified.",
    )

    args = parser.parse_args()

    device = None
    if args.device:
        device = torch.device(args.device)

    generator = DataGenerator(
        rows_per_symbol=args.rows_per_symbol,
        num_symbols=args.num_symbols,
        device=device,
    )

    df = generator.generate()
    generator.save(df, args.output)

    print("\n✓ Data generation complete!")
    print(f"  Load this data in server.py with: --data-file {args.output}")


if __name__ == "__main__":
    main()
