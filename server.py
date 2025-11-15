#!/usr/bin/env python3
"""
Server for GPU inference game.
Streams inference requests to clients and scores their responses.

=== SCORING MODEL ===

For each inference request, the server scores predictions using:

ACCURACY: Predictions within 1e-2 (0.01) are marked correct
- CORRECT prediction: +$6 base reward (reduced by latency)
- INCORRECT prediction: -$1 fixed penalty (regardless of speed)

LATENCY SCALING (for correct predictions only):
- 0ms response: $6 per tower
- 1000ms response: $1 per tower
- Linear interpolation between
- Formula: reward = $1 + max(0, 1 - latency_ms/1000) * $5

TOTAL PER REQUEST (4 towers):
- All correct at 0ms: $24 max
- All correct at 1000ms: $4
- All incorrect: -$4
- Mixed cases: sum of individual tower scores

Each client maintains its own cumulative PNL counter.
Server streams requests continuously at configurable intervals (default 100ms).
"""

import socket
import threading
import time
import numpy as np
import polars as pl
import argparse
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import sys

from protocol import (
    ProtocolHandler,
    SocketReader,
    SocketWriter,
    RegisterMessage,
    InferenceRequest,
    InferenceResponse,
    ScoreUpdate,
    ErrorMessage,
)


class DataLoader:
    """Loads reference data from parquet file for serving."""

    @staticmethod
    def load(data_file: str) -> "pl.DataFrame":
        """Load reference data from parquet file.
        
        Args:
            data_file: Path to parquet file containing reference data.
            
        Returns:
            DataFrame with columns: unique_id, symbol, feature_0..feature_78, target_0..target_3
            
        Raises:
            FileNotFoundError: If data_file does not exist.
        """
        if not Path(data_file).exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                f"Generate it with: python generate_data.py --output {data_file}"
            )
        
        print(f"Loading reference data from {data_file}...")
        df = pl.read_parquet(data_file)
        file_size_mb = Path(data_file).stat().st_size / (1024 * 1024)
        print(f"Loaded {df.height:,} rows ({file_size_mb:.1f} MB)")
        return df


@dataclass
class InFlightRequest:
    """Tracks an in-flight request."""

    unique_id: int
    symbol: str
    features: List[float]
    targets: List[float]  # Ground truth from model
    sent_time: float
    response: Optional[InferenceResponse] = None
    response_time: Optional[float] = None


class GameServer:
    """Main server for the GPU inference game.
    
    Loads pre-generated reference data from parquet and streams inference
    requests to connected clients. Scores their responses based on accuracy
    and latency.
    """

    # Scoring parameters
    ACCURACY_THRESHOLD = 1e-2  # Tolerance for "correct" prediction
    CORRECT_REWARD = 6.0  # Reward per correct tower
    INCORRECT_PENALTY = -1.0  # Penalty per incorrect tower
    SPEED_THRESHOLD_MS = 1000.0  # ms to go from full reward to base reward

    def __init__(
        self,
        data_file: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        mean_request_interval_ms: float = 100.0,
        scoreboard_host: str = "localhost",
        scoreboard_port: int = 9000,
    ):
        self.host = host
        self.port = port
        self.mean_request_interval_ms = mean_request_interval_ms
        self.scoreboard_host = scoreboard_host
        self.scoreboard_port = scoreboard_port

        # Load reference data
        self.data = DataLoader.load(data_file)

        # Server state
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.scoreboard_socket: Optional[socket.socket] = None
        self.scoreboard_writer: Optional[SocketWriter] = None
        self.clients: Dict[int, Dict] = {}  # client_id -> {socket, reader, writer, thread}
        self.client_counter = 0
        self.client_lock = threading.RLock()

        # Request tracking: organize data by symbol for batched streaming
        self.symbol_rows: Dict[str, List[int]] = {}  # symbol -> list of row indices
        self._organize_rows_by_symbol()
        
        self.symbol_row_indices: Dict[str, int] = {}  # symbol -> current row index in its list
        for symbol in self.symbol_rows:
            self.symbol_row_indices[symbol] = 0

        # Precompute feature and target column lists from loaded data
        self.feature_cols = [c for c in self.data.columns if c.startswith("feature_")]
        self.target_cols = [c for c in self.data.columns if c.startswith("target_")]

        self.in_flight: Dict[int, InFlightRequest] = {}  # unique_id -> request
        self.request_lock = threading.RLock()

        # Client scoring (per-client)
        self.client_scores: Dict[int, Dict] = {}  # client_id -> {pnl, num_responses}
        self.score_lock = threading.RLock()

        num_symbols = len(self.symbol_rows)
        rows_per_symbol = self.data.height // num_symbols if num_symbols > 0 else 0

        print(f"GameServer initialized on {host}:{port}")
        print(f"Data: {self.data.height:,} rows ({rows_per_symbol} per symbol, {num_symbols} symbols)")
        print(f"Mean request interval: {mean_request_interval_ms:.1f} ms")
    
    def _organize_rows_by_symbol(self):
        """Organize data rows by symbol for efficient streaming."""
        # Use Polars Series to gather symbols efficiently
        if not hasattr(self.data, "height"):
            # fallback
            n = len(self.data)
        else:
            n = self.data.height

        # collect symbol column once
        symbols = list(self.data["symbol"].to_list())
        for idx, symbol in enumerate(symbols):
            symbol = str(symbol)
            if symbol not in self.symbol_rows:
                self.symbol_rows[symbol] = []
            self.symbol_rows[symbol].append(idx)

    def start(self):
        """Start the server."""
        self.running = True

        # Connect to scoreboard
        self._connect_to_scoreboard()

        # Start server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        print(f"Server listening on {self.host}:{self.port}")

        # Start accept thread
        accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        accept_thread.start()

        # Start request streaming thread
        stream_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        stream_thread.start()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop()

    def _connect_to_scoreboard(self):
        """Connect to the scoreboard and register."""
        try:
            self.scoreboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.scoreboard_socket.connect((self.scoreboard_host, self.scoreboard_port))
            self.scoreboard_writer = SocketWriter(self.scoreboard_socket)
            print(f"✓ Connected to scoreboard at {self.scoreboard_host}:{self.scoreboard_port}")
        except Exception as e:
            print(f"⚠ Could not connect to scoreboard: {e}")
            print(f"  Scoreboard metrics will not be recorded.")
            self.scoreboard_socket = None
            self.scoreboard_writer = None

    def stop(self):
        """Stop the server."""
        self.running = False

        # Close all client connections
        with self.client_lock:
            for client_id, client_info in self.clients.items():
                try:
                    client_info["socket"].close()
                except:
                    pass

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

    def _accept_loop(self):
        """Background thread to accept new client connections."""
        while self.running:
            try:
                assert self.server_socket is not None
                client_socket, client_addr = self.server_socket.accept()
                print(f"New connection from {client_addr}")

                with self.client_lock:
                    client_id = self.client_counter
                    self.client_counter += 1

                    reader = SocketReader(client_socket)
                    writer = SocketWriter(client_socket)

                    # Wait for register message
                    msg = reader.read_message(timeout=5)
                    if isinstance(msg, RegisterMessage):
                        print(f"Client {client_id} registered")

                        self.clients[client_id] = {
                            "socket": client_socket,
                            "reader": reader,
                            "writer": writer,
                            "addr": client_addr,
                        }

                        # Track scores for all clients
                        self.client_scores[client_id] = {"pnl": 0.0, "num_responses": 0}

                        # Start receiver thread for this client
                        recv_thread = threading.Thread(
                            target=self._client_receive_loop, args=(client_id,), daemon=True
                        )
                        recv_thread.start()
                    else:
                        print(f"Expected RegisterMessage, got {type(msg)}")
                        client_socket.close()

            except Exception as e:
                if self.running:
                    print(f"Accept error: {e}")

    def _client_receive_loop(self, client_id: int):
        """Background thread to receive responses from a client."""
        with self.client_lock:
            if client_id not in self.clients:
                return
            reader = self.clients[client_id]["reader"]

        print(f"Client {client_id} receive loop started")

        while self.running:
            try:
                msg = reader.read_message(timeout=1)

                if msg is None:
                    # Timeout or connection closed
                    if not self.running:
                        break
                    continue

                if isinstance(msg, InferenceResponse):
                    self._handle_response(client_id, msg)
                elif isinstance(msg, ErrorMessage):
                    print(f"Client {client_id} error: {msg.error}")

            except Exception as e:
                if self.running:
                    print(f"Client {client_id} receive error: {e}")
                break

        # Clean up client
        with self.client_lock:
            if client_id in self.clients:
                try:
                    self.clients[client_id]["socket"].close()
                except:
                    pass
                del self.clients[client_id]

        print(f"Client {client_id} disconnected")

    def _streaming_loop(self):
        """Background thread that streams batched requests (one per symbol) to clients.
        
        Each request batch contains one row per symbol, allowing clients to process
        all symbols in a single inference call.
        """

        print("Request streaming loop started")

        while self.running:
            try:
                # Get current clients
                with self.client_lock:
                    client_ids = list(self.clients.keys())

                if not client_ids:
                    time.sleep(0.01)
                    continue

                # Pick a random inference client to send to
                client_id = np.random.choice(client_ids)

                # Create a batched request: one row per symbol
                unique_ids = []
                symbols = []
                features_list = []
                targets_list = []

                for symbol in sorted(self.symbol_rows.keys()):
                    rows_for_symbol = self.symbol_rows[symbol]
                    if not rows_for_symbol:
                        continue
                    
                    # Get the current row index for this symbol (with wraparound)
                    row_idx_in_list = self.symbol_row_indices[symbol]
                    data_row_idx = rows_for_symbol[row_idx_in_list]
                    
                    # Advance to next row for this symbol
                    self.symbol_row_indices[symbol] = (row_idx_in_list + 1) % len(rows_for_symbol)
                    
                    # Extract data from this row using Polars APIs
                    unique_id = int(self.data["unique_id"][data_row_idx])
                    # Get features and targets by selecting respective columns
                    features_vals = self.data.select(self.feature_cols).row(data_row_idx)
                    targets_vals = self.data.select(self.target_cols).row(data_row_idx)

                    features = [float(x) for x in features_vals]
                    targets = [float(x) for x in targets_vals]

                    unique_ids.append(unique_id)
                    symbols.append(symbol)
                    features_list.append(features)
                    targets_list.append(targets)
                    
                    # Track in-flight request
                    with self.request_lock:
                        self.in_flight[unique_id] = InFlightRequest(
                            unique_id=unique_id,
                            symbol=symbol,
                            features=features,
                            targets=targets,
                            sent_time=time.time(),
                        )

                # Send batched request to client
                if unique_ids:
                    inference_request = InferenceRequest(
                        unique_ids=unique_ids,
                        symbols=symbols,
                        features=features_list,
                        timestamp=time.time(),
                    )

                    with self.client_lock:
                        if client_id in self.clients:
                            writer = self.clients[client_id]["writer"]
                            if not writer.send_message(inference_request):
                                print(f"Failed to send request batch to client {client_id}")

                # Sleep according to mean request interval (with some randomness)
                interval = np.random.exponential(self.mean_request_interval_ms / 1000.0)
                time.sleep(interval)

            except Exception as e:
                if self.running:
                    print(f"Streaming error: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1)

    def _handle_response(self, client_id: int, response: InferenceResponse):
        """Handle an inference response from a client."""

        response_time = time.time()

        # Score each prediction
        for unique_id, predictions in zip(response.unique_ids, response.predictions):
            with self.request_lock:
                if unique_id not in self.in_flight:
                    print(f"Received response for unknown request {unique_id}")
                    continue

                req = self.in_flight[unique_id]
                req.response = response
                req.response_time = response_time

            # Calculate latency
            latency_ms = (response_time - req.sent_time) * 1000

            # Score this request
            trade_pnl = self._calculate_pnl(predictions, req.targets, latency_ms)

            # Update client score
            with self.score_lock:
                self.client_scores[client_id]["pnl"] += trade_pnl
                self.client_scores[client_id]["num_responses"] += 1

            # Compute accuracy per tower
            accuracies = [abs(p - t) for p, t in zip(predictions, req.targets)]

            # Send score update back to client
            score_update = ScoreUpdate(
                unique_ids=[unique_id],
                trade_pnls=[trade_pnl],
                accuracies=accuracies,
                latencies_ms=[latency_ms],
            )

            # Send score update to scoreboard
            if self.scoreboard_writer:
                try:
                    if not self.scoreboard_writer.send_message(score_update):
                        print(f"Failed to send score update to scoreboard")
                except Exception as e:
                    print(f"Scoreboard connection error: {e}")
                    self.scoreboard_writer = None

            # Print stats periodically
            with self.score_lock:
                num_responses = self.client_scores[client_id]["num_responses"]
                if num_responses > 0 and num_responses % 100 == 0:
                    avg_pnl = self.client_scores[client_id]["pnl"] / num_responses
                    print(
                        f"Client {client_id}: {num_responses} responses, "
                        f"avg PNL: ${avg_pnl:.2f}, total: ${self.client_scores[client_id]['pnl']:.2f}"
                    )

    def _calculate_pnl(self, predictions: List[float], targets: List[float], latency_ms: float) -> float:
        """Calculate PNL for a request.
        
        Scoring rules:
        - Correct (within 1e-2): +$6 at 0ms, +$1 at 1000ms (linear)
        - Incorrect: -$1 (flat, regardless of latency)
        """

        total_pnl = 0.0

        for pred, target in zip(predictions, targets):
            error = abs(pred - target)
            is_correct = error < self.ACCURACY_THRESHOLD

            if is_correct:
                # Correct prediction: scale reward by speed
                # Full reward at 0ms, minimum reward at 1000ms
                speed_factor = max(0.0, 1.0 - latency_ms / self.SPEED_THRESHOLD_MS)
                # Map speed_factor from [0, 1] to reward [$1, $6]
                reward = 1.0 + speed_factor * (self.CORRECT_REWARD - 1.0)
                total_pnl += reward
            else:
                # Incorrect prediction: fixed penalty
                total_pnl += self.INCORRECT_PENALTY

        return total_pnl

    def print_stats(self):
        """Print server statistics."""
        with self.score_lock:
            print("\n=== Server Statistics ===")
            for client_id, scores in self.client_scores.items():
                if scores["num_responses"] > 0:
                    avg_pnl = scores["pnl"] / scores["num_responses"]
                    print(
                        f"Client {client_id}: {scores['num_responses']} responses, "
                        f"${scores['pnl']:.2f} total PNL, ${avg_pnl:.2f} avg"
                    )


def main():
    parser = argparse.ArgumentParser(description="GPU Inference Game Server")
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Parquet file with reference data (generate with generate_data.py)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--mean-request-interval",
        type=float,
        default=100.0,
        help="Mean time between requests in milliseconds",
    )
    parser.add_argument(
        "--scoreboard-host",
        type=str,
        default="localhost",
        help="Scoreboard host",
    )
    parser.add_argument(
        "--scoreboard-port",
        type=int,
        default=9000,
        help="Scoreboard port",
    )

    args = parser.parse_args()

    server = GameServer(
        data_file=args.data_file,
        host=args.host,
        port=args.port,
        mean_request_interval_ms=args.mean_request_interval,
        scoreboard_host=args.scoreboard_host,
        scoreboard_port=args.scoreboard_port,
    )

    server.start()


if __name__ == "__main__":
    main()
