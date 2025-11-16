#!/usr/bin/env python3
"""
Live scoreboard for GPU inference game.

Listens for incoming connections from game servers and displays real-time metrics:
- Total PNL
- Average PNL per response
- Average accuracy per tower
- Average response latency

Servers register and push ScoreUpdate messages to the scoreboard.

Usage:
    python scoreboard.py --port 9000
"""

import socket
import time
import sys
import argparse
import threading
from typing import Optional, Dict, Any
from collections import deque
from dataclasses import dataclass, field

from protocol import (
    ProtocolHandler,
    SocketReader,
    SocketWriter,
    RegisterMessage,
    ScoreUpdate,
)


@dataclass
class ClientStats:
    """Statistics for a single client."""

    client_id: int
    total_pnl: float = 0.0
    num_responses: int = 0
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    ema_pnl_per_second: float = 0.0  # Exponential moving average of PNL per second
    # Smoothing factor (lower = smoother / less reactive to individual spikes)
    ema_alpha: float = 0.12
    # Minimum delta time (seconds) to use when computing batch rate. Protects
    # against very small time deltas that would otherwise produce enormous rates.
    min_dt: float = 0.1
    latencies: deque = field(default_factory=lambda: deque(maxlen=100))  # Keep last 100 latencies
    all_accuracies: deque = field(default_factory=lambda: deque(maxlen=400))  # Keep last 100*4 tower predictions
    stale_threshold_seconds: float = 30.0  # Mark metrics as stale if no update for this long

    def update(self, score_update: ScoreUpdate):
        """Update stats from a score update."""
        # Note: score_update.unique_ids, trade_pnls, latencies_ms have length = number of requests
        # But score_update.accuracies has length = number of requests * 4 (one per tower per request)
        num_requests = len(score_update.unique_ids)
        
        current_time = time.time()
        time_since_last_update = current_time - self.last_update_time
        
        for i in range(num_requests):
            pnl = score_update.trade_pnls[i]
            lat = score_update.latencies_ms[i]
            # Extract the 4 accuracies for this request (one per tower)
            tower_accs = score_update.accuracies[i * 4 : (i + 1) * 4]
            
            self.total_pnl += pnl
            self.num_responses += 1
            self.latencies.append(lat)
            # Add all 4 tower accuracies to the rolling window
            for acc in tower_accs:
                self.all_accuracies.append(acc)
        
        # Update exponential moving average of PNL per second
        if time_since_last_update > 0:
            # Protect against tiny intervals which amplify the observed rate
            dt = max(time_since_last_update, self.min_dt)
            # Calculate the rate for this batch of requests (guarded dt)
            batch_pnl_per_second = sum(score_update.trade_pnls) / dt

            # Optionally, cap extremely large observed rates to avoid large swings
            # relative_cap = max( abs(self.ema_pnl_per_second) * 10.0, 1e6 )
            # batch_pnl_per_second = max(-relative_cap, min(relative_cap, batch_pnl_per_second))

            # EMA update: new_value = alpha * current_observation + (1 - alpha) * old_value
            # Use a smaller alpha so that single small-dt batches don't dominate
            self.ema_pnl_per_second = (
                self.ema_alpha * batch_pnl_per_second + 
                (1 - self.ema_alpha) * self.ema_pnl_per_second
            )
        
        self.last_update_time = current_time

    def get_stats(self) -> Dict[str, Any]:
        """Return computed statistics.
        
        Returns:
            Dict with keys: total_pnl, num_responses, avg_pnl_per_second, avg_latency, avg_accuracy
            Values can be float or None (for stale metrics)
        """
        elapsed_seconds = time.time() - self.start_time
        time_since_last_update = time.time() - self.last_update_time
        
        # Check if metrics are stale (no updates for a long time)
        is_stale = time_since_last_update > self.stale_threshold_seconds
        
        # If stale and never had data, return None values
        if is_stale and self.num_responses == 0:
            return {
                "total_pnl": 0.0,
                "num_responses": 0,
                "avg_pnl_per_second": None,
                "avg_latency": None,
                "avg_accuracy": None,
            }
        
        # If stale, decay PNL per second towards 0
        effective_pnl_per_second = self.ema_pnl_per_second
        if is_stale:
            # Linearly decay towards 0 over time
            decay_factor = max(0.0, 1.0 - (time_since_last_update - self.stale_threshold_seconds) / 30.0)
            effective_pnl_per_second *= decay_factor
        
        avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else (None if is_stale else 0.0)
        avg_accuracy = sum(self.all_accuracies) / len(self.all_accuracies) if self.all_accuracies else (None if is_stale else 0.0)
        
        return {
            "total_pnl": self.total_pnl,
            "num_responses": self.num_responses,
            "avg_pnl_per_second": effective_pnl_per_second,
            "avg_latency": avg_latency,
            "avg_accuracy": avg_accuracy,
        }


class Scoreboard:
    """Scoreboard that listens for server connections and displays metrics."""

    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port
        self.running = True
        self.clients: Dict[int, ClientStats] = {}
        self.client_counter = 0
        self.lock = threading.Lock()
        self.last_display = 0

    def start(self):
        """Start the scoreboard server."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        print(f"✓ Scoreboard listening on {self.host}:{self.port}")
        
        try:
            # Accept connections in background thread
            accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
            accept_thread.start()
            
            # Main display loop
            self._display_loop()
        except KeyboardInterrupt:
            print("\n\nScoreboard stopped.")
        finally:
            self.running = False
            self.server_socket.close()

    def _accept_loop(self):
        """Accept incoming connections from servers."""
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_socket, addr = self.server_socket.accept()
                
                with self.lock:
                    client_id = self.client_counter
                    self.client_counter += 1
                
                print(f"✓ Server {client_id} connected from {addr}")
                
                # Start a thread to handle this server
                handler_thread = threading.Thread(
                    target=self._handle_server,
                    args=(client_id, client_socket, addr),
                    daemon=True
                )
                handler_thread.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Accept error: {e}")

    def _handle_server(self, server_id: int, client_socket: socket.socket, addr):
        """Handle messages from a connected server."""
        try:
            reader = SocketReader(client_socket)
            
            # Read all messages from this server
            while self.running:
                try:
                    client_socket.settimeout(0.5)
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    
                    reader.buffer += data
                    
                    # Parse all complete messages
                    while b"\n" in reader.buffer:
                        line, reader.buffer = reader.buffer.split(b"\n", 1)
                        if line:
                            msg = ProtocolHandler.decode(line)
                            if msg and isinstance(msg, ScoreUpdate):
                                self._handle_score_update(server_id, msg)
                
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Server {server_id} error: {e}")
                    break
        finally:
            client_socket.close()
            print(f"✗ Server {server_id} disconnected")

    def _handle_score_update(self, server_id: int, msg: ScoreUpdate):
        """Process a score update message from a server."""
        with self.lock:
            if server_id not in self.clients:
                self.clients[server_id] = ClientStats(client_id=server_id)
            
            self.clients[server_id].update(msg)

    def _display_loop(self):
        """Main loop that displays metrics."""
        while self.running:
            now = time.time()
            if now - self.last_display >= 1.0:
                self._display_scores()
                self.last_display = now
            
            time.sleep(0.05)

    def _display_scores(self):
        """Display current scores."""
        # Clear screen (works on Unix/Linux/Mac)
        print("\033[2J\033[H", end="", flush=True)

        print("=" * 80)
        print("GPU INFERENCE GAME - LIVE SCOREBOARD")
        print("=" * 80)
        print()

        with self.lock:
            if not self.clients:
                print("  Waiting for servers to connect...\n")
                return

            for server_id, stats in sorted(self.clients.items()):
                data = stats.get_stats()

                print(f"  Server {server_id}")
                print(f"  {'-' * 76}")
                print(f"    Total PNL:                 ${data['total_pnl']:>10.2f}")
                print(
                    f"    Responses:                 {data['num_responses']:>10,d} responses"
                )
                
                # Format PNL per second (can be None if stale)
                pnl_str = f"${data['avg_pnl_per_second']:>10.2f}" if data['avg_pnl_per_second'] is not None else "        N/A"
                print(f"    Avg PNL/Second:            {pnl_str}")
                
                # Format latency (can be None if stale)
                lat_str = f"{data['avg_latency']:>10.1f} ms" if data['avg_latency'] is not None else "        N/A"
                print(f"    Avg Latency:               {lat_str}")
                
                # Format accuracy (can be None if stale)
                acc_str = f"{data['avg_accuracy']:>10.6f}" if data['avg_accuracy'] is not None else "        N/A"
                print(f"    Avg Accuracy:              {acc_str}")

                print()

        print("=" * 80)
        print("(Press Ctrl+C to stop)")
        print()



def main():
    parser = argparse.ArgumentParser(
        description="Live scoreboard for GPU inference game"
    )
    parser.add_argument("--host", type=str, default="localhost", help="Scoreboard host")
    parser.add_argument("--port", type=int, default=9000, help="Scoreboard port")

    args = parser.parse_args()

    scoreboard = Scoreboard(host=args.host, port=args.port)
    scoreboard.start()


if __name__ == "__main__":
    main()
