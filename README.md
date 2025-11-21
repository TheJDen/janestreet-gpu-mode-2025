## The Hackathon Challenge

### What You're Optimizing
You're given a pre-built neural network that's **intentionally slow**. Your job is to make it fast without breaking its accuracy. The model is a complex ensemble of 4 different sequence models (Mamba2, RetNet, Hawk, and xLSTM) that process streaming data for different "symbols" (think stock tickers).

### Your Goal

You need to balance 3 different metrics
- **Latency**: Minimize latency so you can adapt to market conditions
- **Accuracy**: Keep a a small MSE so the model quality is still good
- **Throughput**: Handle as many requests/second as possible

### The Competition Flow
1. You receive a stream of inference requests with 79 features each
2. Each request belongs to a specific symbol (SYM_001, SYM_002, etc.)
3. The model maintains separate state for each symbol (like memory)
4. You predict a single value per request
5. The server scores you in real-time based on speed and accuracy

## Participant Guide

Once you are setup on Northflank, you should see various services on your team:
- `code-participant-x`: those are to hack around individually and try to make changes to the model.
- `code-team`: this is for the client that will connect to the server and run the inference model.

All of those have a single H100.

### Connect to the server

To connect to your server (from the `code-team` box), run the command
```bash
python team/example_model.py --host $SERVER_HOST --port 8001
```

You should see some print statements in the terminal you ran this and some changes in your team on the leaderboard.

The `code-team` box shares data with the participant boxes (you should see `partipant-1` etc. subfolders). This makes it easy to roll changes developed on participant boxes. Make sure you make a backup of your team subfolder before rolling a new version however, in case you need to quickly revert your changes.

### Test changes locally

You can run `python local_evaluator.py` to run the client you have (as defined in `example_model.py`) on some test data. It will give you the average latency of your model as well as the error you got compared to the theoretical prediction. It's a great way to check whether your changes have an impact on latency/accuracy.

This is also a great script to profile to see where your client is spending most of its time.

### What You Need to Change

The main optimization target is the `process_batch()` method in `example_model.py`. This method currently processes requests one-by-one, which is inefficient. Your job is to make it faster through:

### 1. **PyTorch-Level Optimizations** (Easier)
- Use `torch.compile()` to JIT-compile the model
- Implement mixed precision (fp16/bf16) computation

### 2. **Algorithmic Optimizations** (Medium)
- Implement dynamic batching strategies

### 3. **Custom GPU Kernels** (Advanced)
The model has several expensive operations that are perfect targets for custom kernels:

**Key Bottlenecks to Target:**
- **Mamba2 SSM Updates** (`client/model/mamba2.py:90-91`): 4D tensor operations with broadcasting
- **RetNet Rotary Embeddings** (`client/model/retnet.py:72`): Outer products and position encoding
- **xLSTM Cell Updates** (`client/model/xlstm.py:59-62`): Complex gated state updates
- **Causal Convolutions**: Depthwise convs with state management

All the above are just suggestions, please be creative, try things out, profile and find the important bottlenecks.

## Quickstart (local)

These steps help you get a local environment running with the new batched data generation and the inverted-scoreboard architecture (scoreboard listens; server connects and pushes metrics).

### Step 0: Generate reference data (optional, for testing)

The data generator now runs the model in batches of `num_symbols` per step (one row per symbol per step). This is faster and preserves per-symbol state.

```bash
# small quick test (fast)
python generate_data.py --output small_ref.parquet --num-symbols 4 --rows-per-symbol 5

# full generation (example)
python generate_data.py --output reference.parquet --num-symbols 20 --rows-per-symbol 1000
```

### Step 1: Start the scoreboard

```bash
python scoreboard.py --host localhost --port 9000
```

**Why first?** The scoreboard listens for incoming server connections. Start it before the server so the server can connect immediately.

**What you'll see:** Live dashboard updating every second with connected servers and their metrics:
- Total PNL (cumulative)
- Avg PNL/Second (exponential moving average, recent-biased)
- Avg Latency (last 100 requests)
- Avg Accuracy (last 400 tower predictions)
- Stale servers marked as "N/A" after 30 seconds of silence

### Step 2: Start the server

```bash
python server.py --data-file reference.parquet --host 0.0.0.0 --port 8080 --scoreboard-host localhost --scoreboard-port 9000
```

**What happens:**
- Loads reference data from parquet
- Attempts to connect to scoreboard at startup (non-blocking if unavailable)
- Begins streaming batched inference requests (one row per symbol per request)
- Scores client responses and pushes metrics to scoreboard

**Optional flags:**
- `--mean-request-interval 100` (default): average milliseconds between request batches

### Step 3: Connect a client

```bash
# example model client (requires GPU)
python example_model.py --host localhost --port 8080  --token <HF_TOKEN>
```

**What clients do:**
- Connect to server, register, and begin receiving `InferenceRequest` messages
- Process requests and send back `InferenceResponse` with predictions
- Completely unaware of the scoreboard (clean separation)

You can start **multiple clients** in parallel (each in a different terminal).

---

### Quick Example (all-in-one)

Terminal 1 — Scoreboard:
```bash
python scoreboard.py --host localhost --port 9000
```

Terminal 2 — Server:
```bash
python server.py --data-file reference.parquet --host 0.0.0.0 --port 8080
```

Terminal 3+ — Client(s):
```bash
python example_model.py --host localhost --port 8080  --token <HF_TOKEN>
```

Watch the scoreboard update in real-time as the client processes requests!

---

### Tips & Troubleshooting

**Tips:**
- For quick experiments use a small `--num-symbols` and `--rows-per-symbol` when generating data.
- The server sends `InferenceRequest` messages only to inference clients and `ScoreUpdate` messages only to the scoreboard (clients don't know about the scoreboard).

**Troubleshooting:**
- **Server can't connect to scoreboard:** Server will print a warning and continue; start the scoreboard separately and restart the server if you want metrics recorded.
- **Scoreboard shows "N/A" for server:** The server hasn't sent updates recently (stale after 30s). Restart the client or check network connectivity.
- **No data in scoreboard display:** Ensure at least one client is connected and sending responses. The scoreboard only shows metrics from servers that have sent score updates.


