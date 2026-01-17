# Optimizations Walkthrough

This directory is a guided tour through the process of making a slow model fast.

The goal is not just to end up with a fast system, but to understand *why* each change matters, how you would discover it through profiling, and how small decisions compound into large performance gains.

Each subdirectory represents a checkpoint in that journey. They are ordered the way you are meant to read them—and roughly the order you would naturally discover them if you were profiling and optimizing from scratch.

---

## What You’re Optimizing

The base model comes from the original hackathon repo. It is intentionally slow and intentionally interesting: an ensemble of four sequence models (Mamba2, RetNet, Hawk, xLSTM) that maintains per-symbol state and processes a stream of requests.

Every request:

* Has ~79 numeric features
* Belongs to a symbol (e.g. `SYM_042`)
* Must update and reuse symbol-specific state
* Produces a single prediction

This structure creates all the classic performance problems:

* Too many tiny kernel launches
* CPU/GPU syncs in bad places
* Poor batching
* Python overhead on hot paths
* Missed opportunities for compilation and graph capture

Which makes it perfect for learning.

---

## How This Repo Is Organized

* `optimizations` is a folder containing optimized variants of the example client from the hackathon.

Each subdirectory contains a full working variant of the client:

* `model.py`: the model wrapper and hot path

You can run any version independently. The only difference between them is the optimization idea being demonstrated.

The folders are numbered to enforce a learning order:

```
optimizations/
  0_baseline/
  1_minibatch_interleaved_by_symbol/
  2_compile/
  3_cuda_graphs/
  4_index_minibatch_by_symbol/
  5_reduce_syncs/
```

Each step assumes you understand the previous one.

---

## How To Use This

Treat this like a lab notebook, assuming you are running on a machine with H100.

* Cached files coming soon GitHub releaseto follow along with no machine

Install all the necessary dependencies (I'm using CPython 3.10)

`pip install -r requirements.txt`

Start in the optimizations directory:

`cd optimizations`

For each directory:

1. Run `local_evaluator.py <variant subdir name>` to see latency and accuracy
2. Run `profiler.py  <variant subdir name>` to capture traces
3. Load into chrome trace `chrome://tracing` and compare against the previous version, paying particular attention to wall clock time at the top and the GPU stream at the bottom
4. Read the code diff and explain to yourself why it helped

The goal is not just to get fast—it’s to be able to look at a slow system and *know what to try next*.

If you finish this and feel like you can walk into any inference system and start making it faster with confidence, then it worked.

---

## The Walkthrough

Each step is written as if you don’t know the future. You only know what the profiler just told you, and you’re reacting to that evidence.

---

### 0_baseline

You start with something that works, but feels slow.

Profiling shows:

* Tons of tiny kernel launches
* Heavy Python time around every request
* The GPU often waiting on the CPU

The picture is clear: you’re paying overhead per request. So the natural idea is: stop doing things one-by-one.

---

### 1_minibatch_interleaved_by_symbol

You realize you’re overhead-bound, not compute-bound.

Profiling the baseline shows a pattern: the GPU isn’t busy most of the time. Kernels are tiny and frequent. The CPU is busy shepherding work rather than the GPU doing math.

This is the classic low–algorithmic-intensity problem. Algorithmic intensity means “how much math you do per byte moved or per unit of overhead.” Right now, each request does very little work, but pays a lot of fixed costs: Python overhead, framework dispatch, kernel launches. You are overhead-bound, not compute-bound.

So the natural move is to increase algorithmic intensity: do more math per unit of overhead.

Batching does exactly that. By grouping many requests into one call:

* You pay Python and launch overhead once per batch, not once per request
* The GPU gets larger kernels that do more work per launch
* You move from being overhead-bound toward being compute-bound

But state is per-symbol, so you can’t just stack everything blindly. You end up interleaving by symbol and batching what you can while respecting state structure.

Profiling now shows:

* Much fewer launches
* Bigger kernels
* Higher GPU utilization

But you’re still not compute-bound. The profiler now points at the *framework and Python itself* as a major cost.

---

### 2_compile

You are still overhead-bound—just at a different layer.

After batching, you fixed the worst form of overhead (per-request launches), but you’re still paying a lot for:

* Python control flow
* Dispatcher logic
* Small unfused ops between big ones

The math is fast enough that the surrounding machinery is now the bottleneck. You are still not truly compute-bound; you’re just bound by a different kind of overhead.

So you try compilation with `torch.compile`.

Compilation increases algorithmic intensity again, but in a subtler way:

* Multiple ops get fused into fewer kernels
* Python overhead gets lifted out of the hot path
* More work happens per kernel launch

Profiling after this shows:

* Fewer Python frames
* Fewer, larger kernels
* Higher effective work per launch

You’re getting closer to being compute-bound. But now another pattern appears: even though the same kernels run every iteration, you still pay launch overhead every time.

### 3_cuda_graphs

You notice repetition.

With compilation, the kernels are good—but you see the same sequence of kernels being launched every iteration. The shapes don’t change much. The logic doesn’t change much. Yet you still pay launch cost every time.

So you try capturing the whole thing as a CUDA Graph.

Profiling now shows:

* Much lower launch overhead
* Tighter GPU timelines
* But more rigidity: shapes and control flow suddenly matter a lot

You’ve traded flexibility for speed. And now another cost becomes visible.

---

### 4_index_minibatch_by_symbol

You realize structure controls whether you stay on the fast path.

After CUDA Graphs, a new failure mode appears: sometimes you’re blazing fast, and sometimes you suddenly fall off a cliff. Profiling shows why — small changes in shapes, batch layout, or control flow trigger graph re-capture and setup all over again.

Some of that instability isn’t coming from the model. It’s coming from how you build the batch:

* Different symbols produce different layouts

* Control flow depends on batch contents

So this step is about redesigning batching itself so it becomes stable:

* Fixed layouts

* Predictable indexing

* Consistent shapes and control flow

This does two things at once:

* It makes batching cheaper

The profiler had already shown that building the batch was expensive: Python loops, dicts, appends, symbol grouping. Moving to structured indexing and tensor-based layouts cuts that overhead, and allows you to reuse your state tensors without reallocation.

* It keeps you on the captured graph
By stabilizing shapes and structure, you stop triggering graph re-capture. You don’t just make batching faster — you preserve all the speed CUDA Graphs gave you.

Profiling now shows:

* Less Python time in batch construction

* Fewer graph re-captures

* More iterations staying on the fast replay path

At this point, most of the visible work is real work. But latency still isn’t quite as low as you expect — and the profiler shows something subtler than kernels or Python.

---

### 5_reduce_syncs

You chase invisible pauses.

The profiler shows gaps: places where nothing seems to be happening. No big kernels. No heavy Python. Just waiting.

Digging deeper, you find implicit synchronizations:

* Copies that block later work
* Operations on the default stream that serialize everything
* Accidental CPU/GPU sync points

So this step is about removing those hidden brakes and overlapping work where possible.

---