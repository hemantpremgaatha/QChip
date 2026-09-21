"""Execute a schedule for real: one worker process per machine, each running its own list."""
import multiprocessing as mp
import time

from .tasks import Task


def _worker(task_specs, q):
    t0 = time.perf_counter()
    for kind, size in task_specs:
        Task(kind, size).run()
    q.put(time.perf_counter() - t0)


def execute(tasks, assign, machines):
    """Run the assignment on `machines` worker processes; returns wall-clock seconds."""
    lists = [[] for _ in range(machines)]
    for t, a in zip(tasks, assign):
        lists[a].append((t.kind, t.size))
    q = mp.Queue()
    procs = [mp.Process(target=_worker, args=(lst, q)) for lst in lists]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    for _ in procs:
        q.get()
    wall = time.perf_counter() - t0
    for p in procs:
        p.join()
    return wall
