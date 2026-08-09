from enum import Enum

from torch import multiprocessing as mp

from .sr_wait_tracker import SRWaitTracker


class Task(Enum):
    ACQUIRE = 1
    CREATE = 2
    SEND = 3
    START = 4
    FINISH = 5
    CANCEL = 6


class Placeholder:
    def __init__(self, number, manager):
        self.number = number
        self.data_queue = manager.Queue(maxsize=1)

    def set_data(self, data):
        self.data_queue.put(data)

    def get_data(self):
        return self.data_queue.get()


def run_dispatcher(task_queue, init_pipe, worker_target):
    """Own SR FIFO state and isolate all cross-thread task numbering."""
    manager = mp.Manager()
    sr_queue = mp.Queue()
    parent_init_pipe, child_init_pipe = mp.Pipe()
    worker = mp.Process(target=worker_target, args=(sr_queue, child_init_pipe, task_queue))
    worker.start()
    child_init_pipe.close()

    try:
        try:
            edge_sr_time = parent_init_pipe.recv()
        finally:
            parent_init_pipe.close()
        init_pipe.send(edge_sr_time)
        init_pipe.close()

        next_number = 0
        placeholders = {}
        wait_tracker = SRWaitTracker()

        while True:
            task, data = task_queue.get()
            if task is None:
                break

            if task == Task.ACQUIRE:
                dispatch_pipe = data
                dispatch_pipe.send(wait_tracker.estimated_wait())
                dispatch_pipe.close()
            elif task == Task.CREATE:
                dispatch_pipe, predicted_seconds = data
                next_number += 1
                task_number = next_number
                placeholder = Placeholder(task_number, manager)
                placeholders[task_number] = placeholder
                wait_tracker.reserve(task_number, predicted_seconds)
                sr_queue.put(placeholder)
                dispatch_pipe.send(task_number)
                dispatch_pipe.close()
            elif task == Task.SEND:
                task_number, value = data
                try:
                    placeholders[task_number].set_data(value)
                except KeyError as exc:
                    raise RuntimeError(f"unknown SR task {task_number}") from exc
            elif task == Task.CANCEL:
                task_number = data
                try:
                    placeholder = placeholders.pop(task_number)
                except KeyError as exc:
                    raise RuntimeError(f"unknown SR task {task_number}") from exc
                wait_tracker.cancel(task_number)
                placeholder.set_data(None)
            elif task == Task.START:
                wait_tracker.start(data)
            elif task == Task.FINISH:
                wait_tracker.finish(data)
                del placeholders[data]
            else:
                raise RuntimeError(f"unknown SR dispatcher task: {task!r}")
    finally:
        sr_queue.put(None)
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        manager.shutdown()
