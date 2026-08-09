import random
import threading
import time
import unittest

import torch.multiprocessing as mp

from edge_arch.sr_dispatcher import Task, run_dispatcher


def fake_sr_worker(sr_queue, init_pipe, task_queue):
    init_pipe.send({(0, 90): 0.001})
    init_pipe.close()
    while True:
        placeholder = sr_queue.get()
        if placeholder is None:
            return
        data = placeholder.get_data()
        if data is None:
            continue
        result_pipe, payload = data
        task_queue.put((Task.START, placeholder.number))
        time.sleep(0.001)
        result_pipe.send(payload)
        result_pipe.close()
        task_queue.put((Task.FINISH, placeholder.number))


def request_wait(task_queue):
    parent, child = mp.Pipe()
    task_queue.put((Task.ACQUIRE, child))
    if not parent.poll(10):
        raise TimeoutError("dispatcher wait query timed out")
    value = parent.recv()
    parent.close()
    return value


def reserve(task_queue):
    parent, child = mp.Pipe()
    task_queue.put((Task.CREATE, (child, 0.001)))
    if not parent.poll(10):
        raise TimeoutError("dispatcher reservation timed out")
    number = parent.recv()
    parent.close()
    return number


class SRDispatcherTest(unittest.TestCase):
    def test_concurrent_submit_preserves_fifo_reservations(self):
        mp.set_start_method("spawn", force=True)
        task_queue = mp.Queue()
        parent_init, child_init = mp.Pipe()
        dispatcher = mp.Process(target=run_dispatcher, args=(task_queue, child_init, fake_sr_worker))
        dispatcher.start()
        self.assertTrue(parent_init.poll(10), "dispatcher did not initialize")
        parent_init.recv()
        parent_init.close()

        schedule_lock = threading.Lock()
        results = {}
        failures = []

        def client(client_id):
            try:
                with schedule_lock:
                    request_wait(task_queue)
                    number = reserve(task_queue)
                parent_result, child_result = mp.Pipe()
                time.sleep(random.random() * 0.005)
                task_queue.put((Task.SEND, (number, (child_result, client_id))))
                if not parent_result.poll(10):
                    raise TimeoutError(f"SR result {number} timed out")
                results[number] = parent_result.recv()
                parent_result.close()
            except Exception as exc:
                failures.append(repr(exc))

        threads = [threading.Thread(target=client, args=(client_id,)) for client_id in range(100)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20)

            self.assertFalse(failures, failures)
            self.assertTrue(all(not thread.is_alive() for thread in threads), "client thread deadlock")
            self.assertEqual(sorted(results), list(range(1, 101)))
            self.assertEqual(sorted(results.values()), list(range(100)))

            deadline = time.monotonic() + 5
            while request_wait(task_queue) > 1e-9 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(request_wait(task_queue), 0.0)
            self.assertTrue(dispatcher.is_alive(), "dispatcher crashed")
        finally:
            if dispatcher.is_alive():
                task_queue.put((None, None))
                dispatcher.join(timeout=10)
            if dispatcher.is_alive():
                dispatcher.terminate()
                dispatcher.join(timeout=5)
        self.assertFalse(dispatcher.is_alive(), "dispatcher did not stop cleanly")

    def test_cancel_unblocks_a_reserved_placeholder(self):
        mp.set_start_method("spawn", force=True)
        task_queue = mp.Queue()
        parent_init, child_init = mp.Pipe()
        dispatcher = mp.Process(target=run_dispatcher, args=(task_queue, child_init, fake_sr_worker))
        dispatcher.start()
        self.assertTrue(parent_init.poll(10), "dispatcher did not initialize")
        parent_init.recv()
        parent_init.close()
        try:
            canceled_number = reserve(task_queue)
            active_number = reserve(task_queue)
            task_queue.put((Task.CANCEL, canceled_number))
            parent_result, child_result = mp.Pipe()
            task_queue.put((Task.SEND, (active_number, (child_result, "after-cancel"))))
            self.assertTrue(parent_result.poll(10), "post-cancel SR result timed out")
            self.assertEqual(parent_result.recv(), "after-cancel")
            parent_result.close()
            deadline = time.monotonic() + 5
            while request_wait(task_queue) != 0.0 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(request_wait(task_queue), 0.0)
            self.assertTrue(dispatcher.is_alive(), "dispatcher crashed after cancellation")
        finally:
            if dispatcher.is_alive():
                task_queue.put((None, None))
                dispatcher.join(timeout=10)
            if dispatcher.is_alive():
                dispatcher.terminate()
                dispatcher.join(timeout=5)
        self.assertFalse(dispatcher.is_alive(), "dispatcher did not stop cleanly")


if __name__ == "__main__":
    unittest.main()
