import unittest

from edge_arch.sr_wait_tracker import SRWaitTracker


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


class SRWaitTrackerTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(10.0)
        self.tracker = SRWaitTracker(self.clock)

    def test_tracks_active_and_queued_work(self):
        self.tracker.reserve(1, 1.5)
        self.tracker.reserve(2, 2.0)
        self.assertEqual(self.tracker.estimated_wait(), 3.5)

        self.tracker.start(1)
        self.clock.now = 10.4
        self.assertAlmostEqual(self.tracker.estimated_wait(), 3.1)

        self.tracker.finish(1)
        self.assertEqual(self.tracker.estimated_wait(), 2.0)
        self.tracker.start(2)
        self.clock.now = 12.5
        self.assertEqual(self.tracker.estimated_wait(), 0.0)
        self.tracker.finish(2)
        self.assertEqual(self.tracker.estimated_wait(), 0.0)

    def test_does_not_drop_tasks_after_historical_queue_limit(self):
        for number in range(1, 101):
            self.tracker.reserve(number, 0.1)
        self.assertEqual(self.tracker.pending_count, 100)
        self.assertAlmostEqual(self.tracker.estimated_wait(), 10.0)

    def test_rejects_out_of_order_transitions(self):
        self.tracker.reserve(1, 1.0)
        self.tracker.reserve(2, 1.0)
        with self.assertRaisesRegex(RuntimeError, "started out of order"):
            self.tracker.start(2)
        self.tracker.start(1)
        with self.assertRaisesRegex(RuntimeError, "active task"):
            self.tracker.finish(2)

    def test_rejects_invalid_predictions(self):
        with self.assertRaises(ValueError):
            self.tracker.reserve(1, -0.1)
        with self.assertRaises(ValueError):
            self.tracker.reserve(1, float("nan"))

    def test_can_cancel_a_pending_task(self):
        self.tracker.reserve(1, 1.0)
        self.tracker.reserve(2, 2.0)
        self.tracker.cancel(1)
        self.assertEqual(self.tracker.pending_count, 1)
        self.assertEqual(self.tracker.estimated_wait(), 2.0)
        self.tracker.start(2)
        self.tracker.finish(2)


if __name__ == "__main__":
    unittest.main()
