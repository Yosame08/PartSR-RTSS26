import threading
import unittest
import uuid

from config import config
from edge_arch.edge_PartSR import EdgePartSR


class SlowestClientAggregationTest(unittest.TestCase):
    def test_uses_maximum_client_sr_latency(self):
        edge = object.__new__(EdgePartSR)
        fast_id = str(uuid.uuid4())
        slow_id = str(uuid.uuid4())
        edge.streamers = {
            "video": {
                fast_id: {size: float(size) for size in config["sr_sizes"]},
                slow_id: {size: float(size * 2) for size in config["sr_sizes"]},
            }
        }
        edge.streamer_lock = threading.Lock()

        result = edge._EdgePartSR__get_slowest("video")

        self.assertEqual(
            result,
            {size: float(size * 2) for size in config["sr_sizes"]},
        )


if __name__ == "__main__":
    unittest.main()
