from torch.multiprocessing import Manager
from typing import Dict, Any

class Edge:
    @staticmethod
    def static_init():
        manager = Manager()
        streamers = manager.dict()  # save header and other info of different streams
        streamer_lock = manager.Lock()
        return manager, streamers, streamer_lock

    def __init__(self, static_values):
        manager, streamers, streamer_lock = static_values
        self.manager = manager
        self.streamers = streamers
        self.streamer_lock = streamer_lock

    def receive(self, identifier: str, received: bytes, args: Dict, is_init: bool) -> bytes:
        print(f"[Edge] received: {identifier}, is_init: {is_init}, args: {args}")
        if is_init:
            with self.streamer_lock:
                if identifier not in self.streamers:
                    self.streamers[identifier] = self.manager.dict()
                self.streamers[identifier]['header'] = received
            self.init_args(identifier, args)
            return received
        else:
            return self._handle(identifier, received, args)

    def init_args(self, identifier: str, args: Dict[str, Any]):
        pass

    def _handle(self, identifier: str, received: bytes, args) -> bytes:
        """
        Override this function to handle the video at edge server

        :param identifier: Identify the video to distinguish different streamers
        :param received: Binary data from server
        """
        with self.streamer_lock:
            return self.streamers[identifier]['header'] + received
