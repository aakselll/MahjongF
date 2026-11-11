"""
Replay buffer partage entre workers et trainer.

[FIX Windows] Le Manager et la liste partagee sont crees DANS le process
principal. Les workers recoivent uniquement la liste proxy (picklable).
"""
import multiprocessing as mp
import numpy as np


class ReplayBuffer:
    """Conteneur du buffer. Cree dans le main, jamais envoye aux workers."""

    def __init__(self, max_size=10000, manager=None):
        self.max_size = max_size
        if manager is None:
            manager = mp.Manager()
            self._owned_manager = manager
        else:
            self._owned_manager = None
        self.data = manager.list()

    def add(self, transitions):
        self.data.append(transitions)
        while len(self.data) > self.max_size:
            self.data.pop(0)

    def sample(self, n):
        if len(self.data) < n:
            return None
        idx = np.random.choice(len(self.data), n, replace=False)
        batch = [self.data[i] for i in idx]
        return [tr for ep in batch for tr in ep]

    def __len__(self):
        return len(self.data)


# ---------------------------------------------------------------------- #
# Helper functions picklables pour les workers
# ---------------------------------------------------------------------- #
def buffer_add(data, max_size, transitions):
    """Version picklable de add()."""
    data.append(transitions)
    while len(data) > max_size:
        data.pop(0)