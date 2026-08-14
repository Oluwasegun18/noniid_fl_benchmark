from __future__ import annotations

from abc import ABC

import torch

from flbench.experiment.resources import state_dict_num_bytes
from flbench.training.types import ClientUpdate


class FederatedAlgorithm(ABC):
    name = "base"

    def initialize(self, global_model, num_clients, cfg):
        self.num_clients = int(num_clients)

    def client_update(self, client_id, global_model, loader, device, cfg):
        raise NotImplementedError

    def server_update(self, global_model, updates, cfg):
        self.weighted_average(global_model, updates)

    @staticmethod
    def weighted_average(global_model, updates):
        if not updates:
            raise ValueError("At least one client update is required.")
        total = sum(update.num_samples for update in updates)
        if total <= 0:
            raise ValueError("Total client sample count must be positive.")

        current = global_model.state_dict()
        new_state = {}
        for key, reference in current.items():
            if reference.is_floating_point() or reference.is_complex():
                value = torch.zeros_like(reference, dtype=torch.float32)
                for update in updates:
                    value += (
                        update.state_dict[key].float()
                        * (update.num_samples / total)
                    )
                new_state[key] = value.to(reference.dtype)
            else:
                new_state[key] = updates[0].state_dict[key].to(reference.dtype)
        global_model.load_state_dict(new_state, strict=True)

    def download_num_bytes(self, global_model, selected_clients: int) -> int:
        return state_dict_num_bytes(global_model.state_dict()) * int(
            selected_clients
        )

    def client_upload_num_bytes(self, update: ClientUpdate) -> int:
        return state_dict_num_bytes(update.state_dict)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        del state
