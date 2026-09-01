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
        """Sample-weighted aggregation with explicit device handling.

        Client updates are intentionally allowed to reside on CPU after local
        training.  The global model can reside on CUDA, so each update tensor
        is moved to the global tensor's device before arithmetic.  This keeps
        client-state storage memory efficient while avoiding CPU/CUDA mixing.
        """
        if not updates:
            raise ValueError("At least one client update is required.")
        total = sum(int(update.num_samples) for update in updates)
        if total <= 0:
            raise ValueError("Total client sample count must be positive.")

        current = global_model.state_dict()
        new_state = {}
        for key, reference in current.items():
            if reference.is_floating_point() or reference.is_complex():
                accumulation_dtype = (
                    torch.complex64 if reference.is_complex() else torch.float32
                )
                value = torch.zeros_like(
                    reference,
                    dtype=accumulation_dtype,
                    device=reference.device,
                )
                for update in updates:
                    weight = float(update.num_samples) / float(total)
                    local_tensor = update.state_dict[key].to(
                        device=reference.device,
                        dtype=accumulation_dtype,
                    )
                    value.add_(local_tensor, alpha=weight)
                new_state[key] = value.to(
                    device=reference.device, dtype=reference.dtype
                )
            else:
                # Integer/bool buffers are not meaningfully averaged. Preserve
                # the established first-client policy, but place the tensor on
                # the same device as the global-model buffer.
                new_state[key] = updates[0].state_dict[key].to(
                    device=reference.device, dtype=reference.dtype
                )
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
