from __future__ import annotations
from typing import Any
import torch

class LocalTrainingHooks:
    def before_training(self, context:dict[str,Any])->None: pass
    def compute_loss(self, model, inputs, targets, context):
        return context['criterion'](model(inputs),targets)
    def after_backward(self, model, context)->None: pass
    def optimizer_step(self, model, optimizer, context)->None: optimizer.step()
    def after_training(self, model, context)->dict[str,Any]: return {}
