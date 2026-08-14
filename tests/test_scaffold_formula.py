import torch
def test_sign():
    assert torch.allclose(torch.tensor([2.0])-torch.tensor([0.5])+torch.tensor([0.25]),torch.tensor([1.75]))
