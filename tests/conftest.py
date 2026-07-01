import torch


def pytest_sessionstart(session):
    # Tiny test tensors are much faster and more deterministic without CPU thread oversubscription.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
