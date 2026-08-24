"""Linear probe on frozen tokenizer features: standardize, then a single linear layer."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm


class LinearProbe(nn.Module):
    def __init__(self, dim: int, mu: torch.Tensor, sigma: torch.Tensor):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc((x - self.mu) / self.sigma).squeeze(-1)


@torch.no_grad()
def extract_features(extract_fn, loader, device: str):
    fs, ys = [], []
    for x, y in loader:
        fs.append(extract_fn(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(fs), torch.cat(ys)


def train_linear_probe(
    extract_fn, train_loader, val_loader, test_loader, device: str,
    ckpt_path: str, epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-4,
):
    """Trains on train_loader, model-selects on val_loader (best val AUC), reports
    clean test AUC, and saves the best state dict to ckpt_path."""
    Xtr, Ytr = extract_features(extract_fn, train_loader, device)
    Xva, Yva = extract_features(extract_fn, val_loader, device)

    probe = LinearProbe(Xtr.shape[1], Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)).to(device)
    opt = torch.optim.Adam(probe.fc.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    Xtr_d, Ytr_d = Xtr.to(device), Ytr.float().to(device)

    best_auc, best_state = -1.0, None
    for _ in tqdm(range(epochs), desc="probe train", leave=False):
        probe.train()
        opt.zero_grad()
        loss_fn(probe(Xtr_d), Ytr_d).backward()
        opt.step()
        probe.eval()
        with torch.no_grad():
            vl = probe(Xva.to(device)).cpu().numpy()
        va = roc_auc_score(Yva.numpy(), 1 / (1 + np.exp(-vl)))
        if va > best_auc:
            best_auc = va
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    probe.eval()
    torch.save(best_state, ckpt_path)

    Xte, Yte = extract_features(extract_fn, test_loader, device)
    with torch.no_grad():
        te_lg = probe(Xte.to(device)).cpu().numpy()
    test_auc = roc_auc_score(Yte.numpy(), 1 / (1 + np.exp(-te_lg)))
    return probe, best_auc, test_auc
