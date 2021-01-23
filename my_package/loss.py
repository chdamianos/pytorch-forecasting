"""
1.
y_pred.shape -> torch.Size([1, 45, 7])
target.shape -> torch.Size([1, 45])

2. calculate length
lengths = torch.full((target.size(0),), fill_value=target.size(1), dtype=torch.long, device=target.device)
lengths = tensor([45])

3. calculate quantile loss
quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # calculate quantile loss
    losses = []
    for i, q in enumerate(self.quantiles):
        errors = target - y_pred[..., i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
    losses = torch.cat(losses, dim=2)

    return losses
torch.Size([1, 45, 7])
losses.sum()=tensor(505.5811)
(losses/7).sum() = tensor(72.2259)
4. normalize 
losses = losses/losses.size(-1)
losses = losses/7
losses = tensor(72.2259)

5. sum losses
losses.sum()

6. divide over length
losses.sum() / lengths.sum()
72.2259/45 = 1.60502

"""
