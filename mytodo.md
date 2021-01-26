# misc
```python
def get_embedding_size(n: int, max_size: int = 100) -> int:
    """
    Determine empirically good embedding sizes (formula taken from fastai).

    Args:
        n (int): number of classes
        max_size (int, optional): maximum embedding size. Defaults to 100.

    Returns:
        int: embedding size
    """
    if n > 2:
        return min(round(1.6 * n ** 0.56), max_size)
    else:
        return 1
```

# Left at 
* pytorch-forecasting/my_package/tft_2.py:498
* order of rows 
* output is standardized ((x-mu)/std) and also a softplus is applied to it after
  * try x/x.max() and also trainable softplus output activation??
  * have a look at the code to understand if target is normalized
# Embedding of categorical and static variables 
## categorical variables 
for categorical variables we use nn.Embedding
## real variables 
for real variables we use just the real value (size=1)

# VariableSelectionNetwork 
var_outputs = []
weight_inputs = []
## categorical variables 
variable_embedding = cat_embedding
weight_inputs.append(variable_embedding)
1. pass through `ResampleNorm(input_size, hidden_size)` where input_size is the embedding size and hidden size is a hyperparameter (16) -> torch.Size([1, 16])
var_outputs.append(torch.Size([1, 16]))
## real variables 
name = 'step'
1. pass through prescalers[name] = nn.Linear(1, hidden_continuous_size) where hidden_continuous_size is a hyperparameter (8) -> torch.Size([1, 8])
weight_inputs.append(torch.Size([1, 8]))
2.  pass torch.Size([1, 8]) through GatedResidualNetwork -> torch.Size([1, 16])
var_outputs.append(torch.Size([1, 16]))
## stack outputs
var_outputs = torch.stack(var_outputs, dim=-1) -> torch.Size([1, 16, 5])
## flatten weight_inputs
flat_embedding = torch.cat(weight_inputs, dim=-1)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x

class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class ResampleNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0

        output = self.norm(x)
        return output

    
hidden_continuous_size = 8 
single_variable_grns = nn.ModuleDict()
prescalers = nn.ModuleDict()
name = 'market_id'
input_size = 16 # original nn.Embedding() size
hidden_size = 16 # hyperparameter
single_variable_grns[name] = ResampleNorm(input_size, hidden_size)
dropout = 0.1
input_size = 8
name= 'step'
single_variable_grns[name] = GatedResidualNetwork(input_size, min(input_size, hidden_size), output_size=hidden_size, dropout=dropout)
prescalers[name] = nn.Linear(1, hidden_continuous_size)
```