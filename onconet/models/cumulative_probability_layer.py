"""Module implementing a cumulative probability head for survival models.

The :class:`Cumulative_Probability_Layer` predicts a hazard for each
follow-up period and optionally accumulates the hazards into cumulative
probabilities. This is used to transform extracted image features into a
time-dependent risk estimate.

Key constructor arguments
-------------------------
num_features : int
    Dimensionality of the input feature tensor.
max_followup : int
    Number of discrete time steps for which hazards are predicted.
make_probs_indep : bool
    When ``True`` the layer outputs independent hazard predictions without
    the cumulative sum.
"""

import torch
import torch.nn as nn
import pdb



class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, args, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()
        self.args = args
        # Linear layer predicting hazards for each follow-up period
        self.hazard_fc = nn.Linear(num_features, max_followup)
        # Bias term representing baseline hazard
        self.base_hazard_fc = nn.Linear(num_features, 1)
        self.relu = nn.ReLU(inplace=True)
        # Mask used to compute cumulative sums via matrix multiplication
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter('upper_triagular_mask', mask)

    def hazards(self, x):
        # Apply the linear layer and enforce positivity via ReLU
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        if self.args.make_probs_indep:
            # In certain experiments hazards are returned directly
            return self.hazards(x)

        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        # Expand hazards so each time step accumulates past hazards
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)
        masked_hazards = expanded_hazards * self.upper_triagular_mask
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob
