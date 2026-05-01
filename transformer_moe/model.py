import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


WAVELENGTH_PAIRS = [
    (0.48, 0.532), (0.48, 0.565), (0.48, 0.632),
    (0.532, 0.565), (0.532, 0.632), (0.565, 0.632)
]

EMISSIVITY_FIT_COEFFS = [
    [1],
    [1],
    [1],
    [1],
    [1],
    [1]
]


def get_expert_predictions(gray_ratios_batch):

    gray_ratios_np = gray_ratios_batch.cpu().detach().numpy()
    batch_size = gray_ratios_np.shape[0]
    expert_temps_np = np.zeros_like(gray_ratios_np)

    for i in range(batch_size):
        for j in range(6):
            G_known = gray_ratios_np[i, j]
            coeffs = EMISSIVITY_FIT_COEFFS[j]
            x_calculated = np.polyval(coeffs, G_known)

            temp_calculated_K = 0.0
            if x_calculated > 0:
                lambda1_m, lambda2_m = WAVELENGTH_PAIRS[j][0] * 1e-6, WAVELENGTH_PAIRS[j][1] * 1e-6
                C2_mK = 1.43877e-2
                ratio_term = (lambda2_m / lambda1_m) ** 5
                if ratio_term > 0 and x_calculated / ratio_term > 0:
                    ln_term = np.log(x_calculated / ratio_term)
                    denominator = (1 / lambda2_m - 1 / lambda1_m)
                    if denominator != 0 and ln_term != 0:
                        temp_calculated_K = C2_mK * denominator / ln_term

            expert_temps_np[i, j] = temp_calculated_K

    return torch.tensor(expert_temps_np, dtype=torch.float32, device=gray_ratios_batch.device)




class TransformerGatingNetwork(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2, num_experts=6, seq_len=6):

        super(TransformerGatingNetwork, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(seq_len * d_model, num_experts)

    def forward(self, x):

        x = x.unsqueeze(-1)

        x = self.input_proj(x)

        x = x + self.pos_embedding

        x = self.transformer_encoder(x)

        x = x.flatten(start_dim=1)

        logits = self.output_layer(x)
        return logits


class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts=6):
        super(MixtureOfExperts, self).__init__()

        self.gating_network = TransformerGatingNetwork(
            input_dim=1,
            d_model=64,
            nhead=4,
            num_layers=2,
            num_experts=num_experts,
            seq_len=6
        )

    def forward(self, x):

        gating_logits = self.gating_network(x)
        weights = F.softmax(gating_logits, dim=1)

        expert_outputs = get_expert_predictions(x)

        final_prediction = torch.sum(weights * expert_outputs, dim=1, keepdim=True)
        return final_prediction, weights


if __name__ == '__main__':
    from thop import profile
    from thop import clever_format


    model = MixtureOfExperts()

    dummy_input = torch.randn(1, 6)


    flops, params = profile(model, inputs=(dummy_input,))

    flops_str, params_str = clever_format([flops, params], "%.3f")


