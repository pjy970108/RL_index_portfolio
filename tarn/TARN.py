import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- FFC Layers ---
class DCCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, dropout_rate):
        super(DCCLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)  # 추가됨

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)           
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SACLayer(nn.Module):
    def __init__(self, in_channels, head_num, scale_size, stride, dropout_rate):
        super(SACLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=head_num, batch_first=True)
        self.scale = nn.Parameter(torch.ones(1) * scale_size)
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)  # ✅ 추가
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.attn_weights = None

    def forward(self, x):
        x_reshaped = x.transpose(1, 2)  # Reshape for attention
        attn_output, attn_weights = self.attention(x_reshaped, x_reshaped, x_reshaped)
        self.attn_weights = attn_weights.detach().cpu().numpy()  # 나중에 확인 가능하도록 저장

        attn_output = attn_output.transpose(1, 2)  # Reshape back
        attn_output = self.conv(attn_output)
        attn_output = self.bn(attn_output)         # ✅ 추가
        attn_output = self.relu(attn_output)
        attn_output = self.dropout(attn_output)
        return x + attn_output * self.scale


class FFCBlock(nn.Module):
    def __init__(self, structure, hyperparams, use_residual=True):
        super(FFCBlock, self).__init__()
        self.use_residual = use_residual  # 추가된 플래그
        ddc_config = hyperparams["ddc_configs"][structure - 1]
        
        if self.use_residual:
            self.residual_conv = nn.Conv1d(
                in_channels=ddc_config["in_channels"],
                out_channels=ddc_config["residual_out_channels"],
                kernel_size=ddc_config["residual_kernal"],
            )
        
        self.ddc1 = DCCLayer(
            in_channels=ddc_config["in_channels"],
            out_channels=ddc_config["out_channels"],
            kernel_size=ddc_config["kernel_size"],
            stride=ddc_config["stride"],
            padding=ddc_config["padding"],
            dilation=ddc_config["dilation"],
            dropout_rate=hyperparams["dcc_dropout"]
        )
        self.ddc2 = DCCLayer(
            in_channels=ddc_config["out_channels"],
            out_channels=ddc_config["out_channels"],
            kernel_size=ddc_config["kernel_size"],
            stride=ddc_config["stride"],
            padding=ddc_config["padding"],
            dilation=ddc_config["dilation"],
            dropout_rate=hyperparams["dcc_dropout"]
        )
        self.sac = SACLayer(
            in_channels=ddc_config["out_channels"],
            head_num=hyperparams["sac_heads"],
            scale_size=ddc_config["sac_scale"],
            stride=1,
            dropout_rate=hyperparams["sac_dropout"]
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.use_residual:
            residual = self.residual_conv(x)  # Apply Conv1D to the input
        else:
            residual = 0  # Residual 비활성화 시 0으로 설정

        ddc1_out = self.ddc1(x)
        ddc2_out = self.ddc2(ddc1_out)
        sac_out = self.sac(ddc2_out)
        attn_scores = self.sac.attn_weights  # 여기에 numpy array 형태로 attention weight 있음

        combined = residual + ddc2_out + sac_out  # Combine residual and outputs
        return self.relu(combined)  # Apply ReLU


class FFCModule(nn.Module):
    def __init__(self, hyperparams):
        super(FFCModule, self).__init__()
        self.structure1 = FFCBlock(1, hyperparams, use_residual=False)  # residual 사용
        self.structure1 = FFCBlock(1, hyperparams, use_residual=True)  # residual 사용
        self.structure2 = FFCBlock(2, hyperparams, use_residual=True)  # residual 사용
        self.structure3 = FFCBlock(3, hyperparams, use_residual=False)  # residual 미사용
        final_conv_config = hyperparams["final_conv_config"]
        self.final_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=final_conv_config["in_channels"],
                out_channels=final_conv_config["out_channels"],
                kernel_size=final_conv_config["kernel_size"],
                stride=final_conv_config["stride"],
                padding=final_conv_config["padding"]
            ),
            nn.ReLU()
        )
        # flatten
        # self.flatten = nn.Flatten(start_dim=0, end_dim=1)  # Combine Batch and Channels

    def forward(self, x):
        x = self.structure1(x)
        attn_scores_1 = self.structure1.sac.attn_weights  # 첫 번째 블록의 attention score

        x = self.structure2(x)
        x = self.structure3(x)
        x = self.final_conv(x)  # Apply final convolution
        batch_size = x.size(-1)
        x = x.view(batch_size, -1)  # Flatten to [batch_size, feature_dim]


        return x, attn_scores_1