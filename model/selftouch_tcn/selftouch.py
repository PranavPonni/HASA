import torch
from torch import nn
from torch.nn.utils import weight_norm
from einops import rearrange
from selftouch_loss_utils import active_tactile_loss


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.pad1 = nn.ConstantPad1d((padding, 0), 0)
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, dilation=dilation, padding=0))
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.pad2 = nn.ConstantPad1d((padding, 0), 0)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, dilation=dilation, padding=0))
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        out = self.drop1(self.act1(self.conv1(self.pad1(x))))
        out = self.drop2(self.act2(self.conv2(self.pad2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            pad = dilation * (kernel_size - 1)
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 1, dilation, pad, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SelfTouchTCN(nn.Module):
    """Causal TCN for selftouch prediction.

    Input:  Joint Positions(t)  [hand_dim]
    Output: Predicted Self-touch Tactile(t) for index_tip, thumb_tip, and middle_tip [tactile_dim each]
    """

    def __init__(self, param):
        super().__init__()
        self.hand_dim    = int(param['hand_dim'])
        self.tactile_dim = int(param['tactile_dim'])
        self.input_dim   = self.hand_dim
        self.rec_dim     = int(param['rec_dim'])
        self.dropout     = float(param['dropout'])

        k = int(param.get('kernel_size', 5))
        L = int(param.get('num_levels', 4))
        L = max(L, 2)
        channels = [max(self.rec_dim // 4, 8), max(self.rec_dim // 2, 16)] + [self.rec_dim] * (L - 2)

        self.input_drop = nn.Dropout(self.dropout)
        self.tcn = TCN(self.input_dim, channels, kernel_size=k, dropout=self.dropout)

        # Index, thumb, and middle tactile outputs. Ring is intentionally plot-only
        # because the current dataset does not include ring tactile data.
        self.out_dim = self.tactile_dim * 3
        self.decoder = nn.Conv1d(channels[-1], self.out_dim, kernel_size=1)

    def forward(self, hand_jnt_pos, hand_jnt_cmd_pos=None):
        # Use joint position only: pos(t) -> tactile(t).
        x = hand_jnt_pos[:, 1:, :]
        x = self.input_drop(x)

        # TCN expects (B, C, T)
        h = self.tcn(x.transpose(1, 2))   # (B, rec_dim, T-1)
        y = self.decoder(h)               # (B, out_dim, T-1)
        y = y.transpose(1, 2)              # (B, T-1, out_dim)

        idx = y[:, :, :self.tactile_dim]
        thumb = y[:, :, self.tactile_dim:self.tactile_dim * 2]
        middle = y[:, :, self.tactile_dim * 2:]
        return idx, thumb, middle

    def forward_loss(self, tactile_index_tip, tactile_thumb_tip,
                     hand_jnt_pos, hand_jnt_cmd_pos, data_found=None, loss_coef=None,
                     tactile_middle_tip=None, tactile_ring_tip=None, **_unused):
        idx_pred, thumb_pred, middle_pred = self.forward(hand_jnt_pos, hand_jnt_cmd_pos)

        # Targets: tactile at t=1..T-1 (aligned with pos[:,1:]/cmd_pos[:,:-1] inputs)
        loss_idx   = active_tactile_loss(idx_pred,   tactile_index_tip[:, 1:, :], loss_coef)
        loss_thumb = active_tactile_loss(thumb_pred, tactile_thumb_tip[:, 1:, :], loss_coef)
        loss_middle = active_tactile_loss(middle_pred, tactile_middle_tip[:, 1:, :], loss_coef)

        coef = loss_coef or {}
        total_loss = (
            coef.get('tactile_index_tip', 1.0) * loss_idx +
            coef.get('tactile_thumb_tip', 1.0) * loss_thumb +
            coef.get('tactile_middle_tip', 1.0) * loss_middle
        )
        return (total_loss, loss_idx, loss_thumb, loss_middle), (idx_pred, thumb_pred, middle_pred)


if __name__ == "__main__":
    torch.manual_seed(0)
    param = dict(
        hand_dim=8,
        tactile_dim=90,
        rec_dim=256,
        dropout=0.1,
        kernel_size=5,
        num_levels=4,
    )

    model = SelfTouchTCN(param).eval()

    B, T = 2, 100
    hand_jnt_pos     = torch.randn(B, T, param['hand_dim'])
    hand_jnt_cmd_pos = torch.randn(B, T, param['hand_dim'])
    idx, thumb = model(hand_jnt_pos, hand_jnt_cmd_pos)
    print("idx:", idx.shape, "thumb:", thumb.shape)



class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.pad1 = nn.ConstantPad1d((padding, 0), 0)
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                           stride=stride, dilation=dilation, padding=0))
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.pad2 = nn.ConstantPad1d((padding, 0), 0)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size,
                                           stride=stride, dilation=dilation, padding=0))
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        out = self.drop1(self.act1(self.conv1(self.pad1(x))))
        out = self.drop2(self.act2(self.conv2(self.pad2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            pad = dilation * (kernel_size - 1)
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, 1, dilation, pad, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
