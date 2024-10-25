# Adapted from https://github.com/dscripka/piper-sample-generator/blob/master/piper_train/vits/models.py
import sys
import math

import torch

from typing import Optional, Tuple, Union

from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

try:
    from monotonic_align import maximum_path # type: ignore[import-untyped]
except ImportError:
    sys.stderr.write("monotonic_align not found. Please install it using `pip install pip install git+https://github.com/resemble-ai/monotonic_align.git`\n")
    sys.stderr.flush()
    raise

from heybuddy.piper.attentions import Encoder
from heybuddy.piper.common import (
    init_weights,
    sequence_mask,
    rand_slice_segments,
    generate_path
)
from heybuddy.piper.modules import (
    Log,
    ElementwiseAffine,
    ConvFlow,
    Flip,
    DDSConv,
    LayerNorm,
    WN,
    ResBlock1,
    ResBlock2,
    ResidualCouplingLayer,
)

class StochasticDurationPredictor(nn.Module):
    """
    Stochastic Duration Predictor
    """
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        """
        :param in_channels: input channels
        :param filter_channels: filter channels
        :param kernel_size: kernel size
        :param p_dropout: dropout probability
        :param n_flows: number of flows
        :param gin_channels: global conditioning channels
        """
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = Log()
        self.flows = nn.ModuleList()
        self.flows.append(ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
        reverse: bool=False,
        noise_scale: float=1.0
    ) -> torch.Tensor:
        """
        :param x: input tensor
        :param x_mask: input mask
        :param w: target tensor
        :param g: global conditioning tensor
        :param reverse: reverse flag
        :param noise_scale: noise scale
        :return: output tensor
        """
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).type_as(x) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum( # type: ignore[assignment]
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows)) # type: ignore[assignment]
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).type_as(x) * noise_scale

            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw

class DurationPredictor(nn.Module):
    """
    Duration Predictor
    """
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        """
        :param in_channels: input channels
        :param filter_channels: filter channels
        :param kernel_size: kernel size
        :param p_dropout: dropout probability
        :param gin_channels: global conditioning channels
        """
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        :param x: input tensor
        :param x_mask: input mask
        :param g: global conditioning tensor
        :return: output tensor
        """
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

class TextEncoder(nn.Module):
    """
    Text Encoder
    """
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
    ) -> None:
        """
        :param n_vocab: number of vocabularies
        :param out_channels: output channels
        :param hidden_channels: hidden channels
        :param filter_channels: filter channels
        :param n_heads: number of heads
        :param n_layers: number of layers
        :param kernel_size: kernel size
        :param p_dropout: dropout probability
        """
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor
        :param x_lengths: input lengths
        :return: output tensor
        """
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(
            sequence_mask(x_lengths, x.size(2)), 1
        ).type_as(x)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

class ResidualCouplingBlock(nn.Module):
    """
    Residual Coupling Block
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        """
        :param channels: input channels
        :param hidden_channels: hidden channels
        :param kernel_size: kernel size
        :param dilation_rate: dilation rate
        :param n_layers: number of layers
        :param n_flows: number of flows
        :param gin_channels: global conditioning channels
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor]=None,
        reverse: bool=False
    ) -> torch.Tensor:
        """
        :param x: input tensor
        :param x_mask: input mask
        :param g: global conditioning tensor
        :param reverse: reverse flag
        :return: output tensor
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class PosteriorEncoder(nn.Module):
    """
    Posterior Encoder
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        use_weight_norm: bool = True,
    ) -> None:
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param hidden_channels: hidden channels
        :param kernel_size: kernel size
        :param dilation_rate: dilation rate
        :param n_layers: number of layers
        :param gin_channels: global conditioning channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
            use_weight_norm=use_weight_norm,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: input tensor
        :param x_lengths: input lengths
        :param g: global conditioning tensor
        :return: output tensor
        """
        x_mask = torch.unsqueeze(
            sequence_mask(x_lengths, x.size(2)), 1
        ).type_as(x)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

class Generator(torch.nn.Module):
    """
    Generator
    """
    def __init__(
        self,
        initial_channel: int,
        resblock: Optional[str],
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        upsample_rates: Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Tuple[int, ...],
        gin_channels: int = 0,
        use_weight_norm: bool = True,
    ) -> None:
        """
        :param initial_channel: initial channels
        :param resblock: resblock type
        :param resblock_kernel_sizes: resblock kernel sizes
        :param resblock_dilation_sizes: resblock dilation sizes
        :param upsample_rates: upsample rates
        :param upsample_initial_channel: upsample initial channels
        :param upsample_kernel_sizes: upsample kernel sizes
        :param gin_channels: global conditioning channels
        """
        super(Generator, self).__init__()
        self.LRELU_SLOPE = 0.1
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_module = ResBlock1 if resblock == "1" else ResBlock2
        norm_f = weight_norm if use_weight_norm else lambda x, **y: x
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                norm_f(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    resblock_module(ch, k, d, use_weight_norm=use_weight_norm)
                )

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x: torch.Tensor, g: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        :param x: input tensor
        :param g: global conditioning tensor
        :return: output tensor
        """
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            x = up(x)
            xs = torch.zeros(1)
            for j, resblock in enumerate(self.resblocks):
                index = j - (i * self.num_kernels)
                if index == 0:
                    xs = resblock(x)
                elif (index > 0) and (index < self.num_kernels):
                    xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization
        """
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

class Synthesizer(nn.Module):
    """
    Synthesizer for Training
    """
    dp: Union[StochasticDurationPredictor, DurationPredictor]
    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        upsample_rates: Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Tuple[int, ...],
        n_speakers: int = 1,
        gin_channels: int = 0,
        use_sdp: bool = True,
        use_weight_norm: bool = True,
        encoder_use_weight_norm: bool = True
    ) -> None:
        """
        :param n_vocab: number of vocabularies
        :param spec_channels: spectrogram channels
        :param segment_size: segment size
        :param inter_channels: intermediate channels
        :param hidden_channels: hidden channels
        :param filter_channels: filter channels
        :param n_heads: number of heads
        :param n_layers: number of layers
        :param kernel_size: kernel size
        :param p_dropout: dropout probability
        :param resblock: resblock type
        :param resblock_kernel_sizes: resblock kernel sizes
        :param resblock_dilation_sizes: resblock dilation sizes
        :param upsample_rates: upsample rates
        :param upsample_initial_channel: upsample initial channels
        :param upsample_kernel_sizes: upsample kernel sizes
        :param n_speakers: number of speakers
        :param gin_channels: global conditioning channels
        :param use_sdp: use stochastic duration predictor
        """
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            use_weight_norm=use_weight_norm,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
            use_weight_norm=encoder_use_weight_norm,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        :param x: input tensor
        :param x_lengths: input lengths
        :param y: target tensor
        :param y_lengths: target lengths
        :param sid: speaker id
        :return: output tensors
        """
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 1:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
                x_mask
            )  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = rand_slice_segments(
            z, int(y_lengths), self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )

    def infer(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: Optional[torch.Tensor]=None,
        noise_scale: float=0.667,
        length_scale: float=1.0,
        noise_scale_w: float=0.8,
        max_len: Optional[int]=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        :param x: input tensor
        :param x_lengths: input lengths
        :param sid: speaker id
        :param noise_scale: noise scale
        :param length_scale: length scale
        :param noise_scale_w: noise scale for w
        :param max_len: maximum length
        :return: output tensors
        """
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 1:
            assert sid is not None, "Missing speaker id"
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            sequence_mask(y_lengths, int(y_lengths.max())), 1
        ).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return o, attn, y_mask, (z, z_p, m_p, logs_p)
