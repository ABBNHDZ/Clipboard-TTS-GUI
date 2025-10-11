"""
HiFi-GAN 
Ce module définit :
- ResBlock1 / ResBlock2 : blocs résiduels utilisés par le générateur
- HifiganGenerator : générateur principal (MRF + upsampling)
- HifiDecoder : wrapper combinant le générateur et un speaker encoder
"""

import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
import gc
import torch_directml

import logging

logger = logging.getLogger(__name__)

# Importer un encodeur de locuteur (version existante dans xtts2_m)
from xtts2m.speaker_encoder import ResNetSpeakerEncoder

LRELU_SLOPE = 0.1

def _prenormalize_conv_weight(layer: nn.Module):
    """Convertit une couche avec weight_norm en couche normalisée fixe."""
    if hasattr(layer, "parametrizations") and "weight" in layer.parametrizations:
        param = layer.parametrizations.weight[0]
        if hasattr(param, "original"):
            v = param.original
            g = getattr(param, "weight_g", None)
            dim = getattr(param, "dim", 0)
            if g is not None:
                with torch.no_grad():
                    w = g * v / torch.norm(v, dim=dim, keepdim=True)
                layer.weight = nn.Parameter(w)
        remove_parametrizations(layer, "weight")

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calcule le padding nécessaire pour conserver la taille temporelle."""
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):
    """Bloc résiduel multi-dilation (type 1) utilisé dans HiFi-GAN."""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_parametrizations(layer, "weight")
        for layer in self.convs2:
            remove_parametrizations(layer, "weight")


class ResBlock2(nn.Module):
    """Bloc résiduel type 2 (structure plus courte)."""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
        ])

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_parametrizations(layer, "weight")


class HifiganGenerator(nn.Module):
    """Générateur HiFi-GAN avec Multi-Receptive Field Fusion (MRF).

    Ce générateur prend en entrée des latents (features) et produit une
    forme d'onde via des couches de transposed convolution et des blocs résiduels.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,
        resblock_dilation_sizes,
        resblock_kernel_sizes,
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_pre_weight_norm=True,
        conv_post_weight_norm=True,
        conv_post_bias=True,
        cond_in_each_up_layer=False,
        pre_linear=None,
    ):
        r"""HiFiGAN Generator with Multi-Receptive Field Fusion (MRF)
        Network:
            x -> lrelu -> upsampling_layer -> resblock1_k1x1 -> z1 -> + -> z_sum / #resblocks -> lrelu -> conv_post_7x1 -> tanh -> o
                                                 ..          -> zI ---|
                                              resblockN_kNx1 -> zN ---'
        Args:
            in_channels (int): number of input tensor channels.
            out_channels (int): number of output tensor channels.
            resblock_type (str): type of the `ResBlock`. '1' or '2'.
            resblock_dilation_sizes (List[List[int]]): list of dilation values in each layer of a `ResBlock`.
            resblock_kernel_sizes (List[int]): list of kernel sizes for each `ResBlock`.
            upsample_kernel_sizes (List[int]): list of kernel sizes for each transposed convolution.
            upsample_initial_channel (int): number of channels for the first upsampling layer. This is divided by 2
                for each consecutive upsampling layer.
            upsample_factors (List[int]): upsampling factors (stride) for each upsampling layer.
            inference_padding (int): constant padding applied to the input at inference time. Defaults to 5.
            pre_linear (int): If not None, add nn.Linear(pre_linear, in_channels) before the convolutions.
        """
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        self.cond_in_each_up_layer = cond_in_each_up_layer

        # initial upsampling layers
        if pre_linear is not None:
            self.lin_pre = nn.Linear(pre_linear, in_channels)
        self.conv_pre = weight_norm(Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        # upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))
        # post convolution layer
        self.conv_post = weight_norm(Conv1d(ch, out_channels, 7, 1, padding=3, bias=conv_post_bias))
        if cond_channels > 0:
            self.cond_layer = nn.Conv1d(cond_channels, upsample_initial_channel, 1)

        if not conv_pre_weight_norm:
            remove_parametrizations(self.conv_pre, "weight")

        if not conv_post_weight_norm:
            remove_parametrizations(self.conv_post, "weight")

        if self.cond_in_each_up_layer:
            self.conds = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel // (2 ** (i + 1))
                self.conds.append(nn.Conv1d(cond_channels, ch, 1))

    def forward1(self, x, g=None):
        # Propagation avant du générateur.
        # x: [B, C, T] features
        # g: [B, cond_channels, T] condition global (optionnel)
        # retourne: waveform [B, 1, T]
        with torch.no_grad():
            if hasattr(self, "lin_pre"):
                x = self.lin_pre(x)
                x = x.permute(0, 2, 1)
            o = self.conv_pre(x)
            if hasattr(self, "cond_layer") and g is not None:
                o = o + self.cond_layer(g)
            for i in range(self.num_upsamples):
                o = F.leaky_relu(o, LRELU_SLOPE)
                o = self.ups[i](o)
                if self.cond_in_each_up_layer and g is not None:
                    o = o + self.conds[i](g)
                z_sum = None
                for j in range(self.num_kernels):
                    if z_sum is None:
                        z_sum = self.resblocks[i * self.num_kernels + j](o)
                    else:
                        z_sum += self.resblocks[i * self.num_kernels + j](o)
                
                o = z_sum / self.num_kernels
                del z_sum
            o = F.leaky_relu(o)
            o = self.conv_post(o)
            o = torch.tanh(o)
        return o

    def forward(self, x, g=None):
        """Propagation avant optimisée pour torch-directml."""
        import torch_directml, gc, time
        device = torch_directml.device()
        if hasattr(self, "lin_pre"):
            x = self.lin_pre(x)
            x = x.permute(0, 2, 1)
        o = self.conv_pre(x)
        if hasattr(self, "cond_layer") and g is not None:
            o = o + self.cond_layer(g)
        for i in range(self.num_upsamples):
            # étape principale
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            if self.cond_in_each_up_layer and g is not None:
                o = o + self.conds[i](g)
            # bloc MRF
            z_sum = None
            for j in range(self.num_kernels):
                res = self.resblocks[i * self.num_kernels + j](o)
                z_sum = res if z_sum is None else z_sum.add_(res)
                del res  # libération immédiate
                gc.collect()

            o = z_sum.div_(self.num_kernels)
            del z_sum
            gc.collect()

            # flush implicite GPU → CPU (force exécution DML)
            _ = o.mean().cpu()
            time.sleep(0.02)
        # sortie finale
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        # flush final
        _ = o.mean().cpu()
        gc.collect()
        time.sleep(0.05)
        return o
    

    @torch.inference_mode()
    def inference(self, c):
        # c est déplacé sur le device du modèle
        c = c.to(self.conv_pre.weight.device)
        c = torch.nn.functional.pad(c, (self.inference_padding, self.inference_padding), "replicate")
        return self.forward(c)
    
    def remove_weight_norm(self):
        print("Pré-normalisation et suppression des weight_norm...")

        for l in self.ups:
            _prenormalize_conv_weight(l)
        for l in self.resblocks:
            if hasattr(l, "remove_weight_norm"):
                l.remove_weight_norm()  # récursif
        _prenormalize_conv_weight(self.conv_pre)
        _prenormalize_conv_weight(self.conv_post)

    """
    def remove_weight_norm(self):
        for l in self.ups:
            try:
                l = pre_normalize_weights(l)
                remove_parametrizations(l, "weight")
            except Exception:
                pass
        for l in self.resblocks:
            try:
                l = pre_normalize_weights(l)
                l.remove_weight_norm()
            except Exception:
                pass"""

class HifiDecoder(nn.Module):
    """Wrapper HiFi-GAN + SpeakerEncoder.

    Fournit une API simple pour décoder des latents en forme d'onde, et
    calcule un embedding de locuteur si nécessaire.
    """

    def __init__(self,                  
        input_sample_rate=22050,
        output_sample_rate=22050,
        output_hop_length=256,
        ar_mel_length_compression=1024,
        decoder_input_dim=1024,
        resblock_type_decoder="1",
        resblock_dilation_sizes_decoder=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes_decoder=[3, 7, 11],
        upsample_rates_decoder=[8, 8, 2, 2],
        upsample_initial_channel_decoder=512,
        upsample_kernel_sizes_decoder=[16, 16, 4, 4],
        d_vector_dim=512,
        cond_d_vector_in_each_upsampling_layer=True,
        speaker_encoder_audio_config={
            "fft_size": 512,
            "win_length": 400,
            "hop_length": 160,
            "sample_rate": 16000,
            "preemphasis": 0.97,
            "num_mels": 64,
        },):
        super().__init__()
        # Paramètres principaux (valeurs par défaut basées sur l'implémentation d'origine)
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.output_hop_length = output_hop_length
        self.ar_mel_length_compression = ar_mel_length_compression
        self.speaker_encoder_audio_config = speaker_encoder_audio_config
        self.waveform_decoder = HifiganGenerator(
            decoder_input_dim,
            1,
            resblock_type_decoder,
            resblock_dilation_sizes_decoder,
            resblock_kernel_sizes_decoder,
            upsample_kernel_sizes_decoder,
            upsample_initial_channel_decoder,
            upsample_rates_decoder,
            inference_padding=0,
            cond_channels=d_vector_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
            cond_in_each_up_layer=cond_d_vector_in_each_upsampling_layer,
        )
        self.speaker_encoder = ResNetSpeakerEncoder(
            input_dim=64,
            proj_dim=512,
            log_input=True,
            use_torch_spec=True,
            audio_config=speaker_encoder_audio_config,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, latents, g=None):
        """Decode latents -> waveform via le générateur."""
        """
        Args:
            x (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.
        Returns:
            Tensor: output waveform.
        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        """z = torch.nn.functional.interpolate(
            latents.transpose(1, 2),
            scale_factor=[self.ar_mel_length_compression / self.output_hop_length],
            mode="area",
        ).squeeze(1) 
        """
        z = torch.nn.functional.interpolate(
            latents.transpose(1, 2).unsqueeze(-1),  # ajoute dimension H=1
            scale_factor=(self.ar_mel_length_compression / self.output_hop_length, 1),
            mode="bilinear",  # supporté par DirectML
            align_corners=False,
        ).squeeze(-1).squeeze(1)        
        # Ajustement du sample rate (seconde interpolation)
        if self.output_sample_rate != self.input_sample_rate:
            z = torch.nn.functional.interpolate(
                z.unsqueeze(-1),                    # [B, C, T', 1]
                scale_factor=(self.output_sample_rate / self.input_sample_rate, 1),
                mode="bilinear",
                align_corners=False,
            ).squeeze(-1)  
        #o = self.waveform_decoder(z, g=g)
        return self.waveform_decoder(z, g=g)
        #return self.waveform_decoder(latents, g=g)

    def forward_No_Dml(self, latents, g=None):
        """
        Args:
            x (Tensor): feature input tensor (GPT latent).
            g (Tensor): global conditioning input tensor.
        Returns:
            Tensor: output waveform.
        Shapes:
            x: [B, C, T]
            Tensor: [B, 1, T]
        """
        z = torch.nn.functional.interpolate(
            latents.transpose(1, 2),
            scale_factor=[self.ar_mel_length_compression / self.output_hop_length],
            mode="linear",
        ).squeeze(1)
        # upsample to the right sr
        if self.output_sample_rate != self.input_sample_rate:
            z = torch.nn.functional.interpolate(
                z,
                scale_factor=[self.output_sample_rate / self.input_sample_rate],
                mode="linear",
            ).squeeze(0)
        o = self.waveform_decoder(z, g=g)
        return o

