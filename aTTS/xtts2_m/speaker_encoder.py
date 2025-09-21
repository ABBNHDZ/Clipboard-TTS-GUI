"""
Encodeur de locuteur (ResNet) formaté et commenté en français.

Ce module définit :
- SELayer : couche Squeeze-and-Excitation
- SEBasicBlock : bloc résiduel avec SE
- PreEmphasis : filtre de pré-emphase
- ResNetSpeakerEncoder : encodeur de locuteur complet

Remarques :
- `get_torch_mel_spectrogram_class` renvoie une séquence d'opérations pour
  calculer un spectrogramme Mel si `use_torch_spec` est activé.
- La méthode `compute_embedding` découpe l'utterance en fenêtres et calcule
  la moyenne des embeddings (utile lors de l'inférence sur segments).
"""

import numpy as np
import torch
from torch import nn
import torchaudio


class SELayer(nn.Module):
    """Couche Squeeze-and-Excitation.

    Permet au réseau de recalibrer l'importance de chaque canal en
    effectuant un squeeze (global pooling) suivi d'un gate (MLP + sigmoid).
    """

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    """Bloc résiduel basique incorporant une couche SE.

    Utilisé pour construire l'architecture ResNet de l'encodeur.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class PreEmphasis(nn.Module):
    """Filtre de pré-emphase implémenté en convolution 1D.

    Ce filtre est appliqué aux formes d'onde avant calcul du spectrogramme
    pour renforcer les hautes fréquences.
    """

    def __init__(self, coefficient=0.97):
        super().__init__()
        self.coefficient = coefficient
        # filtre simple [-coef, 1]
        self.register_buffer("filter", torch.FloatTensor([-self.coefficient, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # x attendu de forme (T,) ou (B, T) selon le contexte ; ici on assume (B, T)
        assert len(x.size()) == 2
        x = torch.nn.functional.pad(x.unsqueeze(1), (1, 0), "reflect")
        return torch.nn.functional.conv1d(x, self.filter).squeeze(1)


class ResNetSpeakerEncoder(nn.Module):
    """Encodeur de locuteur basé sur ResNet + attention (ASP/SAP).

    - input_dim : dimension parallèle du spectrogramme (nombre de bandes mel)
    - proj_dim  : dimension de l'embedding de sortie
    - layers, num_filters : paramètres de l'architecture ResNet
    - encoder_type : 'ASP' (attentive statistics pooling) ou 'SAP' (simple attentive pooling)
    - use_torch_spec : si True, calcule le MelSpectrogram à la volée via torchaudio
    - audio_config : dictionnaire décrivant les paramètres du mel (sample_rate, fft_size, ...)
    """

    def __init__(
        self,
        input_dim=64,
        proj_dim=512,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input=False,
        use_torch_spec=False,
        audio_config=None,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.input_dim = input_dim
        self.log_input = log_input
        self.use_torch_spec = use_torch_spec
        self.audio_config = audio_config or {}
        self.proj_dim = proj_dim

        # Première convolution d'entrée
        self.conv1 = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        # Construction des couches ResNet
        self.inplanes = num_filters[0]
        self.layer1 = self.create_layer(SEBasicBlock, num_filters[0], layers[0])
        self.layer2 = self.create_layer(SEBasicBlock, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self.create_layer(SEBasicBlock, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self.create_layer(SEBasicBlock, num_filters[3], layers[3], stride=(2, 2))

        # Instance norm sur la dimension des bandes mel
        self.instancenorm = nn.InstanceNorm1d(input_dim)

        # Option : calculer le spectrogramme Mel à la volée
        if self.use_torch_spec:
            self.torch_spec = self.get_torch_mel_spectrogram_class(self.audio_config)
        else:
            self.torch_spec = None

        # Taille de la map de sortie après 3 downsamples (approx: input_dim / 8)
        outmap_size = int(self.input_dim / 8)

        # Mécanisme d'attention pour le pooling
        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        # Dimension de sortie variable selon le type de pooling
        if self.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("encoder_type doit être 'SAP' ou 'ASP'")

        self.fc = nn.Linear(out_dim, proj_dim)
        self._init_layers()

    def get_preemphasis(self, coef):
        """Retourne une transformation de pré-emphase fournie par torchaudio."""
        return torchaudio.transforms.Preemphasis(coef)

    def _init_layers(self):
        """Initialisation des poids (He pour convs, 1/0 pour BatchNorm)."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def create_layer(self, block, planes, blocks, stride=1):
        """Construit une ``nn.Sequential`` de `blocks` blocs résiduels."""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, l2_norm=False):
        """Passage avant.

        Arguments :
        - x : Tensor, soit une forme d'onde (B, 1, T) soit un spectrogramme (B, D, T)
        - l2_norm : bool, normaliser ou non l'embedding final

        Retour : embedding de dimension (B, proj_dim) ou (B, proj_dim) normalisé
        si l2_norm=True.
        """
        # Suppression de la dimension canal si présente
        x.squeeze_(1)

        # Si activé, calculer le spectrogramme Mel à partir de la forme d'onde
        if self.use_torch_spec and self.torch_spec is not None:
            x = self.torch_spec(x)

        if self.log_input:
            x = (x + 1e-6).log()

        # InstanceNorm attend (B, D, T)
        x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Reshape pour le pooling attentionnel : (B, C * outmap, T')
        x = x.reshape(x.size()[0], -1, x.size()[-1])

        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def get_torch_mel_spectrogram_class(self, audio_config):
        """Construit un transform qui renvoie un spectrogramme Mel via torchaudio.

        `audio_config` doit être un dict contenant : sample_rate, fft_size,
        win_length, hop_length, num_mels, preemphasis.
        """
        return torch.nn.Sequential(
            PreEmphasis(audio_config["preemphasis"]),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=audio_config["sample_rate"],
                n_fft=audio_config["fft_size"],
                win_length=audio_config["win_length"],
                hop_length=audio_config["hop_length"],
                window_fn=torch.hamming_window,
                n_mels=audio_config["num_mels"],
            ),
        )

    @torch.inference_mode()
    def inference(self, x, l2_norm=True):
        """Alias verbeux pour `forward` en mode inference (pas de grads)."""
        return self.forward(x, l2_norm)

    @torch.inference_mode()
    def compute_embedding(self, x, num_frames=250, num_eval=10, return_mean=True, l2_norm=True):
        """Calcule l'embedding pour une ou plusieurs utterances en découpant en
        fenêtres d'évaluation.

        - x : Tensor shape (1, T, D) ou (B, T) selon l'usage (on suppose 1D ou 2D)
        - num_frames : nombre d'échantillons (si use_torch_spec True, il sera
          multiplié par hop_length)
        - num_eval : nombre de fenêtres à évaluer (moyennées ensuite)
        - return_mean : si True, moyenne des embeddings renvoyés
        """
        # Si on utilise le spectrogramme calculé à la volée, num_frames est en
        # nombre de frames mel ; on convertit donc en nombre d'échantillons
        if self.use_torch_spec:
            # audio_config devrait contenir 'hop_length'
            num_frames = num_frames * self.audio_config["hop_length"]

        max_len = x.shape[1]

        if max_len < num_frames:
            num_frames = max_len

        # Générer offsets uniformly espacés
        offsets = np.linspace(0, max_len - num_frames, num=num_eval)

        frames_batch = []
        for offset in offsets:
            offset = int(offset)
            end_offset = int(offset + num_frames)
            frames = x[:, offset:end_offset]
            frames_batch.append(frames)

        frames_batch = torch.cat(frames_batch, dim=0)
        embeddings = self.inference(frames_batch, l2_norm=l2_norm)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)
        return embeddings
