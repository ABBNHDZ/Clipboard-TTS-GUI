"""
Module `model.py` formaté et commenté en français.

Cette version est une transcription commentée du fichier original situé
dans `xtts2_m/model.py`. Elle préserve l'API et la logique d'origine mais
ajoute des docstrings et commentaires en français pour faciliter la lecture.

Remarques:
- Les imports locaux pointent vers le package `xtts2_m2` (imports relatifs)
  pour permettre d'utiliser la copie commentée du reste des modules.
- Ce module expose la classe `XTTS` (en tant que nn.Module) et des utilitaires
  pour charger/convertir l'audio et gérer les voices/metadata.
"""

import os
import datetime
from pathlib import Path
import logging
import torch
from safetensors.torch import load_file
import torchaudio
from typing import Any, Callable, Optional
import librosa
from torch.nn import functional as F
from torch import nn
import numpy as np
import time

from .tokenizer import VoiceBpeTokenizer
from .gpt import GPT
from .hifigan import HifiDecoder
import unicodedata
import re
from .xttsConfig import XttsConfig
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)


def is_pytorch_at_least_2_4() -> bool:
    try:
        from packaging.version import Version

        return Version(torch.__version__) >= Version("2.4")
    except Exception:
        return False


@dataclass
class VoiceMetadata:
    """Métadonnées associées à une voix clonée/sauvegardée.

    Champs principaux : `model` (info sur le modèle), `speaker_id`,
    `source_files` (fichiers audio sources), `created_at` (timestamp ISO),
    `coqui_version` (tag de version).
    """
    model: dict[str, str | float | bool]
    speaker_id: str
    source_files: list[str] | None = None
    created_at: str | None = None
    coqui_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceMetadata":
        return cls(**data)


class SpeakerManager:
    """Gestionnaire simple pour « speakers » pré-enregistrés.

    Le fichier fourni doit contenir un dict mapping speaker_id -> voice_dict
    où `voice_dict` contient au moins `gpt_conditioning_latents` et
    `speaker_embedding`.
    """

    def __init__(self, speaker_file_path: str | None = None):
        self.speakers: dict[str, Any] = {}
        if speaker_file_path and os.path.exists(speaker_file_path):
            try:
                # Utiliser weights_only si disponible pour éviter de charger
                # des objets inutiles (nouveau paramètre PyTorch 2.4+)
                self.speakers = (
                    torch.load(speaker_file_path, weights_only=True) or {}
                )
            except Exception as e:
                logger.warning("Impossible de charger %s: %s", speaker_file_path, e)
                self.speakers = {}
        else:
            logger.info("Fichier speakers_xtts.pth non trouvé. Aucun speaker pré-enregistré chargé.")

    @property
    def name_to_id(self) -> dict:
        """Retourne le mapping nom->données du speaker."""
        return self.speakers

    @property
    def num_speakers(self) -> int:
        """Nombre de speakers connus."""
        return len(self.speakers)

    @property
    def speaker_names(self) -> list:
        """Liste des noms de speakers disponibles."""
        return list(self.speakers.keys())


class LanguageManager:
    """Wrapper minimal pour la structure `languages` du config.

    Conserve une référence simple et expose des helpers basiques.
    """

    def __init__(self, config: Any):
        # attendre que config.languages soit une structure indexable (dict ou list)
        self.langs = config.languages

    @property
    def name_to_id(self) -> Any:
        return self.langs

    @property
    def num_languages(self) -> int:
        return len(self.langs)

    @property
    def language_names(self) -> list:
        # si langs est un dict, retourner ses clés; sinon construire la liste
        if isinstance(self.langs, dict):
            return list(self.langs.keys())
        return list(self.langs)


def wav_to_mel_cloning(
    wav,
    mel_norms_file="../experiments/clips_mel_norms.pth",
    mel_norms=None,
    device=torch.device("cpu"),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    power=2,
    normalized=False,
    sample_rate=22050,
    f_min=0,
    f_max=8000,
    n_mels=80,
):
    """Convertit un waveform en spectrogramme mel pour clonage.

    Si `mel_norms` est absent, tente de charger `mel_norms_file`. En cas
    d'échec, une normalisation unitaire est utilisée.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    # Chargement sécurisé des statistiques de normalisation
    if mel_norms is None:
        try:
            if os.path.exists(mel_norms_file):
                mel_norms = torch.load(
                    mel_norms_file,
                    map_location=device,
                    weights_only=True,
                )
            else:
                raise FileNotFoundError(f"mel_norms_file not found: {mel_norms_file}")
        except Exception as e:
            logger.warning(
                "Impossible de charger mel_norms (%s): %s — utilisation d'une normalisation unitaire.",
                mel_norms_file,
                e,
            )
            mel_norms = torch.ones(n_mels, device=device)

    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    """Charge un fichier audio, force mono et resample si nécessaire.

    Retourne Tensor (1, N) ou None en cas d'erreur.
    """
    try:
        audio, lsr = torchaudio.load(audiopath)
        # stereo -> mono si nécessaire (moyenne des canaux)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        # resample si besoin
        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
        # Vérifier plage d'amplitude attendue [-1, 1]
        amax = float(audio.max())
        amin = float(audio.min())
        if amax > 1.1 or amin < -1.1:
            logger.warning(
                "Audio %s hors plage attendue: max=%.3f min=%.3f — clipping appliqué.",
                audiopath,
                amax,
                amin,
            )
        # S'assurer que l'audio reste dans [-1, 1]
        audio = torch.clamp(audio, -1.0, 1.0)
        return audio
    except Exception as e:
        logger.error("Erreur lors du chargement de %s: %s", audiopath, e)
        return None



class XTTS(nn.Module):
    """Classe principale pour XTTSv2 (chargement, synthèse, utilitaires).

    Comporte :
    - initialisation des sous-modèles (GPT, HiFi-GAN)
    - méthodes de chargement de checkpoints
    - interfaces haut-niveau `synthesize`, `inference` et `clone_voice`.
    """
    MODEL_TYPE = "xttsv2"

    def __init__(self, config: XttsConfig):
        super().__init__()
        self.config = config
        # référence optionnelle vers l'application GUI si utilisée
        #from clipboard_tts_gui import App

        self.app= None #: Optional[App] 
        self.tokenizer = None
        self.speaker_manager = None
        self.language_manager = None
        self.config.num_chars = self.config.model_args.num_chars
        self.args = self.config.model_args
        self.mel_stats_path = None
        self.gpt_checkpoint = self.args.gpt_checkpoint
        self.decoder_checkpoint = self.args.decoder_checkpoint
        self.models_path = config.model_path
        self.gpt_batch_size = self.args.gpt_batch_size
        self.gpt_cond_latents_cache = None
        self.speaker_embeddings_cache = None
        self.last_speaker_id = None
        self.last_speaker_wav = None
        self.init_models()
        # buffer de stats mel initialisé à 1 pour éviter div/0 avant chargement
        self.register_buffer("mel_stats", torch.ones(80))

    def init_models(self):
        """Initialise GPT et HiFiGAN decoder en fonction des args.

        La méthode sépare l'initialisation pour permettre de charger le tokenizer
        avant de charger les poids si nécessaire.
        """
        self.gpt = (
            GPT(
                layers=self.args.gpt_layers,
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                max_prompt_tokens=self.args.gpt_max_prompt_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
                use_perceiver_resampler=self.args.gpt_use_perceiver_resampler,
                code_stride_len=self.args.gpt_code_stride_len,
            )
            .to(self.config.device)
        )

        self.hifigan_decoder = (
            HifiDecoder(
                input_sample_rate=self.args.input_sample_rate,
                output_sample_rate=self.args.output_sample_rate,
                output_hop_length=self.args.output_hop_length,
                ar_mel_length_compression=self.args.gpt_code_stride_len,
                decoder_input_dim=self.args.decoder_input_dim,
                d_vector_dim=self.args.d_vector_dim,
                cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
            )
            .to(self.config.device)
        )

    @staticmethod
    def init_from_config(config: XttsConfig):
        # Retourne une instance initialisée à partir d'un objet config
        return XTTS(config)

    # **Libération de la mémoire**
    @staticmethod
    def empty_cache(checkpoint):
        if checkpoint:
            del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()


    TARGET_DTYPE = torch.float32  # stockage final pour inference (fp16 possible)

    def load_checkpoint(
        self,
        config: XttsConfig,
        model_path: str | None = None,
        eval: bool = True,
        strict: bool = True,
        use_deepspeed: bool = False,
        speaker_file_path: str | None = "",
        vocab_path: str | None = None,
        progress_cb: Optional[Callable] = None,
        done_cb: Optional[Callable] = None,
    ):
        """Charge le vocabulaire (tokenizer) puis le checkpoint de poids du modèle.

        - Tente `weights_only` lorsque supporté, puis fallback.
        - Essaie `load_state_dict(strict=True)` puis `strict=False` si nécessaire
          et logge les différences.
        """
        if config.model_path is None and model_path is not None:
            config.model_path = model_path
        if model_path is None:
            model_path = config.model_path

        if config.model_args.tokenizer_file is not None and (vocab_path is None or vocab_path == ""):
            vocab_path = config.model_args.tokenizer_file
        if vocab_path is None or vocab_path == "":
            vocab_path = config.model_args.tokenizer_file1
        if not os.path.exists(vocab_path):
            if os.path.exists(os.path.join(model_path, vocab_path)):
                vocab_path = os.path.join(model_path, vocab_path)

        if config.model_args.speaker_file1 is not None and (speaker_file_path is None or speaker_file_path == ""):
            speaker_file_path = config.model_args.speaker_file1
        if not os.path.exists(speaker_file_path):
            if os.path.exists(os.path.join(model_path, speaker_file_path)):
                speaker_file_path = os.path.join(model_path, speaker_file_path)

        if os.path.exists(vocab_path):
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
            if progress_cb:
                progress_cb(15, "Tokenizer chargé")
        else:
            msg = (
                f"`vocab.json` file not found in `{vocab_path}`. Move the file there or "
                "specify alternative path in `model_args.tokenizer_file` in `config.json`"
            )
            raise FileNotFoundError(msg)

        self.language_manager = LanguageManager(config)
        if progress_cb:
            progress_cb(17, "Language manager prêt")
        self.speaker_manager = SpeakerManager(speaker_file_path=speaker_file_path)

        var_i = 15
        var_i = var_i + 5 if var_i < 100 else var_i - 10
        time.sleep(0.1)

        # phase 3: load checkpoints
        if progress_cb:
            progress_cb(var_i, "Chargement hifigan_decoder")
            time.sleep(0.1)

        fnext = "fp16"
        if self.config.Qntf == "fp32":
            fnext = ".safetensors"
        elif self.config.Qntf == "fp16":
            fnext = "_fp16.safetensors"
        elif self.config.Qntf == "bf16":
            fnext = "_bf16.safetensors"
        elif self.config.Qntf == "f8e4":
            fnext = "_f8e4.safetensors"
        elif self.config.Qntf == "f8e5":
            fnext = "_f8e5.safetensors"

        model_path = os.path.join(self.config.model_path, "ckpt_hfd" + fnext)
        logger.info("Chargement du checkpoint hifigan_decoder : %s", model_path)
        checkpoint = load_file(model_path)
        self.hifigan_decoder.load_state_dict(checkpoint, strict=True)

        self.empty_cache(checkpoint)

        var_i = var_i + 5 if var_i < 100 else var_i - 10
        if progress_cb:
            progress_cb(var_i, "Chargement du checkpoint_gpt")
            time.sleep(0.1)

        model_path = os.path.join(self.config.model_path, "ckpt_gpt1" + fnext)
        logger.info("Chargement du checkpoint_gpt : %s", model_path)
        checkpoint = load_file(model_path)
        self.gpt.load_state_dict(checkpoint, strict=False)

        self.empty_cache(checkpoint)

        var_i = var_i + 5 if var_i < 100 else var_i - 10
        if progress_cb:
            progress_cb(var_i, "Chargement du gpt_ln")

        # Charger les parties "hors gpt.h"
        other_path = os.path.join(self.config.model_path, "gpt_ln" + fnext)
        state = load_file(other_path)
        self.gpt.load_state_dict(state, strict=False)
        print(f"Chargé {other_path}")

        if self.config.Qntf == "fp32":
            fnext = "fp32.safetensors"
        DTYPE = torch.float32  # modèle final reconverti en float32 pour l'inférence
        # Charger shard par shard pour gpt.h
        for fname in sorted(os.listdir(self.config.model_path + "/gpt_h")):
            if fname.startswith("gpt_h_") and fname.endswith(fnext):
                fpath = os.path.join(self.config.model_path + "/gpt_h", fname)
                sub_state = load_file(fpath)
                if self.config.Qntf != "fp32":
                    sub_state = {k: v.to(DTYPE) for k, v in sub_state.items()}
                self.gpt.load_state_dict(sub_state, strict=False)
                print(f"Chargé {fname}")
                self.empty_cache(sub_state)
                var_i = var_i + 5 if var_i < 100 else var_i - 10
                if progress_cb:
                    progress_cb(var_i, "Chargement du gpt_h")
                    time.sleep(0.1)

        model_path = os.path.join(self.config.model_path, "ckpt_mel.safetensors")
        logger.info("Chargement du checkpoint ckpt_mel : %s", model_path)
        checkpoint = load_file(model_path)
        self.mel_stats.copy_(checkpoint["mel_stats"])
        var_i = var_i + 5 if var_i < 100 else var_i - 10
        if progress_cb:
            progress_cb(var_i, "Chargement du mel_stats")

        self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=use_deepspeed)
        # Mode évaluation pour sous-modèles si demandé
        if eval:
            self.hifigan_decoder.eval()
            self.gpt.eval()

        self.empty_cache(checkpoint)


    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Calcule les latents de conditionnement GPT à partir d'un audio de référence.

        Le calcul supporte deux modes : utilisation du perceiver resampler (morceaux)
        ou traitement global du mel.
        """
        MIN_AUDIO_SECONDS = 0.33
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]

        if self.args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]
                # if the chunk is too short ignore it
                if audio_chunk.size(-1) < 22050 * MIN_AUDIO_SECONDS:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.device), None)
                style_embs.append(style_emb)

            # mean style embedding
            if len(style_embs) == 0:
                msg = f"Provided reference audio too short (minimum length: {MIN_AUDIO_SECONDS:.2f} seconds)."
                raise RuntimeError(msg)
            cond_latent = torch.stack(style_embs).mean(dim=0)
        else:
            mel = wav_to_mel_cloning(
                audio,
                mel_norms=self.mel_stats.cpu(),
                n_fft=4096,
                hop_length=1024,
                win_length=4096,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            cond_latent = self.gpt.get_style_emb(mel.to(self.device))
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        """Retourne l'embedding de speaker à partir d'un waveform (résamplé à 16k)."""
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )

    def _clone_voice(
        self,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        **generate_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        gpt_conditioning_latents, speaker_embedding = self.get_conditioning_latents(
            audio_path=speaker_wav,
            **generate_kwargs,
        )
        voice = {
            "gpt_conditioning_latents": gpt_conditioning_latents,
            "speaker_embedding": speaker_embedding,
        }
        metadata = {"name": "XTTSv2"}
        return voice, metadata

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path: str | os.PathLike[Any] | list[str | os.PathLike[Any]],
        max_ref_length: int = 30,
        gpt_cond_len: int = 6,
        gpt_cond_chunk_len: int = 6,
        librosa_trim_db: int | None = None,
        sound_norm_refs: bool = False,
        load_sr: int = 22050,
    ):
        """Récupère les latents de conditionnement depuis un ou plusieurs fichiers audio de référence."""
        # deal with multiples references
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            if audio is None:
                raise RuntimeError(f"Impossible de charger le fichier audio de référence: {file_path}")
            audio = audio[:, : load_sr * max_ref_length].to(self.config.device)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)
            audios.append(audio)

        # merge all the audios and compute the latents pour le gpt
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )

        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def synthesize(
        self,
        text: str,
        config: Any | None = None,
        *,
        speaker: str | None = None,
        speaker_wav: (
            str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None
        ) = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        language: str | None = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Synthétise le texte fourni en utilisant la voix ou le speaker donné.

        Retourne le waveform numpy (ou None en cas d'erreur).
        """
        if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
            speaker = speaker_id
        speaker_id = speaker
        for key in ("use_griffin_lim", "do_trim_silence", "extra_aux_input"):
            kwargs.pop(key, None)
        assert ("zh-cn" if language == "zh" else language in self.config.languages), (
            f" ❗ Language {language} is not supported, use : {self.config.languages}"
        )
        voice_settings = {
            "gpt_cond_len": self.config.gpt_cond_len,
            "gpt_cond_chunk_len": self.config.gpt_cond_chunk_len,
            "max_ref_length": self.config.max_ref_len,
            "sound_norm_refs": self.config.sound_norm_refs,
        }
        inference_settings = {
            "temperature": self.config.temperature,
            "length_penalty": self.config.length_penalty,
            "repetition_penalty": self.config.repetition_penalty,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
        }
        inference_settings.update(kwargs)

        gpt_cond_latent = self.gpt_cond_latents_cache
        speaker_embedding = self.speaker_embeddings_cache

        if (self.last_speaker_id != speaker_id) or (self.last_speaker_wav != speaker_wav):
            self.last_speaker_id = speaker_id
            self.last_speaker_wav = speaker_wav
            if (
                speaker_wav is None
                and speaker is not None
                and speaker in self.speaker_manager.speakers
            ):
                gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker].values()
            else:
                voice = self.clone_voice(speaker_wav, speaker, voice_dir, **voice_settings)
                gpt_cond_latent = voice["gpt_conditioning_latents"]
                speaker_embedding = voice["speaker_embedding"]

        self.gpt_cond_latents_cache = gpt_cond_latent
        self.speaker_embeddings_cache = speaker_embedding
        return self.inference(
            text, language, self.gpt_cond_latents_cache, self.speaker_embeddings_cache, **inference_settings
        )

    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # GPT inference
        temperature: float = 0.75,
        length_penalty: float = 1.0,
        repetition_penalty: float = 10.0,
        top_k: int = 50,
        top_p: float = 0.85,
        do_sample: bool = True,
        num_beams: int = 1,
        speed: float = 1.0,
        enable_text_splitting: bool = False,
        **hf_generate_kwargs: Any,
    ):
        """Procède à la génération (GPT -> HiFiGAN) et retourne un ndarray audio.

        Arguments et comportements préservent ceux de l'implémentation d'origine.
        """
        language = language.split("-")[0]
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to(self.config.device)
        speaker_embedding = speaker_embedding.to(self.config.device)
        wavs = []
        sent = text.strip().lower()
        text_tokens = (
            torch.IntTensor(self.tokenizer.encode(sent, lang=language))
            .unsqueeze(0)
            .to(self.config.device)
        )
        assert (
            text_tokens.shape[-1] < self.args.gpt_max_text_tokens
        ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

        with torch.no_grad():
            gpt_codes = self.gpt.generate(
                cond_latents=gpt_cond_latent,
                text_inputs=text_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=self.gpt_batch_size,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                output_attentions=False,
                **hf_generate_kwargs,
            )
            expected_output_len = torch.tensor(
                [gpt_codes.shape[-1] * self.gpt.code_stride_len],
                device=text_tokens.device,
            )

            text_len = torch.tensor([text_tokens.shape[-1]], device=self.config.device)
            gpt_latents = self.gpt(
                text_tokens,
                text_len,
                gpt_codes,
                expected_output_len,
                cond_latents=gpt_cond_latent,
                return_attentions=False,
                return_latent=True,
            )

            if length_scale != 1.0:
                gpt_latents = F.interpolate(
                    gpt_latents.transpose(1, 2),
                    scale_factor=length_scale,
                    mode="linear",
                ).transpose(1, 2)

            wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())
        return torch.cat(wavs, dim=0).numpy()

    def eval(self):
        """Passe tous les sous-modèles en mode évaluation et retire weight_norm du décodeur."""
        super().eval()
        self.gpt.eval()
        self.hifigan_decoder.eval()
        self.hifigan_decoder.remove_weight_norm()
        return self

    def clone_voice(
        self,
        speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None,
        speaker_id: str | None = None,
        voice_dir: str | os.PathLike[Any] | None = None,
        **generate_kwargs: Any,
    ) -> dict[str, Any]:
        """Charge une voice pré-générée ou génère une voice depuis audio(s) de référence.

        Si `speaker_wav` n'est pas spécifié, la méthode tente de charger
        `<voice_dir>/<speaker_id>.pth`. Si `speaker_id` et `voice_dir` sont fournis
        la nouvelle voice sera enregistrée dans ce dossier.
        """
        if speaker_wav is None or (
            isinstance(speaker_wav, list) and len(speaker_wav) == 0
        ):
            if speaker_id is None:
                raise RuntimeError("Neither `speaker_wav` nor `speaker_id` was specified")
            if voice_dir is None:
                raise RuntimeError(
                    "Specified only `speaker_id`, but no `voice_dir` to load the voice from"
                )
            return self.load_voice_file(speaker_id, voice_dir)

        voice, model_metadata = self._clone_voice(speaker_wav, **generate_kwargs)
        logger.info("Generated voice from reference audio")
        if speaker_id is not None and voice_dir is not None:
            speaker_id = slugify(speaker_id)
            voice_fn = Path(voice_dir) / f"{speaker_id}.pth"
            voice_fn.parent.mkdir(exist_ok=True, parents=True)
            speaker_wav = speaker_wav if isinstance(speaker_wav, list) else [speaker_wav]
            metadata = self._create_voice_metadata(model_metadata, speaker_id, [str(p) for p in speaker_wav])
            voices = self.get_voices(voice_dir)
            if speaker_id in voices:
                logger.info(
                    "Voice `%s` already exists in `%s`, overwriting it",
                    speaker_id,
                    voice_fn,
                )
            voice_dict = {**voice, "metadata": metadata.to_dict()}
            # Sauvegarde sûre: utiliser torch.save (format pth) pour stocker dictionnaire voice
            torch.save(voice_dict, voice_fn)
            logger.info("Voice `%s` saved to: %s", speaker_id, voice_fn)
        return voice

    def _create_voice_metadata(
        self,
        model: dict[str, str | float | bool],
        speaker_id: str,
        source_files: list[str],
    ) -> VoiceMetadata:
        return VoiceMetadata(
            model=model,
            speaker_id=speaker_id,
            source_files=source_files,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
            coqui_version="coqui-tts",
        )

    def load_voice_file(
        self,
        speaker_id: str,
        voice_dir: str | os.PathLike[Any],
    ) -> dict[str, Any]:
        """Charge le fichier de voix sauvegardé (`<voice_dir>/<slugified_speaker>.pth`)."""
        voices = self.get_voices(voice_dir)
        if speaker_id not in voices:
            msg = f"Voice file `{slugify(speaker_id)}.pth` for speaker `{speaker_id}` not found in: {voice_dir}"
            raise FileNotFoundError(msg)
        # Chargement du voice dict en mémoire (map to CPU pour éviter consommation GPU inutile)
        try:
            voice = torch.load(
                voices[speaker_id],
                map_location="cpu",
                weights_only=is_pytorch_at_least_2_4(),
            )
        except TypeError:
            voice = torch.load(voices[speaker_id], map_location="cpu")
        logger.info("Loaded voice `%s` from: %s", speaker_id, voices[speaker_id])
        return voice

    def get_voices(self, voice_dir: str | os.PathLike[Any]) -> dict[str, Path]:
        """Retourne un dict mapping speaker_id -> Path(fichier .pth)."""
        return {path.stem: path for path in Path(voice_dir).glob("*.pth")}


def slugify(text: str) -> str:
    """Normalise une chaîne pour en faire un nom de fichier sûr.

    Exemple: Zoë -> Zoe, remplace caractères non-alphanumériques par '_'.
    """
    # Normalize to ASCII (e.g., Zoë -> Zoe)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace unsafe characters with underscores
    safe = re.sub(r"[^\w\-]", "_", ascii_str)
    # Collapse repeated underscores
    return re.sub(r"_+", "_", safe).strip("_")
