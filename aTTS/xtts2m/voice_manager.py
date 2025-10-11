import os
import logging
import torch
import torchaudio
from typing import Any


from xtts2m.utils import maybe_to_fp16
from xtts2m.speaker_encoder import SELayer

logger = logging.getLogger(__name__)

class VoiceManager:
    """Encapsulates speaker-related logic for XTTSv2."""    
    def __init__(self, gpt, hifigan_decoder, config):
        self.gpt = gpt
        self.hifigan_decoder = hifigan_decoder
        self.speaker_manager = None
        self.config = config
        # Cache state
        self.last_speaker_id = None
        self.last_speaker_wav = None
        self.gpt_cond_latents = None
        self.speaker_embeddings = None
        self.mel_stats =  torch.rand(80).to(self.config.device)

    def set_speaker_manager(self,speaker_manager):
        self.speaker_manager = speaker_manager

    def update_voice_cache(self, speaker_id: str | None = None,
                            speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None) -> None:
        """
        Update cached voice data based on the given speaker identifier
        or reference audio(s).
        """
        if (speaker_wav is None) and (speaker_id is not None):
            if self.last_speaker_id != speaker_id:
                self.gpt_cond_latents, self.speaker_embeddings = self.speaker_manager.speakers[speaker_id].values()
                self.last_speaker_id = speaker_id
        else:
            if (speaker_wav is not None): 
                if self.last_speaker_wav != speaker_wav:
                    # Generate or load a new voice from the reference audio(s).    
                    voice_settings = {
                                        "gpt_cond_len": self.config.gpt_cond_len,
                                        "gpt_cond_chunk_len": self.config.gpt_cond_chunk_len,
                                        "max_ref_length": self.config.max_ref_len,
                                        "sound_norm_refs": self.config.sound_norm_refs, }            
                    self.gpt_cond_latents, self.speaker_embeddings = self.get_conditioning_latents(speaker_wav,  **voice_settings )
                    self.last_speaker_wav = speaker_wav

    # -----------------------------------------------------------------
    #  Voice cloning
    # -----------------------------------------------------------------
     
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
        """Get the conditioning latents for the GPT model from the given audio.
        Args:
            audio_path (str or List[str]): Path to reference audio file(s).
            max_ref_length (int): Maximum length of each reference audio in seconds. Defaults to 30.
            gpt_cond_len (int): Length of the audio used for gpt latents. Defaults to 6.
            gpt_cond_chunk_len (int): Chunk length used for gpt latents. It must be <= gpt_conf_len. Defaults to 6.
            librosa_trim_db (int, optional): Trim the audio using this value. If None, not trimming. Defaults to None.
            sound_norm_refs (bool, optional): Whether to normalize the audio. Defaults to False.
            load_sr (int, optional): Sample rate to load the audio. Defaults to 22050.
        """
        logger.info("Generated voice from reference audio")
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
            audio = audio[:, : load_sr * max_ref_length].to(self.config.device)#
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            #if librosa_trim_db is not None:
            #    audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            #speaker_embedding = maybe_to_fp16(speaker_embedding)
            speaker_embeddings.append(speaker_embedding)

            audios.append(audio)

        # merge all the audios and compute the latents for the gpt
        full_audio = torch.cat(audios, dim=-1).to(self.config.device)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )  # [1, 1024, T]

        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)
            #speaker_embedding = maybe_to_fp16(speaker_embedding)

        return gpt_cond_latents, speaker_embedding

 
    def _update_cache(self, speaker_id, speaker_wav, voice):
        """Update internal caches with new speaker data."""
        self.last_speaker_id = speaker_id
        self.last_speaker_wav = speaker_wav
        self.gpt_cond_latents = voice["gpt_conditioning_latents"]
        self.speaker_embeddings = voice["speaker_embedding"]

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

        if self.config.model_args.gpt_use_perceiver_resampler:
            style_embs = []
            for i in range(0, audio.shape[1], 22050 * chunk_length):
                audio_chunk = audio[:, i : i + 22050 * chunk_length]
                # if the chunk is too short ignore it
                if audio_chunk.size(-1) < 22050 * MIN_AUDIO_SECONDS:
                    continue

                mel_chunk = wav_to_mel_cloning(
                    audio_chunk,
                    mel_norms=self.mel_stats.to(self.config.device),#
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                    device= self.config.device
                )
                mel_chunk = maybe_to_fp16(mel_chunk)
                style_emb = self.gpt.get_style_emb(mel_chunk.to(self.config.device), None)#self.config.device
                #style_emb = maybe_to_fp16(style_emb)
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
                device=self.config.device
            )
            #mel = maybe_to_fp16(mel)
            cond_latent = self.gpt.get_style_emb(mel.to(self.config.device))#
            #cond_latent = maybe_to_fp16(cond_latent)
        return cond_latent.transpose(1, 2)



    # @  torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        """Retourne l'embedding de speaker à partir d'un waveform (résamplé à 16k)."""
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        audio_16k = maybe_to_fp16(audio_16k)
        return (            
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.config.device), l2_norm=True) #
            .unsqueeze(-1)
            #.to( self.config.device)
        )
       


def wav_to_mel_cloning( wav, mel_norms=None, device=torch.device("cpu"),
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

    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    try:
        audio, lsr = torchaudio.load(audiopath)
        if audio.size(0) != 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
        amax = float(audio.max())
        amin = float(audio.min())
        if amax > 1.1 or amin < -1.1:
            logger.warning("Audio %s hors plage attendue: max=%.3f min=%.3f — clipping appliqué.", audiopath, amax, amin)
        audio = torch.clamp(audio, -1.0, 1.0)
        return audio
    except Exception as e:
        logger.error("Erreur lors du chargement de %s: %s", audiopath, e)
        return None

