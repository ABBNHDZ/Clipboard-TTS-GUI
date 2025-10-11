"""
  Ce module expose la classe `XTTS` (en tant que nn.Module) et des utilitaires
  pour charger/convertir l'audio et gérer les voices.
"""

import os
import torch_directml
import ctypes
from pathlib import Path
import logging
import torch
from safetensors.torch import load_file , save_file
from typing import Any, Callable, Optional
from torch.nn import functional as F
from torch import nn
import gc
import numpy as np
import time
import re


from xtts2m.tokenizer import VoiceBpeTokenizer
from xtts2m.gpt import GPT, setup_seed
from xtts2m.hifigan import HifiDecoder
from xtts2m.xttsConfig import XttsConfig
from xtts2m.utils import logger_ram_used, get_valid_dtype_for_device ,maybe_to_fp16

# accelerate imports (optional)
init_empty_weights = None
load_checkpoint_and_dispatch = None
HAVE_ACCELERATE = False

try:
    from accelerate import init_empty_weights

    HAVE_ACCELERATE = True
except Exception:
    HAVE_ACCELERATE = False

logger = logging.getLogger(__name__)

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
                self.speakers = (torch.load(speaker_file_path, weights_only=True) or {} )                
            except Exception as e:
                logger.warning("Impossible de charger %s: %s", speaker_file_path, e)
                self.speakers = {}
        else:
            logger.info("Fichier speakers_xtts.pth non trouvé. Aucun speaker pré-enregistré chargé.")

    @property
    def speaker_names(self) -> list:
        """Liste des noms de speakers disponibles."""
        if isinstance(self.speakers, dict):
            return list(self.speakers.keys())
        return list(self.speakers)

class LanguageManager:
    """Wrapper minimal pour la structure `languages` du config.
    Conserve une référence simple et expose des helpers basiques.
    """
    def __init__(self, config: Any):
        self.langs = config.languages

    @property
    def language_names(self) -> list:
        # si langs est un dict, retourner ses clés; sinon construire la liste
        if isinstance(self.langs, dict):
            return list(self.langs.keys())
        return list(self.langs)

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
        self.args = config.model_args
        self.TARGET_DTYPE = get_valid_dtype_for_device(self.config.device,self.config.sdtype)
        self.gpt:GPT 
        self.hifigan_decoder : HifiDecoder
        self.checkpoint = None
        self.mel_stats = torch.rand(80).to(self.config.device)  # Random stats for non-instantiated models        
        self._init_components()        
        self._init_managers()   
        self._reset_cache_state()

    def _reset_cache_state(self):
        """Reset all caches and references for memory efficiency."""
        self.mel_stats[:] = 1
        self.voice_manager.mel_stats[:] = 1
        self.gpt_cond_latents = self.speaker_embeddings = None
        self.voice_manager.gpt_cond_latents = self.voice_manager.speaker_embeddings = None
        self.last_speaker_id = self.last_speaker_wav = None

    def _init_managers(self):
        """Initialize management utilities with deferred loading."""
        self.speaker_manager = None
        self.language_manager = LanguageManager(self.config)        

    def _init_components(self):
        """Initialise les modules du modèle avec une gestion d'exception uniforme."""
        try:
            self._init_gpt()
            self._init_decoder()
            from .voice_manager import VoiceManager
            self.voice_manager = VoiceManager(self.gpt, self.hifigan_decoder, self.config)
        except Exception as e:
            logger.exception(f"Échec d'initialisation du modèle : {e}")
            raise

    def _init_gpt(self):
        """Stratégie d'initialisation du GPT avec gestion du meta-device."""        
        if HAVE_ACCELERATE and init_empty_weights is not None :
            try:
                logger.info("Using accelerate.init_empty_weights() to instantiate GPT")
                with init_empty_weights():
                    self.gpt = GPT(
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
                logger_ram_used("après self.gpt = GPT() (créé en meta via init_empty_weights)")
                return
            except Exception as e:
                logger.exception(f"init_empty_weights instantiation failed, falling back to CPU instantiation : {e}")
        self.gpt = GPT(
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
        logger_ram_used("après self.gpt = GPT() (créé sur CPU fallback)")

    
    def _init_decoder(self):
        """Construit le décodeur hiFiGAN sans allouer sur GPU."""
        if HAVE_ACCELERATE and init_empty_weights is not None :
            try:
                logger.info("Using accelerate.init_empty_weights() to instantiate GPT")
                with init_empty_weights():
                    self.hifigan_decoder = HifiDecoder(
                            input_sample_rate=self.args.input_sample_rate,
                            output_sample_rate=self.args.output_sample_rate,
                            output_hop_length=self.args.output_hop_length,
                            ar_mel_length_compression=self.args.gpt_code_stride_len,
                            decoder_input_dim=self.args.decoder_input_dim,
                            d_vector_dim=self.args.d_vector_dim,
                            cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
                        ) 
                    logger_ram_used("après self.hifigan_decoder = HifiDecoder(créé sur meta)")
                return
            except Exception:
                logger.exception("init_empty_weights instantiation failed, falling back to CPU instantiation")
        self.hifigan_decoder = HifiDecoder(
                            input_sample_rate=self.args.input_sample_rate,
                            output_sample_rate=self.args.output_sample_rate,
                            output_hop_length=self.args.output_hop_length,
                            ar_mel_length_compression=self.args.gpt_code_stride_len,
                            decoder_input_dim=self.args.decoder_input_dim,
                            d_vector_dim=self.args.d_vector_dim,
                            cond_d_vector_in_each_upsampling_layer=self.args.cond_d_vector_in_each_upsampling_layer,
                        ) 
    
    @staticmethod
    def init_from_config(config: XttsConfig):
        # Retourne une instance initialisée à partir d'un objet config
        logger_ram_used("avant init_from_config")
        return XTTS(config)

    # **Libération de la mémoire**
    # @ staticmethod
    def empty_cache(self):
        # Forcer GC et vider le cache CUDA si présent
        if self.checkpoint is not None:
            self.checkpoint = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        """Force la libération du cache GPU DirectML."""
        try:            
            torch_directml.device()  # remet un handle valide
            gc.collect()
            time.sleep(0.05)
            # Forcer le driver D3D12 à purger les heaps
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except Exception as e:
            logger.exception(f'flush_dml Exception : {e}')    
        logger_ram_used("après empty_cache")

    def unload_models(self):
        logger_ram_used("avant unload_models")
        del self.gpt
        del self.hifigan_decoder 
        del self.tokenizer 
        del self.speaker_manager 
        del self.gpt_cond_latents
        del self.speaker_embeddings 
        del self.mel_stats 
        del self.voice_manager 
        flush_dml()
        logger_ram_used("after del unload_models")
        self.gpt = None
        self.hifigan_decoder = None
        self.tokenizer = None
        self.speaker_manager = None
        self.gpt_cond_latents = None
        self.speaker_embeddings = None
        self.mel_stats = torch.ones(1)
        self.voice_manager = torch.ones(1)
        self.empty_cache()
        logger_ram_used("après unload_models")

    def load_checkpoint_core(self,progress_cb: Optional[Callable] = None):
        """Load model weights with unified progress management."""
        progress = 15        
        progress = self._load_tokenizer(progress)
        progress = self._load_speaker_manager(progress)
                
        progress = self._load_hifigan_decoder(progress,progress_cb)                        
        progress = self._load_gpt_weights(progress,progress_cb)
        progress = self._load_latent_norm_weights(progress,progress_cb)
        progress = self._load_gpt_shards(progress,progress_cb)
        dtype = self.TARGET_DTYPE
        if (self.gpt is not None) :
            try:
                if (self.gpt.gpt.dtype !=dtype) or (self.gpt.gpt.device.type !=self.config.device) :
                    self.gpt.to(device=self.config.device, dtype=dtype)
            except TypeError:
                if (self.gpt.gpt.device.type !=self.config.device) :
                    self.gpt.to(self.config.device)
                if (self.gpt.gpt.dtype !=dtype) :
                    self.gpt.to(dtype)
            self.empty_cache()
        fname = "ckpt_mel.safetensors"
        model_path = os.path.join(self.config.model_path, fname)
        self.checkpoint = load_file(model_path)
        self.mel_stats.copy_(self.checkpoint["mel_stats"])                
        self.voice_manager.mel_stats = self.mel_stats
        logger.info(f"load {fname} : ok")                
        self.empty_cache()
        if self.mel_stats is not None:
            try:
                self.mel_stats.to(device=self.config.device, dtype=dtype)
                self.voice_manager.mel_stats.to(device=self.config.device, dtype=dtype)
            except TypeError:
                self.mel_stats.to(self.config.device)
                self.mel_stats.to(dtype)
                self.voice_manager.mel_stats.to(self.config.device)
                self.voice_manager.mel_stats.to(dtype)
        if progress_cb:
           progress_cb(95, "handle device placement")

        
        
    def _load_tokenizer(self, progress):
        """Load and validate tokenizer with unified error handling."""
        try:
            vocab_path = self._resolve_vocab_path()
            self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            self.tokenizer = None
        return progress + 2

    def _resolve_vocab_path(self):
        """Determine correct tokenizer path from config."""
        vocab_path = self.config.model_args.tokenizer_file or self.config.model_args.tokenizer_file1
        return self._ensure_full_path(vocab_path)

    def _load_speaker_manager(self, progress):
        """Load and validate tokenizer with unified error handling."""
        try:
            speaker_path = self._resolve_speaker_path()
            self.speaker_manager = SpeakerManager(speaker_file_path=speaker_path)                    
            self.voice_manager.set_speaker_manager(self.speaker_manager)
            logger.info("Chargé : %s", speaker_path) 

        except Exception as e:
            logger.error(f"Failed to load speaker: {str(e)}")
            self.speaker_manager = None
        return progress

    def _resolve_speaker_path(self):
        """Determine correct speaker path from config."""
        speaker_file = self.config.model_args.speaker_file or self.config.model_args.speaker_file1
        return self._ensure_full_path(speaker_file)

    def _ensure_full_path(self, relative_path):
        """Ensure full path for file resources."""
        model_path = self.config.model_path
        return os.path.join(model_path, relative_path) if not os.path.exists(relative_path) else relative_path
    
    def _load_hifigan_decoder(self, progress,progress_cb: Optional[Callable] = None):
        """Apply HiFi-GAN decoder weight transformations."""
        path =  "ckpt_hfd_f8e4.safetensors"
        self.checkpoint = load_file(os.path.join(self.config.model_path, path))
        self.hifigan_decoder.load_state_dict(self.checkpoint, strict=True,assign=True)
        progress += 5
        self.empty_cache()
        logger.info(f"load {path} : ok")
        if progress_cb:
            progress_cb(progress, path)                
        return progress
    
    def _load_gpt_weights(self, progress,progress_cb: Optional[Callable] = None):
        """Apply GPT core weight transformations."""
        path = "ckpt_gpt1_f8e4.safetensors"
        self.checkpoint = load_file(os.path.join(self.config.model_path, path))
        self.gpt.load_state_dict(self.checkpoint, strict=False,assign=True)
            #{k: v.to(self.TARGET_DTYPE) for k, v in self.checkpoint.items()}, strict=False,assign=True)
            #{k: v.to(torch.float32) for k, v in self.checkpoint.items()}, strict=False,assign=True)            
        progress += 5
        self.empty_cache()
        logger.info(f"load {path} : ok")
        if progress_cb:
            progress_cb(progress, path)                
        return progress

    def _load_latent_norm_weights(self, progress,progress_cb: Optional[Callable] = None):
        """Apply latent normalization weights."""
        path = "gpt_ln_f8e4.safetensors"
        self.checkpoint = load_file(os.path.join(self.config.model_path, path))
        self.gpt.load_state_dict(self.checkpoint, strict=False,assign=True)
            #{k: v.to(self.TARGET_DTYPE) for k, v in self.checkpoint.items()}, strict=False,assign=True)
            #{k: v.to(torch.float32) for k, v in self.checkpoint.items()}, strict=False,assign=True)
        progress += 5
        self.empty_cache()
        logger.info(f"load {path} : ok")
        if progress_cb:
            progress_cb(progress, path)                
        return progress

    def _load_gpt_shards(self, progress,progress_cb: Optional[Callable] = None):
        """Load GPT attention heads with dynamic progress updates."""
        shard_dir = os.path.join(self.config.model_path, "gpt_h")
        
        for shard in sorted(
            os.listdir(shard_dir),
            key=lambda x: re.findall(r"\d+", x) and int(re.findall(r"\d+", x)[0])
        ):
            if shard.endswith("_f8e4.safetensors"):
                shard_path = os.path.join(shard_dir, shard)
                self.checkpoint = load_file(shard_path)
                self.gpt.load_state_dict(self.checkpoint, strict=False,assign=True)
                #{k: v.to(self.TARGET_DTYPE) for k, v in self.checkpoint.items()}, strict=False,assign=True)
                progress += 5
                self.empty_cache()
                logger.info(f"load {shard} : ok")
                if progress_cb:
                    progress_cb(progress, shard)                
        return progress

    def _validate_language(self, lang):
        """Ensure language is supported with appropriate fallback."""
        normalized_lang = lang.split("-")[0] if "-" in lang else lang
        if normalized_lang not in self.config.languages:
            raise ValueError(f"Unsupported language: {normalized_lang}, use: {self.config.languages}")
        return normalized_lang


    def _prepare_inference_config(self, **kwargs):
        """Apply unified config merging and validation."""
        return {
            **{
                "temperature": self.config.temperature,
                "length_penalty": self.config.length_penalty,
                "repetition_penalty": self.config.repetition_penalty,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
            },
            **kwargs
        }    

    def load_checkpoint(
        self,
        config: XttsConfig,
        eval: bool = True,
        use_deepspeed: bool = False,      
        progress_cb: Optional[Callable] = None,
        done_cb: Optional[Callable] = None,    ):
        """Charge le vocabulaire (tokenizer) puis les checkpoints de poids du modèle.
        """
        logger_ram_used("load_checkpoint avant")
        torch.set_num_threads(4)
        self.TARGET_DTYPE = get_valid_dtype_for_device(self.config.device,self.config.sdtype)

        self.load_checkpoint_core(progress_cb)        
       
        # Initialiser l'inférence GPT après que les modules soient déplacés
        # et castés — ainsi les dtypes/device sont cohérents.            try:
        if self.gpt is not None:
            self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=use_deepspeed,config=self.config)
        if eval:
            try:
                self.eval()                
            except Exception:
                logger.exception('Failed to set models to eval()')

        self.empty_cache()

    

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def update_voice_cache(self, speaker_id: str | None = None,
                            speaker_wav: str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None = None) -> None:
        """
        Update the cached voice data based on the given speaker identifier
        and/or reference audio(s). This function replaces the logic that was
        previously embedded inside `synthesize()`.
        """
        # Determine if we can fetch from the pre‑registered speakers.
        if (speaker_wav is None) and (speaker_id is not None) and \
                (speaker_id in self.speaker_manager.speakers):
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker_id].values()
        else:
            # Generate or load a new voice from the reference audio(s).
            # The `voice_settings` parameters used in `synthesize` are
            # taken directly here; adjust if needed.
            voice = self.clone_voice(
                speaker_wav,
                speaker_id,
                voice_dir=None,   # no directory for caching in this helper
                **{
                    "gpt_cond_len": self.config.gpt_cond_len,
                    "gpt_cond_chunk_len": self.config.gpt_cond_chunk_len,
                    "max_ref_length": self.config.max_ref_len,
                    "sound_norm_refs": self.config.sound_norm_refs,
                }
            )
            gpt_cond_latent = voice["gpt_conditioning_latents"]
            speaker_embedding = voice["speaker_embedding"]

        # Update internal state
        self.last_speaker_id = speaker_id
        self.last_speaker_wav = speaker_wav
        self.gpt_cond_latents_cache = gpt_cond_latent
        self.speaker_embeddings_cache = speaker_embedding

    def synthesize(
        self,
        text: str,
        config: Any | None = None,
        *,
        speaker: str | None = None,
        speaker_wav: (
            str | os.PathLike[Any] | list[str | os.PathLike[Any]] | None
        ) = None,
        language: str | None = None,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """
        Synthétise le texte fourni en utilisant la voix ou le speaker donné.
        The caching logic has been moved to `update_voice_cache()`.
        """
        #with maybe_autocast_fp16(self.config.device, self.config.sdtype):
        if True:
            # Determine speaker_id from kwargs if present
            if (speaker_id := kwargs.pop("speaker_id", None)) is not None:
                speaker = speaker_id
            speaker_id = speaker

            # Remove legacy options that are no longer needed
            for key in ("use_griffin_lim", "do_trim_silence", "extra_aux_input"):
                kwargs.pop(key, None)

            assert (
                ("zh-cn" if language == "zh" else language in self.config.languages)
                ), (
                    f" ❗ Language {language} is not supported, use : {self.config.languages}"
                )

            inference_settings = {
                "temperature": self.config.temperature,
                "length_penalty": self.config.length_penalty,
                "repetition_penalty": self.config.repetition_penalty,
                "top_k": self.config.top_k,
                "top_p": self.config.top_p,
            }
            inference_settings.update(kwargs)

            # Update the voice cache if needed
            if (self.last_speaker_id != speaker_id) or \
               (self.last_speaker_wav != speaker_wav) or \
               (self.gpt_cond_latents_cache is None) or \
               (self.speaker_embeddings_cache is None):
                self.voice_manager.update_voice_cache(speaker_id=speaker_id,
                                            speaker_wav=speaker_wav)
                
            # Retrieve cached data from the manager
            #gpt_cond_latent = maybe_to_fp16(self.voice_manager.gpt_cond_latents_cache, self.config.device, self.config.sdtype)
            #speaker_embedding = maybe_to_fp16(self.voice_manager.speaker_embeddings_cache)
            #gpt_cond_latent = maybe_to_fp16(gpt_cond_latent)
            #speaker_embedding = maybe_to_fp16(speaker_embedding)
            gpt_cond_latent = maybe_to_fp16(self.voice_manager.gpt_cond_latents)
            speaker_embedding = maybe_to_fp16(self.voice_manager.speaker_embeddings)

            # Now the caches are ready
            return self.inference(text, language, gpt_cond_latent,
                                  speaker_embedding, **inference_settings)
            

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
        seed: int = -1,
        enable_text_splitting: bool = False,
        **hf_generate_kwargs: Any,
    ):
        """Procède à la génération (GPT -> HiFiGAN) et retourne un ndarray audio.
        Arguments et comportements préservent ceux de l'implémentation d'origine.
        """
        #with maybe_autocast_fp16(self.config.device, self.config.sdtype):
        if True:
        #with maybe_autocast_fp16(self.config.device, self.config.sdtype):
            setup_seed(seed) 
            language = language.split("-")[0]
            length_scale = 1.0 / max(speed, 0.05)
            #gpt_cond_latent = gpt_cond_latent.to(self.config.device)
            #speaker_embedding = speaker_embedding.to(self.config.device)
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
            #gpt_cond_latent = force_to_fp16(gpt_cond_latent)
            with torch.no_grad():
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.args.gpt_batch_size,
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
                #gpt_latents = gpt_latents.to(dtype=torch.float32)
                #speaker_embedding = speaker_embedding.to(dtype=torch.float32)
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())
            flush_dml()
            return torch.cat(wavs, dim=0).float().numpy()

    def eval(self):
        """Passe tous les sous-modèles en mode évaluation et retire weight_norm du décodeur."""
        super().eval()        
        if self.gpt is not None:
            self.gpt.eval()

        dtype = self.TARGET_DTYPE # torch.float32
        if self.hifigan_decoder is not None:
            try:
                #if (self.hifigan_decoder.device.type !=self.config.device) :
                self.hifigan_decoder.to(device="cpu", dtype=dtype)
            except Exception:
                try:
                    self.hifigan_decoder.to("cpu")
                    self.hifigan_decoder.to(dtype)   
                except Exception:
                    pass
            self.empty_cache()   
            self.hifigan_decoder.eval()
            self.hifigan_decoder.waveform_decoder.remove_weight_norm()
            try:
                #if (self.hifigan_decoder.device.type !=self.config.device) :
                self.hifigan_decoder.to(device=self.config.device, dtype=dtype)
            except Exception:
                try:
                    self.hifigan_decoder.to(self.config.device)
                    self.hifigan_decoder.to(dtype)   
                except Exception:
                    pass
            self.empty_cache()   
        return self
    

def flush_dml():
    """Force la libération du cache GPU DirectML."""
    try:
        import torch_directml, gc, time, ctypes
        torch_directml.device()  # remet un handle valide
        gc.collect()
        time.sleep(0.05)

        # Forcer le driver D3D12 à purger les heaps
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception as e:
        logger.exception(f'flush_dml Exception : {e}')
