"""
Configuration formatée et commentée en français pour XTTS v2.

Ce module contient :
- constantes globales (langues supportées, speakers par défaut, etc.)
- classes de configuration : XttsConfig, ModelArgs, AudioConfig
- fonctions utilitaires pour charger la config depuis un JSON

Le contenu est basé sur `xtts2_m/xttsConfig.py` mais reformatté et
documenté en français pour faciliter la lecture et la maintenance.
"""

import json
import os
import torch
from typing import Any

# -----------------------------------------------------------------------------
# Constantes globales
# -----------------------------------------------------------------------------

# Langues supportées par défaut par XTTS v2
SUPPORTED_LANGUAGES = [
    "ar",
    "en",
    "es",
    "fr",
    "de",
    "it",
    "ru",
]

# Quelques voices/speakers par défaut fournis dans le projet
DEFAULT_SPEAKERS = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
]

# Quantifications supportées (ex: précision de modèle)
DEFAULT_QNTF = [
    "fp32",
    "fp16",
    "bf16",
    "f8e4",
    "f8e5",
]

# Valeurs par défaut utilisées si aucune configuration externe n'est fournie
DEFAULT_LANGUAGE = "ar"
DEFAULT_SPEAKER = "Tammie Ema"
DEFAULT_MODEL_PATH = "../model_xtts_v2/"

# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------

def _resolve_device(requested: str) -> str:
    """Résout et normalise la chaîne `device` pour PyTorch.

    - 'auto' -> 'cuda' si CUDA disponible sinon 'cpu'
    - si 'cuda' est demandé mais non disponible, retombe sur 'cpu'
    - sinon renvoie la valeur telle quelle
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


# -----------------------------------------------------------------------------
# Classe principale de configuration
# -----------------------------------------------------------------------------
class XttsConfig:
    """Classe de configuration pour XTTS v2.

    Cette classe charge des valeurs par défaut puis, si un fichier
    `config.json` est trouvé dans `model_path`, surcharge ses attributs
    avec ceux du JSON.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, FileName: str = "config.json", device: str = "auto"):
        # Emplacement des fichiers du modèle
        self.model_path = model_path
        self.config_fn = FileName

        # Device runtime (cpu / cuda)
        self.device = _resolve_device(device)

        # Quantification et flags d'initialisation
        self.Qntf = "fp16"
        self.model_is_Initialized = False
        self.model_is_loded = False

        # Objets de configuration imbriqués
        self.model_args = ModelArgs()
        self.audio = AudioConfig()

        # Paramètres d'entraînement / exécution (valeurs par défaut)
        self.num_chars = 250
        self.mixed_precision = False
        self.precision = "fp16"
        self.epochs = 1000
        self.batch_size = 32
        self.eval_batch_size = 16
        self.grad_clip = 0.0
        self.scheduler_after_epoch = True
        self.lr = 0.001
        self.optimizer = "radam"
        self.use_grad_scaler = False
        self.allow_tf32 = False
        self.cudnn_enable = True
        self.cudnn_deterministic = False
        self.cudnn_benchmark = False
        self.training_seed = 54321
        self.model = "xtts"
        self.num_loader_workers = 0
        self.num_eval_loader_workers = 0
        self.use_noise_augment = False
        self.use_phonemes = False
        self.phonemizer = None
        self.phoneme_language = None
        self.compute_input_seq_cache = False
        self.text_cleaner = None
        self.enable_eos_bos_chars = False
        self.test_sentences_file = ""
        self.phoneme_cache_path = None
        self.characters = None
        self.add_blank = False
        self.batch_group_size = 0
        self.loss_masking = None
        self.min_audio_len = 1
        self.max_audio_len = float("inf")
        self.min_text_len = 1
        self.max_text_len = float("inf")
        self.compute_f0 = False
        self.compute_energy = False
        self.compute_linear_spec = False
        self.precompute_num_workers = 0
        self.start_by_longest = False
        self.shuffle = False
        self.drop_last = False
        self.test_sentences = []
        self.eval_split_max_size = None
        self.eval_split_size = 0.01
        self.use_speaker_weighted_sampler = False
        self.speaker_weighted_sampler_alpha = 1.0
        self.use_language_weighted_sampler = False
        self.language_weighted_sampler_alpha = 1.0
        self.use_length_weighted_sampler = False
        self.length_weighted_sampler_alpha = 1.0

        # Paramètres de génération / sampling
        self.temperature = 0.75
        self.length_penalty = 1.0
        self.repetition_penalty = 5.0
        self.top_k = 50
        self.top_p = 0.85
        self.num_gpt_outputs = 1
        self.gpt_cond_len = 30
        self.gpt_cond_chunk_len = 4
        self.max_ref_len = 30
        self.sound_norm_refs = False

        # Langues supportées par la configuration
        self.languages = ["en", "ar", "fr", "es", "de", "it", "ru"]

        # Chargement automatique depuis config.json si disponible
        config_path = None
        if self.model_path is not None and os.path.exists(self.model_path) and self.config_fn is not None:
            config_path = os.path.join(self.model_path, self.config_fn)
            if not os.path.exists(config_path):
                config_path = None

        if config_path is not None:
            # Si le fichier existe, on charge et on applique les valeurs
            if os.path.exists(config_path):
                self.load_from_json(config_path)

    def load_from_json(self, config_path: str = "") -> bool:
        """Charge la configuration depuis un fichier JSON et applique les valeurs.

        - Retourne True si le chargement s'est bien passé.
        - En cas d'erreur, lève une exception descriptive.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            # Charger les paramètres du modèle (s'il y en a)
            if "model_args" in config_data:
                for key, value in config_data["model_args"].items():
                    setattr(self.model_args, key, value)

            # Charger les paramètres audio (s'il y en a)
            if "audio" in config_data:
                for key, value in config_data["audio"].items():
                    setattr(self.audio, key, value)

            # Charger les autres paramètres au niveau supérieur
            for key, value in config_data.items():
                if key not in ["model_args", "audio"]:
                    setattr(self, key, value)
            return True
        except Exception as e:
            raise Exception(f"Erreur lors du chargement de la configuration : {e}")

    def to_dict(self) -> dict:
        """Sérialise la configuration en dictionnaire.

        - Inclut les sous-objets `model_args` et `audio`.
        - Exclut certains attributs internes si nécessaire.
        """
        return {
            "model_args": self.model_args.__dict__,
            "audio": self.audio.__dict__,
            "model_dir": self.model_path,
            "device": self.device,
            # Ajouter tous les autres attributs en évitant les références récursives
            **{k: v for k, v in self.__dict__.items() if k not in ["model_args", "audio", "config_path"]},
        }


# -----------------------------------------------------------------------------
# Classes internes : ModelArgs et AudioConfig
# -----------------------------------------------------------------------------
class ModelArgs:
    """Arguments et hyperparamètres liés à l'architecture du modèle.

    Contient des valeurs par défaut utilisées par le code d'entraînement et
    d'inférence.
    """

    def __init__(self):
        self.gpt_batch_size = 1
        self.enable_redaction = False
        self.kv_cache = True
        self.gpt_checkpoint = None
        self.clvp_checkpoint = None
        self.decoder_checkpoint = None
        self.num_chars = 255
        self.tokenizer_file1 = "vocab.json"
        self.speaker_file1 = "speakers_xtts.pth"
        self.tokenizer_file = None
        self.speaker_file = None
        self.gpt_max_audio_tokens = 605
        self.gpt_max_text_tokens = 402
        self.gpt_max_prompt_tokens = 70
        self.gpt_layers = 30
        self.gpt_n_model_channels = 1024
        self.gpt_n_heads = 16
        self.gpt_number_text_tokens = 6681
        self.gpt_start_text_token = 261
        self.gpt_stop_text_token = 0
        self.gpt_num_audio_tokens = 1026
        self.gpt_start_audio_token = 1024
        self.gpt_stop_audio_token = 1025
        self.gpt_code_stride_len = 1024
        self.gpt_use_masking_gt_prompt_approach = True
        self.gpt_use_perceiver_resampler = True
        self.input_sample_rate = 22050
        self.output_sample_rate = 24000
        self.output_hop_length = 256
        self.decoder_input_dim = 1024
        self.d_vector_dim = 512
        self.cond_d_vector_in_each_upsampling_layer = True
        self.duration_const = 102400


class AudioConfig:
    """Paramètres relatifs à l'audio et au prétraitement des signaux."""

    def __init__(self):
        self.sample_rate = 22050
        self.output_sample_rate = 24000
        self.fft_size = 1024
        self.win_size = 1024
        self.hop_size = 256
        self.n_mels = 80
        self.fmin = 0
        self.fmax = 8000
        self.mel_norm = "slaney"
        self.preemphasis = 0.97
        self.max_audio_length = 100000


# -----------------------------------------------------------------------------
# Fonctions d'assistance à l'extérieur de la classe
# -----------------------------------------------------------------------------

def load_config(model_path: str = DEFAULT_MODEL_PATH, FileName: str = "config.json") -> XttsConfig:
    """Renvoie une instance de `XttsConfig` chargée depuis `model_path`.

    - Si le chargement échoue, une exception est levée avec un message
      explicite.
    """
    try:
        config = XttsConfig(model_path=model_path, FileName=FileName)
        return config
    except Exception as e:
        raise Exception(f"Erreur lors du chargement de la configuration : {e}")


def dict_to_obj(d: Any) -> Any:
    """Convertit récursivement un dictionnaire en un objet arbitraire
    (accès via l'attribut pointé, ex: obj.key).

    - Utilisé quand on veut convertir une structure JSON en objet simple.
    """
    if isinstance(d, dict):
        obj = type("XttsConfig", (), {})()
        for key, value in d.items():
            setattr(obj, key, dict_to_obj(value))
        return obj
    elif isinstance(d, list):
        return [dict_to_obj(item) for item in d]
    else:
        return d
