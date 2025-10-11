"""
xtts2m.xttsManager — model manager, audio player and synthesis worker
EN:
This module contains the high-level runtime pieces:
- AudioPlayer: asynchronous audio playback (uses sounddevice if present).
- SynthesisWorker: background thread turning text into audio via TTSManager.
- TTSManager: loads/unloads models, coordinates synthesis and model state.

FR:
Ce module rassemble les éléments runtime principaux :
- AudioPlayer : lecture audio asynchrone (utilise sounddevice si présent).
- SynthesisWorker : thread en arrière-plan convertissant texte en audio.
- TTSManager : charge/décharge les modèles et coordonne la synthèse.
"""

from __future__ import annotations

import os
import threading
import gc
import time
import queue
from typing import Optional, Callable, List

import numpy as np
import torch
import logging
import sounddevice as sd

from xtts2m.model import XTTS
from xtts2m.xttsConfig import XttsConfig, SUPPORTED_LANGUAGES, DEFAULT_SPEAKERS
from xtts2m.tokenizer import split_text, multilingual_cleaners
from xtts2m.utils import get_valid_dtype_for_device, logger_ram_used, resolve_device,SUPPORTED_FP16_GPUS


logger = logging.getLogger(__name__)

# Utilitaire : créer un répertoire si nécessaire
def ensure_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        logger.info(f"Error ensure_dir : {e}")
        pass
    return path

class AudioPlayer:
    """Lecteur audio simple basé sur sounddevice.OutputStream.

    - Les buffers audio (numpy float32 mono) sont empilés dans une Queue.
    - Supporte pause, reprise et arrêt propre.
    - Chaque item placé dans la queue est un tuple (wav, texte) pour permettre
      la mise en surbrillance de la portion jouée dans l'UI.
    """

    class StopSignal:
        """Objet signal pour indiquer la fin de flux."""
        pass

    def __init__(self, app=None):
        self.audio_queue = queue.Queue()
        self.pause_event = threading.Event()
        self.pause_event.set()  # True => lecture autorisée
        self.stop_event = threading.Event()
        self.running = False
        self.playing = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        self.sample_rate = 22050

        #from appConfig import AppConfig
        #from clipboard_tts_gui import App
        
        self.appCfg : Optional['AppConfig'] = None
        self.app: Optional['App'] = None

    def set_sample_rate(self, sr: int):
        if isinstance(sr, int) and sr > 0:
            self.sample_rate = sr

    def start(self) -> None:
        """Démarre le thread de lecture si nécessaire."""
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._run, name="audio_player", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        """Boucle de lecture qui consomme la queue et écrit dans un stream sounddevice."""
        try:            
            blocksize = int(self.sample_rate * 0.5)  # bloc de 500 ms
            with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as stream:
                while self.running and not self.stop_event.is_set():
                    self.playing = False
                    if self.app:
                        self.app.remove_highlight()
                    try:
                        item = self.audio_queue.get(timeout=1.0)
                        if isinstance(item, AudioPlayer.StopSignal):
                            break
                        self.playing = True
                        wav, text = item
                        wav = wav.astype('float32').flatten()

                        if self.app and text:
                            # Méthode UI attendue pour surbrillance
                            try:
                                self.app._highlight_text(text)
                            except Exception as e:
                                logger.warning(f"highlight text error : {e}")
                                pass

                        for i in range(0, len(wav), blocksize):
                            if self.stop_event.is_set():
                                break
                            self.pause_event.wait()
                            chunk = wav[i:i+blocksize]
                            stream.write(chunk)
                    except queue.Empty:
                        time.sleep(0.01)
        except Exception as e:
            logger.warning(f"Stream error: {e}")
        finally:            
            self.playing = False
            self.running = False
            self.paused = False
            self.stop_event.clear()
            if self.app:
                self.app.remove_highlight()
            

    def enqueue(self, wav: np.ndarray, text: Optional[str] = None) -> None:
        """Place un buffer audio dans la queue avec le texte associé."""
        if text is None:
            text = ""
        self.audio_queue.put((wav, text))
        if self.app:
            try:
                self.app._set_status("Lecture …")
                self.app._update_state_indicator("playing")
            except Exception:
                pass

    def pause(self) -> None:
        self.pause_event.clear()
        self.paused = True
        if self.app:
            try:
                self.app._set_status("Pause …")
                self.app._update_state_indicator("paused")
            except Exception:
                pass

    def resume(self) -> None:
        self.pause_event.set()
        self.paused = False
        if self.app:
            try:
                self.app._set_status("Lecture …")
                self.app._update_state_indicator("playing")
            except Exception:
                pass

    def stop(self, drain: bool = False) -> None:
        if self.app:
            try:
                self.app._set_status("Stop …")
                self.app._update_state_indicator("stopped")
            except Exception:
                pass
        self.stop_event.set()
        if drain:
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        self.audio_queue.put(AudioPlayer.StopSignal())


class SynthesisWorker:
    """Worker consommant une file de textes et appelant le manager pour synthèse."""

    def __init__(self, ttsMan: TTSManager, audio_player: AudioPlayer):
        self.ttsMan = ttsMan
        self.audio = audio_player
        self.text_queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.synthesizing = False

        #from appConfig import AppConfig
        #from clipboard_tts_gui import App

        self.appCfg : Optional['AppConfig'] = None
        self.app: Optional['App'] = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break

    def submit_text(self, text: str):
        if text and text.strip():
            raw = text.strip()
            # split_sentence renvoie une liste de segments
            try:
                text_max = self.app.ttsMan.tts_model.tokenizer.char_limits[self.appCfg.lang] 
                text_min = self.app.appCfg.text_min
                if text_max > self.app.appCfg.text_max:
                    text_max = self.app.appCfg.text_max
                raw_list = split_text(raw, text_min, text_max)
            except Exception:
                raw_list = [raw]

            for raw_txt in raw_list:
                cleaned_txt = multilingual_cleaners(text=raw_txt, lang=self.appCfg.lang)
                cleaned_txt = cleaned_txt.strip().lower()
                if len(cleaned_txt) > 1:
                    self.text_queue.put((cleaned_txt, raw_txt))

    def _run(self):
        while self.running:
            try:
                self.synthesizing = False
                text, raw = self.text_queue.get()
                if not text:
                    time.sleep(0.01)
                    continue
                if not self.running:
                    break
                self.synthesizing = True
                wav = self.ttsMan.synthesize(text)
                if not self.running:
                    break
                if wav is not None:
                    # audio playback expects float32 numpy arrays
                    self.audio.enqueue(np.asarray(wav, dtype=np.float32), text=raw)
            except Exception:
                logger.exception("[SynthWorker] Error")
                self.synthesizing = False

class TTSManager:
    """Gestionnaire principal qui charge le modèle et synthétise du texte.
    - load_async : chargement asynchrone avec callbacks de progression
    - unload : libère le modèle et la mémoire
    - synthesize : wrapper vers XTTS.synthesize
    """

    def __init__(self, model_path: str, device: str = "auto", sel_dtype: str = ""):
        self.model_path = ensure_dir(model_path)
        self.device = resolve_device(device,sel_dtype, True)
        self.tts_model: Optional[XTTS] = None
        self.config: Optional[XttsConfig] = None
        self.is_model_loaded = False
        self._loading_lock = threading.Lock()

        #from appConfig import AppConfig
        #from clipboard_tts_gui import App

        self.appCfg : Optional['AppConfig'] = None
        self.app: Optional['App'] = None

    def load_async(self, model_path: str, device: str, progress_cb: Optional[Callable]=None, done_cb: Optional[Callable]=None):
        """Charge le modèle en tâche de fond.
        progress_cb(percent:int, message:str) est appelé pour indiquer la progression.
        done_cb(success:bool) est appelé à la fin.
        """
        self.device = resolve_device(device,"",True)                    
        self.TARGET_DTYPE = get_valid_dtype_for_device(self.device,self.appCfg.sdtype)
        self.appCfg.sdtype = "float16" if self.TARGET_DTYPE == torch.float16 else "float32"        
        #self.appCfg.sdtype_var.set(self.appCfg.sdtype)

        def _load():
            with self._loading_lock:       
                logger_ram_used("avant chargement")
                try:                    
                    if self.tts_model is not None:
                        self.tts_model = None

                    logger_ram_used("après self.tts_model = None")

                    self.is_model_loaded = False                    
                    if progress_cb:
                        progress_cb(2, "Lecture config")
                    self.config = XttsConfig(model_path=model_path, device=self.device)
                    self.config.sdtype = self.appCfg.sdtype
                    self.TARGET_DTYPE = get_valid_dtype_for_device(self.config.device,self.config.sdtype)
                    
                    # Assurer que le flag global reste False pendant tout le chargement
                    # (certaines fonctions appelées ci‑dessus peuvent avoir mis le flag à True)
                    time.sleep(0.1)
                    logger_ram_used("après self.config = XttsConfig")

                    if progress_cb:
                        progress_cb(5, "Initialisation des modules")
                    # Initialise le modèle; n'utilise fp16/autocast que si CUDA est disponible
                    #with maybe_autocast_fp16(self.device):
                    self.tts_model = XTTS(self.config)
                    time.sleep(0.1)
                    logger_ram_used("après self.tts_model = XTTS")

                    if progress_cb:
                        progress_cb(10, "Chargement des checkpoints (début)")
                    self.tts_model.app = None

                    def model_progress(mpercent: int, message: str):
                        try:
                            if progress_cb:
                                overall = int(15 + (mpercent * 80) / 100)
                                progress_cb(min(max(overall, 15), 95), message)
                                logger_ram_used(message)
                        except Exception:
                            pass

                    # sdtype depuis la config GUI
                    self.config.sdtype = self.appCfg.sdtype

                    # Charger checkpoints (autocast fp16 uniquement sur CUDA)
                    #with maybe_autocast_fp16(self.device):
                    self.tts_model.load_checkpoint(
                        config=self.config,      
                        progress_cb=model_progress,
                    )
                    time.sleep(0.1)

                    if progress_cb:
                        progress_cb(100, "Terminé : Modèle chargé.")
                    self.is_model_loaded = True
                    if done_cb:
                        done_cb(True)
                    logger_ram_used("après chargemen")
                except Exception as e:
                    logger.exception(f"[TTSManager] Load failed: {e}")
                    self.tts_model = None
                    self.is_model_loaded = False
                    if done_cb:
                        done_cb(False)
                    logger_ram_used("après Exception chargement")

        threading.Thread(target=_load, daemon=True).start()

    def unload(self):
        import torch
        logger_ram_used("avant unload")
        import psutil        
        proc = psutil.Process(os.getpid())
        logger.info(f'RSS avant: {proc.memory_info().rss / 1e6:.2f} MB')       
        try:                    
            if self.tts_model is not None:
                try:
                    self.tts_model.unload_models()
                except Exception as e:
                    logger.error(f"Error during unload_models: {e}")
                #del self.tts_model
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"Error during cuda.empty_cache: {e}")

            # Break internal references
            #for attr in ["gpt", "hifigan_decoder", "speaker_manager", "tokenizer"]:
            #    if hasattr(self.tts_model, attr):
            #        try:
            #            delattr(self.tts_model, attr)
            #        except Exception as e:
            #            logger.error(f"Error deleting attribute {attr}: {e}")

            #self.tts_model = None
            self.is_model_loaded = False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        logger_ram_used("après unload")
        time.sleep(0.5)
        # unload comme ci-dessus
        gc.collect()
        time.sleep(0.5)
        logger.info(f'RSS après:{proc.memory_info().rss / 1e6:.2f} MB')
        self.app.set_status("Terminé : Modèle déchargé.")
        

    def get_supported_languages(self) -> List[str]:
        if self.config:
            return sorted(set(self.config.languages))
        return SUPPORTED_LANGUAGES

    def get_supported_speakers(self) -> List[str]:
        speakers = []
        try:
            if self.tts_model and hasattr(self.tts_model, 'speaker_manager') and self.tts_model.speaker_manager.speakers is not None:
                speakers.extend(self.tts_model.speaker_manager.speakers)
                return sorted(set(speakers))
        except Exception:
            pass
        return DEFAULT_SPEAKERS

    def synthesize(self, text: str):
        if not self.is_model_loaded or not self.tts_model:
            logger.warning("Model not loaded")
            return None
        # Autocast fp16 uniquement si on tourne sur CUDA
        kwargs = {"text": text}
        try:
            if self.appCfg.u_lang:
                kwargs['language'] = self.appCfg.lang
            if self.appCfg.u_speaker:
                kwargs['speaker_id'] = self.appCfg.speaker
            elif self.appCfg.u_speaker_wav:
                kwargs['speaker_wav'] = self.appCfg.speaker_wav
            if self.appCfg.u_speed:
                kwargs['speed'] = self.appCfg.speed
            if self.appCfg.u_temp:
                kwargs['temperature'] = self.appCfg.temp
            if self.appCfg.u_lenp:
                kwargs['length_penalty'] = self.appCfg.lenp
            if self.appCfg.u_repp:
                kwargs['repetition_penalty'] = self.appCfg.repp
            if self.appCfg.u_topk:
                kwargs['top_k'] = self.appCfg.topk
            if self.appCfg.u_topp:
                kwargs['top_p'] = self.appCfg.topp
            if self.appCfg.u_seed:
                kwargs['seed'] = self.appCfg.seed
        except Exception:
            pass
        try:
            start_time = time.time()
            logger.info(f"Input: [ {text} ]")
            ret = self.tts_model.synthesize(**kwargs)
            process_time = time.time() - start_time
            logger.info("Processing time: %.3f", process_time)
            return ret
        except Exception as e:
            logger.exception(f"synthesize error : {e}")
        return None