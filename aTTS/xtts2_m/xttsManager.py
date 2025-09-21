"""
Manager TTS formaté et commenté en français.
Contient :
- AudioPlayer : lecteur audio asynchrone (utilise sounddevice si disponible)
- SynthesisWorker : thread consommant une file de textes et appelant TTSManager
- TTSManager : gestion du chargement/déchargement du modèle et interface de synthèse

Ce fichier est une transcription commentée de `xtts2_m/xttsManager.py`.
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

from .model import XTTS
from .xttsConfig import XttsConfig, SUPPORTED_LANGUAGES, DEFAULT_SPEAKERS
from .tokenizer import split_text, multilingual_cleaners


# Utilitaire : créer un répertoire si nécessaire
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
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
        self.app = app

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
            import sounddevice as sd
            blocksize = int(self.sample_rate * 0.5)  # bloc de 500 ms
            with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as stream:
                while self.running and not self.stop_event.is_set():
                    self.playing = False
                    self.app._remove_highlight()
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
                            except Exception:
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
            print(f"[AudioPlayer] Stream error: {e}")
        finally:            
            self.playing = False
            self.running = False
            self.paused = False
            self.stop_event.clear()
            self.app._remove_highlight()
            

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
        self.appCfg = None
        self.app = None

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
                text_min = self.app.appCfg.text_min_var
                if text_max > self.app.appCfg.text_max_var:
                    text_max = self.app.appCfg.text_max_var
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
                    self.audio.enqueue(np.asarray(wav, dtype=np.float32), text=raw)
            except Exception as e:
                print(f"[SynthWorker] Error: {e}")
                self.synthesizing = False


class TTSManager:
    """Gestionnaire principal qui charge le modèle et synthétise du texte.

    - load_async : chargement asynchrone avec callbacks de progression
    - unload : libère le modèle et la mémoire
    - synthesize : wrapper vers XTTS.synthesize
    """

    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = ensure_dir(model_path)
        self.device = device
        self.tts_model: Optional[XTTS] = None
        self.config: Optional[XttsConfig] = None
        self.is_model_loaded = False
        self._loading_lock = threading.Lock()
        self.appCfg = None
        self.app = None

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested

    def load_async(self, model_path: str, device: str, progress_cb: Optional[Callable]=None, done_cb: Optional[Callable]=None):
        """Charge le modèle en tâche de fond.
        progress_cb(percent:int, message:str) est appelé pour indiquer la progression.
        done_cb(success:bool) est appelé à la fin.
        """
        def _load():
            with self._loading_lock:
                try:
                    self.is_model_loaded = False
                    self.device = self._resolve_device(device)
                    if progress_cb:
                        progress_cb(2, "Lecture config")
                    self.config = XttsConfig(model_path=model_path, device=self.device)
                    time.sleep(0.1)

                    if progress_cb:
                        progress_cb(5, "Initialisation des modules")
                    self.tts_model = XTTS(self.config)
                    time.sleep(0.1)

                    if progress_cb:
                        progress_cb(10, "Chargement des checkpoints (début)")
                    self.tts_model.app = None

                    def model_progress(mpercent: int, message: str):
                        try:
                            if progress_cb:
                                overall = int(15 + (mpercent * 80) / 100)
                                progress_cb(min(max(overall, 15), 95), message)
                        except Exception:
                            pass

                    # qntf depuis la config GUI
                    try:
                        self.config.Qntf = self.appCfg.qntf
                    except Exception:
                        pass

                    self.tts_model.load_checkpoint(
                        config=self.config,
                        model_path=self.config.model_path,
                        progress_cb=model_progress,
                    )
                    time.sleep(0.1)

                    if progress_cb:
                        progress_cb(100, "Terminé : Modèle chargé.")
                    self.is_model_loaded = True
                    if done_cb:
                        done_cb(True)
                except Exception as e:
                    print(f"[TTSManager] Load failed: {e}")
                    self.tts_model = None
                    self.is_model_loaded = False
                    if done_cb:
                        done_cb(False)

        threading.Thread(target=_load, daemon=True).start()

    def unload(self):
        try:
            if self.tts_model is not None:
                del self.tts_model
            self.tts_model = None
            self.is_model_loaded = False
        finally:
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

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
            print("Model not loaded")
            return None
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
        except Exception:
            pass
        try:
            return self.tts_model.synthesize(**kwargs)
        except Exception as e:
            print(f"[TTSManager] synthesize error: {e}")
            return None
