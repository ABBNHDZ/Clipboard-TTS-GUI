"""
Configuration handling for the Clipboard TTS GUI.
"""

import json
import tkinter as tk
import logging

logger = logging.getLogger(__name__)

SETTINGS_FILE = "appCfg_aTTS.json"


class AppConfig:
    """Configuration of the application settings."""

    def __init__(self):
        # Model and device
        self.model_path: str = ""
        self.device: str = "auto"
        self.qntf: str = "fp16"

        # Voice / text
        self.lang: str = "en"
        self.lang_rtl: str = "ar"
        self.lang_ltr: str = "fr"
        self.speaker: str = "Tammie Ema"
        self.speaker_wav: str = ""

        self.u_lang: bool = False
        self.u_speaker: bool = False
        self.u_speaker_wav: bool = False

        # Advanced parameters
        self.speed: float = 1.0
        self.temp: float = 0.75
        self.lenp: float = 1.0
        self.repp: float = 5.0
        self.topk: int = 50
        self.topp: float = 0.85

 
        self.font_size_var:int = 14   
        self.font_family_var:str = "Arial" 
        # Direction variable – Auto (default)/ RTL / LTR
        self.text_dir_var:str = "Auto" 
        self.text_min_var:int = 80 
        self.text_max_var:int = 120 

        self.u_speed: bool = False
        self.u_temp: bool = False
        self.u_lenp: bool = False
        self.u_repp: bool = False
        self.u_topk: bool = False
        self.u_topp: bool = False
        
        self.auto_load: bool = False
        # Clipboard toggle
        self.auto_clipboard: bool = False

    def load_from_json(self, path: str = SETTINGS_FILE) -> bool:
        """Load settings from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                setattr(self, k, v)
            return True
        except Exception as e:
            logger.warning("Failed to load config: %s", e)
            return False

    def save_to_json(self, path: str = SETTINGS_FILE):
        """Save current settings to JSON file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(vars(self), f, indent=4)
        except Exception as e:
            logger.warning("Failed to save config: %s", e)


# -----------------------------------------------------------

def load_config(app , cfg: AppConfig) -> None:
    """Load settings from JSON file and set UI variables in the App."""
    if cfg.load_from_json():
        # Set Tkinter variable values
        app.model_path_var.set(cfg.model_path)
        app.device_var.set(cfg.device)
        app.qntf_var.set(cfg.qntf)

        app.lang_var.set(cfg.lang)
        app.speaker_var.set(cfg.speaker)        
        app.speaker_wav_var.set(cfg.speaker_wav)

        app.use_lang_var.set(cfg.u_lang)
        app.use_speaker_var.set(cfg.u_speaker)        
        app.use_speaker_wav.set(cfg.u_speaker_wav)       

        app.speed_var.set(cfg.speed)
        app.temp.set(cfg.temp)
        app.lenp.set(cfg.lenp)
        app.repp.set(cfg.repp)
        app.topk.set(cfg.topk)
        app.topp.set(cfg.topp)

        app.use_speed.set(cfg.u_speed)
        app.use_temp.set(cfg.u_temp)
        app.use_lenp.set(cfg.u_lenp)
        app.use_repp.set(cfg.u_repp)
        app.use_topk.set(cfg.u_topk)
        app.use_topp.set(cfg.u_topp)

        app.font_size_var.set(cfg.font_size_var)
        app.font_family_var.set(cfg.font_family_var)
        app.text_dir_var.set(cfg.text_dir_var) 
        app.text_min_var.set(cfg.text_min_var)
        app.text_max_var.set(cfg.text_max_var)

        app.auto_load.set(cfg.auto_load)
        app.auto_clipboard.set(cfg.auto_clipboard)

        # Update the TTSManager with loaded values
        app.ttsMan.model_path = cfg.model_path
        app.ttsMan.device = cfg.device
        app.ttsMan.Qntf = cfg.qntf
        


def save_settings(app , cfg: AppConfig) -> None:
    """Save current settings to JSON file."""
    # Copy values from Tkinter variables into the config object
    cfg.model_path = str(app.model_path_var.get())
    cfg.device = str(app.device_var.get())
    cfg.qntf = str(app.qntf_var.get())

    cfg.lang = str(app.lang_var.get())
    cfg.speaker = str(app.speaker_var.get())
    cfg.speaker_wav = str(app.speaker_wav_var.get())

    cfg.u_lang = bool(app.use_lang_var.get())
    cfg.u_speaker = bool(app.use_speaker_var.get())
    cfg.u_speaker_wav = bool(app.use_speaker_wav.get())

    cfg.speed = float(app.speed_var.get())
    cfg.temp = float(app.temp.get())
    cfg.lenp = float(app.lenp.get())
    cfg.repp = float(app.repp.get())
    cfg.topk = int(app.topk.get())
    cfg.topp = float(app.topp.get())

    cfg.u_speed = bool(app.use_speed.get())
    cfg.u_temp = bool(app.use_temp.get())
    cfg.u_lenp = bool(app.use_lenp.get())
    cfg.u_repp = bool(app.use_repp.get())
    cfg.u_topk = bool(app.use_topk.get())
    cfg.u_topp = bool(app.use_topp.get())

    cfg.font_size_var = int(app.font_size_var.get())
    cfg.font_family_var = str(app.font_family_var.get())
    cfg.text_dir_var = str(app.text_dir_var.get()) 
    cfg.text_min_var = int(app.text_min_var.get())
    cfg.text_max_var = int(app.text_max_var.get())

    cfg.auto_load = bool(app.auto_load.get())
    cfg.auto_clipboard = bool(app.auto_clipboard.get())
    # Persist to JSON
    cfg.save_to_json()


def register_trace(app , cfg: AppConfig) -> None:
    """Bind each Tkinter variable to save settings when it changes."""
    var_map = [
        app.model_path_var,
        app.device_var,
        app.qntf_var,

        app.lang_var,
        app.speaker_var,        
        app.speaker_wav_var,

        app.use_lang_var,
        app.use_speaker_var,   
        app.use_speaker_wav,

        app.speed_var,
        app.temp,
        app.lenp,
        app.repp,
        app.topk,
        app.topp,

        app.use_speed,
        app.use_temp,
        app.use_lenp,
        app.use_repp,
        app.use_topk,
        app.use_topp,

        app.auto_load,
        app.auto_clipboard,

        app.font_size_var,
        app.font_family_var,
        app.text_dir_var,
        app.text_min_var,
        app.text_max_var,
    ]
    for var in var_map:
        var.trace_add("write", lambda *_: save_settings(app, cfg))