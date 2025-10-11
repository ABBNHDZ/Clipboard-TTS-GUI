"""
AppConfig — application settings for Clipboard TTS GUI
Simple configuration container for the GUI. Handles loading/saving
settings to a JSON file and provides helpers to bind Tkinter variables.
"""

import json
import tkinter as tk
import logging

logger = logging.getLogger(__name__)

SETTINGS_FILE = "aTTSappCfg.json" # Define as a constant


class AppConfig:
    """Configuration of the application settings."""

    def __init__(self):
        # Model and device
        self.is_loaded: bool = False
        self.model_path: str = ""
        self.device: str = "auto"
        self.sdtype: str = "float16"

        self.console: bool = False
        self.theme : str = "clam"
       

        # Voice / text
        self.lang: str = "fr"
        self.lang_rtl: str = "ar"
        self.lang_ltr: str = "fr"
        self.speaker: str = "Tammie Ema"
        self.speaker_wav: str = ""

        self.u_lang: bool = True
        self.u_speaker: bool = True
        self.u_speaker_wav: bool = False

        # Advanced parameters
        self.speed: float = 1.0
        self.temp: float = 0.75
        self.lenp: float = 1.0
        self.repp: float = 5.0
        self.topk: int = 50
        self.topp: float = 0.85
        self.seed: int = 0

 
        self.font_size:int = 14   
        self.font_family:str = "Arial" 
        # Direction variable – Auto (default)/ RTL / LTR
        self.text_dir:str = "Auto" 
        self.text_min:int = 80 
        self.text_max:int = 120 

        self.u_speed: bool = True
        self.u_temp: bool = True
        self.u_lenp: bool = True
        self.u_repp: bool = True
        self.u_topk: bool = True
        self.u_topp: bool = True
        self.u_seed: bool = True
        
        self.auto_load: bool = True
        # Clipboard toggle
        self.auto_clipboard: bool = True

    def load_from_json(self, path: str = SETTINGS_FILE) -> bool:
        """Load settings from JSON file."""
        try:            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                try:
                    value = getattr(self, k)
                    if isinstance(value, bool):
                        setattr(self, k, bool(v))
                    elif isinstance(value, int):
                        setattr(self, k, int(v))
                    elif isinstance(value, float):
                        setattr(self, k, float(v))
                    elif isinstance(value, str):
                        setattr(self, k, v)
                except Exception as e:
                    logger.warning(f"Error loading setting {k}: {e}")  
            self.is_loaded = True
            return True
        except FileNotFoundError:
            logger.warning(f"Config file not found: {path}")
            self.is_loaded = False
            return False
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to decode JSON in {path}: {e}")
            self.is_loaded = False
            return False
        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")
            self.is_loaded = False
            return False

    def save_to_json(self, path: str = SETTINGS_FILE):
        """Save current settings to JSON file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(vars(self), f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save config to {path}: {e}")


    def set_app_Var_from_config(self,  app ) :
        """Load settings from JSON file and set UI variables in the App."""
        if not self.is_loaded :
            self.is_loaded = self.load_from_json()
        for attr in dir(self):
            if not attr.startswith("__"):
                value = getattr(self, attr)
                if isinstance(value, bool):
                    setattr(app, f"{attr}_var", tk.BooleanVar(value=value))
                elif isinstance(value, int):
                    setattr(app, f"{attr}_var", tk.IntVar(value=value))
                elif isinstance(value, float):
                    setattr(app, f"{attr}_var", tk.DoubleVar(value=value))
                elif isinstance(value, str):
                    setattr(app, f"{attr}_var", tk.StringVar(value=str(value)))
    

    def update_config_save_settings(self, app) -> None:
        # Copy values from Tkinter variables into the config object
        for attr in dir(self):
            if not attr.startswith("__"):
                if hasattr(app, f"{attr}_var"):
                    try:
                        value = getattr(app, f"{attr}_var").get()
                        if isinstance(getattr(self, attr), bool):
                            setattr(self, attr, bool(value))
                        elif isinstance(getattr(self, attr), int):
                            setattr(self, attr, int(value))
                        elif isinstance(getattr(self, attr), float):
                            setattr(self, attr, float(value))
                        elif isinstance(getattr(self, attr), str):
                            setattr(self, attr, str(value))
                        else:
                            #setattr(self, attr, str(value))
                            i = None
                    except Exception as e:
                        logger.warning(f"Error saving : {attr}: {e}")                    
        """Save current settings to JSON file."""
        self.save_to_json()


    def register_trace(self, app) -> None:
        """Bind each Tkinter variable to save settings when it changes."""                
        var_map = [
            app.model_path_var,
            app.device_var,
            app.sdtype_var,
            
            app.console_var,
            app.theme_var,

            app.lang_var,
            app.speaker_var,        
            app.speaker_wav_var,

            app.u_lang_var,
            app.u_speaker_var,   
            app.u_speaker_wav_var,

            app.speed_var,
            app.temp_var,
            app.lenp_var,
            app.repp_var,
            app.topk_var,
            app.topp_var,
            app.seed_var,

            app.u_speed_var,
            app.u_temp_var,
            app.u_lenp_var,
            app.u_repp_var,
            app.u_topk_var,
            app.u_topp_var,

            app.auto_load_var,
            app.auto_clipboard_var,

            app.font_size_var,
            app.font_family_var,
            app.text_dir_var,
            app.text_min_var,
            app.text_max_var,
        ]
        for var in var_map:
            var.trace_add("write", lambda *_: self.update_config_save_settings(app))
