"""
Clipboard TTS GUI — lightweight Tkinter app for text-to-speech

This application provides a simple GUI to run XTTS/Coqui TTS locally.
- TTSManager: model load/unload and synthesis entry points.
- AudioPlayer: async playback of numpy audio buffers.
- SynthesisWorker: background worker converting text to audio chunks.
- App (GUI): Tkinter interface, settings, and clipboard monitoring.

Author: ABBN
Version: 0.30
"""
from __future__ import annotations

import os
import ctypes
import sys
import psutil
import time
import threading
import torch
import pyperclip
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font


from xtts2m.xttsManager import TTSManager, AudioPlayer, SynthesisWorker 
from xtts2m.xttsConfig import SUPPORTED_LANGUAGES,DEFAULT_LANGUAGE, DEFAULT_SPEAKERS,DEFAULT_SPEAKER
from appConfig import AppConfig
from xtts2m.utils import list_available_devices

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

AUTHOR = "ABBN <abbndz@gmail.com>"
APPNAME = "Clipboard TTS GUI"
APP_TITLE = "🗣️ Clipboard TTS GUI"
APP_VERSION = "0.30"

# Constants for auto direction detection
SHORT_TEXT_SAMPLE_LENGTH = 6
MEDIUM_TEXT_SAMPLE_LENGTH = 12
LONG_TEXT_SAMPLE_LENGTH = 24
VERY_LONG_TEXT_SAMPLE_LENGTH = 48


class ClipboardTTSApp():
    """ Main application window.
    - Built to be clear, maintainable and close to the previous `clipboard_tts_gui44.py` API.
    - FR/EN comments for important methods.
    """
    def __init__(self,root: tk.Tk, appcfg: AppConfig):
        self.root = root
        self.root.title(f"{APP_TITLE} – v{APP_VERSION}")
        self.root.geometry("580x600")

        self.appCfg = appcfg
        self.appCfg.set_app_Var_from_config(self)

        # Manager + audio + worker
        self.ttsMan = TTSManager(model_path=appcfg.model_path, device = appcfg.device)
        self.ttsMan.app = self
        self.ttsMan.appCfg = self.appCfg

        self.audio_player = AudioPlayer(app=self)
        self.audio_player.app = self

        self.synth_worker = SynthesisWorker(self.ttsMan, self.audio_player)
        self.synth_worker.app = self 
        self.synth_worker.appCfg = self.appCfg 
        
        
         
        self._last_index:str = "1.0"
        self.last_text:str = ""
        try:
            self.last_text = pyperclip.paste()
        except Exception as e:
            logger.warning("Clipboard listener error: %s", e)

        self.cur_text:str = self.last_text

        self.appCfg.set_app_Var_from_config(self)
        # **Register a trace on each Tkinter variable** 
        # – automatically save changes.
        # - update var Tkinter variables Aap -> Config variables
        self.appCfg.register_trace(self)

        # Build UI and start background components
        self._build_menu()
        self._build_ui()
        # Start audio & worker threads early so submission works immediately
        
        self.audio_player.start()
        self.synth_worker.start()                               

        # Start clipboard listener and polling
        self._start_clipboard_listener()
        self._poll_status()

        # ---------- Fermeture ----------
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._update_text_font()

        # Auto load if configured
        if self.appCfg.auto_load:
            self.load_model()

        self._set_console()

        self._stop_clipboard_listener_event = threading.Event()

    # ---------------- UI construction ----------------
    def _build_ui(self) -> None:
        """Construct the notebook, controls and status bar.
        Build a predictable layout with grouped frames for readability.
        """
        # Notebook
        self.notebook = ttk.Notebook(self.root)        
        self.tab_play = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_play, text="Play")
        self.notebook.add(self.tab_settings, text="Settings")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)        

        # --- Lecture tab ---
        top_frame = ttk.Frame(self.tab_play)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        ui_text_setting = ttk.Frame(top_frame)
        ui_text_setting.pack(fill=tk.X, padx=4, pady=6) 

        # Font size control , Sélecteur police
        available_fonts = sorted(font.families())
        self.font_family_cb = ttk.Combobox(ui_text_setting, textvariable=self.font_family_var, values=available_fonts, width=12 )
        self.font_family_cb.pack(side=tk.LEFT, padx=4)
        self.font_family_cb.bind("<<ComboboxSelected>>", self._update_text_font)
        self.size_box_cb = ttk.Combobox(ui_text_setting, textvariable=self.font_size_var, values=["8","10","12","14","16","18","20","24","28","32"], width=4 )
        self.size_box_cb.pack(side=tk.LEFT, padx=4)
        self.size_box_cb.bind("<<ComboboxSelected>>", self._update_text_font)

        self.dir_box_cb = ttk.Combobox(ui_text_setting, textvariable=self.text_dir_var, values=["RTL", "LTR", "Auto"], width=5)
        self.dir_box_cb.pack(side=tk.LEFT, padx=4)
        self.dir_box_cb.bind("<<ComboboxSelected>>", self._auto_direction_cb)

        self.lang_cb0 = ttk.Combobox(ui_text_setting, textvariable=self.lang_var, values=SUPPORTED_LANGUAGES, width=4)
        self.lang_cb0.pack(side=tk.LEFT, padx=4)
        self.speaker_cb0 = ttk.Combobox(ui_text_setting, textvariable=self.speaker_var, values=DEFAULT_SPEAKERS, width=14 )
        self.speaker_cb0.pack(side=tk.LEFT, padx=4)        
        ttk.Spinbox(ui_text_setting, from_=0.5, to=2.0, increment=0.05, textvariable=self.speed_var, width=4).pack(side=tk.LEFT, padx=6)

        vscroll = tk.Scrollbar(top_frame, orient="vertical")
        vscroll.pack(side="right", fill="y")
        
        self.ui_text_box = tk.Text(top_frame, height=10, wrap=tk.WORD,undo=True,yscrollcommand=vscroll.set, font=(self.font_family_var.get(), self.font_size_var.get()) )
        self.ui_text_box.insert("1.0", "Bonjour")  # Set initial text

        self.ui_text_box.pack(fill=tk.BOTH, expand=True)

        # Context menu
        self.context_menu = tk.Menu(self.ui_text_box, tearoff=0)
        self.context_menu.add_command(label="Cut", command=self._cut)
        self.context_menu.add_command(label="Copy", command=self._copy)
        self.context_menu.add_command(label="Paste", command=self._paste)
        self.context_menu.add_command(label="Delete", command=self.clear_text)
        self.context_menu.add_command(label="Undo", command=self._undo)
        self.ui_text_box.bind("<Button-3>", self._show_context_menu)


        vscroll.config(command=self.ui_text_box.yview)

        ui_text_controls = ttk.Frame(self.tab_play)
        ui_text_controls.pack(fill=tk.X, padx=4, pady=4)        
        ttk.Button(ui_text_controls, text="Read", width=5,command=self._play_text).pack(side=tk.LEFT, padx=4)  
        self.btn_pause_resume = ttk.Button(ui_text_controls, text="Pause", width=6)
        self.btn_pause_resume.pack(side=tk.LEFT, padx=4)
        self.btn_pause_resume.config(command=self.pause_resume_audio)
        ttk.Button(ui_text_controls, text="Stop", width=5, command=self.stop_all).pack(side=tk.LEFT, padx=4)        
        ttk.Button(ui_text_controls, text="Clear Text", width=9, command=self.clear_text).pack(side=tk.LEFT, padx=4)   
        ttk.Button(ui_text_controls, text="Coller", command=self.paste_text).pack(side=tk.LEFT, padx=4)       

        # Status frame
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.label_status = ttk.Label(status_frame, text="Status: Ready")
        self.label_status.pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(status_frame, text="Auto", variable=self.auto_clipboard_var).pack(side=tk.RIGHT)

        # Ajout des compteurs
        self.label_text_queue = ttk.Label(status_frame, text="TTS : 0")
        self.label_text_queue.pack(side=tk.RIGHT, padx=4)
        self.label_audio_queue = ttk.Label(status_frame, text="Wav : 0")
        self.label_audio_queue.pack(side=tk.RIGHT, padx=4)

        self.progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self.progress.pack(side=tk.RIGHT, padx=(4, 0), fill=tk.X, expand=True)
        self.progress.pack_forget()  # <-- CACHÉ au départ

        self.label_memory_ram = ttk.Label(status_frame, text="RAM: 0 MB")
        self.label_memory_ram.pack(side=tk.RIGHT, padx=4)

        self.label_memory_gpu = ttk.Label(status_frame, text="GPU: 0/0 MB")
        self.label_memory_gpu.pack(side=tk.RIGHT, padx=4)

        # --- Paramètres tab ---
        pfrm = ttk.Frame(self.tab_settings)
        pfrm.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        # Model panel
        model_panel = ttk.LabelFrame(pfrm, text="Model")
        model_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=4, pady=4)
        ttk.Checkbutton(model_panel, text="Auto load", variable=self.auto_load_var, width=12).grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(model_panel, textvariable=self.model_path_var, width=56).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(model_panel, text="...", command=self.browse_model, width=6).grid(row=0, column=2, padx=4, pady=4)

        ttk.Checkbutton(model_panel, text="Speaker Wav", variable=self.u_speaker_wav_var, width=12).grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(model_panel, textvariable=self.speaker_wav_var, width=56).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(model_panel, text="...", command=self.browse_speaker_wav, width=6).grid(row=1, column=2, padx=4, pady=4)

        model_device = ttk.LabelFrame(pfrm, text="Device")
        model_device.grid(row=1, column=0, sticky=tk.NSEW, padx=4, pady=8)
        
        ttk.Label(model_device, text="Précision", width=12).grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(model_device, textvariable=self.sdtype_var, values=("float32", "float16"), width=12).grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)

        self.btn_load = ttk.Button(model_device, text="Load", command=self.load_model)
        self.btn_load.grid(row=0, column=2, sticky=tk.E, padx=4, pady=4)
        dvices = ("AUTO", "CPU","iGPU0","iGPU1", "CUDA")
        dvices = list_available_devices()        
        ttk.Label(model_device, text="Device", width=12).grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(model_device, textvariable=self.device_var, values=dvices, width=12).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)

        self.btn_unload = ttk.Button(model_device, text="Unload", command=self.unload_model)
        self.btn_unload.grid(row=1, column=2, sticky=tk.E, padx=4, pady=4)     

        # Voice panel
        voice_panel = ttk.LabelFrame(pfrm, text="Voice")
        voice_panel.grid(row=3, column=0, sticky=tk.NSEW, padx=4, pady=8)

        ttk.Checkbutton(voice_panel, text="Language", variable=self.u_lang_var, width=10).grid(row=0, column=0, sticky=tk.W, padx=4, pady=6)
        self.lang_cb1 = ttk.Combobox(voice_panel, textvariable=self.lang_var, values=SUPPORTED_LANGUAGES, width=4)
        self.lang_cb1.grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Checkbutton(voice_panel, text="Speaker", variable=self.u_speaker_var, width=8).grid(row=0, column=2, sticky=tk.W, padx=4, pady=6)
        self.speaker_cb1 = ttk.Combobox(voice_panel, textvariable=self.speaker_var, values=DEFAULT_SPEAKERS, width=16 )
        self.speaker_cb1.grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)        
        ttk.Checkbutton(voice_panel, text="Speed", variable=self.u_speed_var, width=6).grid(row=0, column=4, sticky=tk.W, padx=4, pady=6)
        ttk.Spinbox(voice_panel, from_=0.1, to=4.0, increment=0.1, textvariable=self.speed_var, width=8).grid(row=0, column=5, sticky=tk.W)

        # Advanced panel
        adv_panel = ttk.LabelFrame(pfrm, text="Advanced Settings")
        adv_panel.grid(row=4, column=0, sticky=tk.NSEW, padx=4, pady=8)        
        
        ttk.Checkbutton(adv_panel, text="Temperature", variable=self.u_temp_var).grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=0.01, to=2.0, increment=0.05, textvariable=self.temp_var, width=8).grid(row=1, column=1, padx=4, pady=4)

        ttk.Checkbutton(adv_panel, text="Repetition penalty", variable=self.u_repp_var).grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=0.01, to=20.0, increment=0.5, textvariable=self.repp_var, width=8).grid(row=2, column=1, padx=4, pady=4)

        ttk.Checkbutton(adv_panel, text="Top-K", variable=self.u_topk_var).grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=1, to=200, textvariable=self.topk_var, width=8).grid(row=1, column=3, padx=4, pady=4)
        ttk.Checkbutton(adv_panel, text="Top-P", variable=self.u_topp_var).grid(row=2, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=0.01, to=2.0, increment=0.05, textvariable=self.topp_var, width=8).grid(row=2, column=3, padx=4, pady=4)

        ttk.Checkbutton(adv_panel, text="Seed", variable=self.u_seed_var).grid(row=1, column=4, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=-1, to=20000000,increment=5, textvariable=self.seed_var, width=8).grid(row=1, column=5, padx=4, pady=4)

        # Advanced Text Setting 
        adv_txt_settg = ttk.LabelFrame(pfrm, text="Advanced Text Setting ")
        adv_txt_settg.grid(row=5, column=0, sticky=tk.NSEW, padx=4, pady=8)
        ttk.Label(adv_txt_settg, text="Text length min ", width=16).grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox( adv_txt_settg, from_=10, to=200, increment=5, width=3, textvariable=self.text_min_var).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(adv_txt_settg, text="Text length max", width=16).grid(row=0, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox( adv_txt_settg, from_=20, to=400, increment=5, width=3, textvariable=self.text_max_var).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(adv_txt_settg, text="Language RTL ", width=16).grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        self.lang_cbRTL = ttk.Combobox(adv_txt_settg, textvariable=self.lang_rtl_var, values=["ar"] , state="readonly", width=4).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(adv_txt_settg, text="Language LTR ", width=16).grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)
        self.lang_cbLTR = ttk.Combobox(adv_txt_settg, textvariable=self.lang_ltr_var, values=SUPPORTED_LANGUAGES, width=4).grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Checkbutton(adv_txt_settg, text="Auto Read Clipboard", variable=self.auto_clipboard_var, width=20).grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)

       

    # ---------- Menu ----------
    def _build_menu(self) -> None:
        # Create the main menu bar
        m = tk.Menu(self.root)  
        # Create a sub menu for File operations
        mf = tk.Menu(m, tearoff=0) 
        mf.add_command(label="Load Model", command=self.load_model)
        mf.add_command(label="Unload Model", command=self.unload_model)
        mf.add_separator()
        mf.add_command(label="Exit", command=self._on_close)
        m.add_cascade(label="File", menu=mf)

        # Create a sub menu for Lecture operations
        ml = tk.Menu(m, tearoff=0) 
        ml.add_command(label="Read", command=self._play_text)
        ml.add_command(label="Pause", command=lambda: self.pause_audio())
        ml.add_command(label="Resume", command=lambda: self.resume_audio())
        ml.add_command(label="Stop", command=lambda: self.stop_all())
        m.add_cascade(label="Playback", menu=ml)

        # View menu - added theme selection and console control
        mv = tk.Menu(m, tearoff=0)
        mv.add_checkbutton(label="Toggle Console", command=self._toggle_console,variable=self.console_var)
        mv.add_separator()
        mv.add_radiobutton(label="Clam Theme", command=lambda: self._set_theme("clam"), 
                          variable=self.theme_var, value="clam")
        mv.add_radiobutton(label="Alt Theme", command=lambda: self._set_theme("alt"),
                          variable=self.theme_var, value="alt")
        mv.add_radiobutton(label="Default Theme", command=lambda: self._set_theme("default"),
                          variable=self.theme_var, value="default")
        mv.add_radiobutton(label="Classic Theme", command=lambda: self._set_theme("classic"),
                          variable=self.theme_var, value="classic")
        m.add_cascade(label="View", menu=mv)

        # Aide
        ma = tk.Menu(m, tearoff=0) # Create a sub menu for Help operations
        ma.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About", f"{APP_TITLE}\nVersion {APP_VERSION}"
            ),
        )
        m.add_cascade(label="Help", menu=ma)

        # Add the menu to the main window
        self.root.config(menu=m)

    # ---------------- Actions / helpers ----------------
    def browse_model(self) -> None:
        path = filedialog.askdirectory(title="Select model path", initialdir=self.model_path_var.get())
        if path:
            self.model_path_var.set(path)

    def browse_speaker_wav(self) -> None:
        path = filedialog.askopenfilename(title="Select speaker wav", filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if path:
            self.speaker_wav_var.set(path)

    def set_status(self, text: str) -> None:
        self.label_status.config(text=f"Status: {text}")

    def load_model(self) -> None:
        """Start background loading of the model and update the progress bar.
        This delegates to TTSManager.load_async and maps callbacks to the UI.
        """
        model_path = self.model_path_var.get()
        device = self.device_var.get() 
        self.set_status("Loading model...")
        self.progress['value'] = 0
        self._loading = True 
        # Afficher le progress bar
        self.progress.pack(side=tk.RIGHT, padx=(4, 0), fill=tk.X, expand=True)

        def progress_cb(percent: int, message: str) -> None:
            self.progress['value'] = percent
            self.set_status(message)

        def on_done(success: bool) -> None:
            self._loading = False
            if success:
                self._refresh_lang_speakers() 
                logger.info("Loading complete, Model loaded.")
            else:
                logger.warning("Loading Error, Loading failed.")
            try:
                self.progress.pack_forget()
            except Exception:
                pass

        threading.Thread(target=self.ttsMan.load_async, args=(model_path, device, progress_cb, on_done), daemon=True).start()

    def unload_model(self) -> None:
        """ unloading of the model and update the progress bar.
        """
        self.set_status("Unloading model...")
        self.ttsMan.unload()        

    def _play_text(self) -> None:
        self.restart_workers()
        text = self.ui_text_box.get("1.0", tk.END).strip()
        if not text:
            self.set_status("No text in Text Box")
            return
        self.auto_detect_direction(text)
        self.synth_worker.submit_text(text)
        self.set_status("Queued for synthesis")

    def clear_text(self) -> None:
        self.ui_text_box.delete('1.0', 'end')

    def paste_text(self):
        """Paste text from clipboard."""
        try:
            clip = self.root.clipboard_get()
            self.ui_text_box.insert("insert", clip)
            self.auto_detect_direction(clip)
        except tk.TclError as e:
            logger.warning("Paste text error: %s", e)

    # -------- Refresh languages & speakers after model load ----------
    def _refresh_lang_speakers(self):
        """Update the language and speaker combobox values after loading a new model."""
        try:
            langs = self.ttsMan.get_supported_languages()
            spks = self.ttsMan.get_supported_speakers()
            # Langues set default & update combo
            try:
                # MainMenu (tab)
                self.lang_cb0["values"] = langs 
                # SettingMenu (tab)
                self.lang_cb1["values"] = langs 
                if self.lang_var.get() not in langs:
                    self.lang_var.set(langs[0] if langs else DEFAULT_LANGUAGE)
            except Exception as e:
                logger.warning("Failed to refresh languages: %s", e)

            # Speaker set default & update combo
            try:
                # MainMenu (Tab)
                self.speaker_cb0["values"] = spks
                # SettingMenu (Tab)
                self.speaker_cb1["values"] = spks
                if self.speaker_var.get() not in spks:
                    self.speaker_var.set(spks[0] if spks else DEFAULT_SPEAKER)
            except Exception as e:
                logger.warning("Failed to refresh speakers: %s", e)

        except Exception as e:
            logger.warning("Failed to refresh languages/speakers: %s", e)
    
    def _update_text_font(self,*args):
        """Update the font size of `self.text_box` when the spinbox changes."""
        selected_font = self.font_family_var.get()
        selected_size = int(self.font_size_var.get())
        self.ui_text_box.config(font=(selected_font, selected_size))
    
    def _auto_direction_cb(self, text_dir_var: str = ""):
        """Determine direction automatically if Auto is selected."""
        if not text_dir_var:
            text_dir_var = self.text_dir_var.get() 
        if text_dir_var == "Auto":
            self.auto_detect_direction()            
        else:
            self._apply_direction_cb(text_dir_var)

    def _apply_direction_cb(self, text_dir_var: str=""):
        """Apply LTR or RTL visual."""
        if not text_dir_var:
            text_dir_var = self.text_dir_var.get() 
        if text_dir_var == "LTR":
            self.ui_text_box.tag_configure("all", justify="left")
            self.ui_text_box.tag_add("all", "1.0", "end")
        elif text_dir_var == "RTL":
            self.ui_text_box.tag_configure("all", justify="right")
            self.ui_text_box.tag_add("all", "1.0", "end")
        else:
            self.auto_detect_direction()
   
    def toggle_direction(self):
        """Toggle manual LTR <-> RTL"""
        if self.ui_text_box.tag_cget("all", "justify") == "right":
            self.ui_text_box.tag_configure("all", justify="left")
        else:
            self.ui_text_box.tag_configure("all", justify="right")
        self.ui_text_box.tag_add("all", "1.0", "end")

    def _is_rtl_char(self, ch: str = "A") -> bool:
        """Return True if character belongs to a RTL range."""        
        code = ord(ch)
        return (
            0x0590 <= code <= 0x08FF   # Arabe + Syriac + ...
            or 0xFB1D <= code <= 0xFEFC # Présentations RTL
        )

    def auto_detect_direction(self,text:str=""):
        """Auto analyze after input/paste"""
        text_dir_var = self.text_dir_var.get()
        if text_dir_var != "Auto" :
            return
        if not text:
            text = self.ui_text_box.get("1.0", "end-1c")
        text_length = len(text)
        if text_length > VERY_LONG_TEXT_SAMPLE_LENGTH:
            sample_start = LONG_TEXT_SAMPLE_LENGTH
            sample_end = VERY_LONG_TEXT_SAMPLE_LENGTH
        elif text_length > MEDIUM_TEXT_SAMPLE_LENGTH:
            sample_start = 6
            sample_end = MEDIUM_TEXT_SAMPLE_LENGTH
        elif text_length > SHORT_TEXT_SAMPLE_LENGTH:
            sample_start = 3
            sample_end = SHORT_TEXT_SAMPLE_LENGTH
        else:
            return           
        # On saute les len_1 premiers, on prend les len_2-len_1 suivants
        sample = text[sample_start:sample_end]
        rtl_count = sum(self._is_rtl_char(char) for char in sample)
        ltr_count = len(sample) - rtl_count
        if rtl_count * 4 >= ltr_count:
            self.ui_text_box.tag_configure("all", justify="right")
            if self.lang_var.get() != self.lang_rtl_var.get():
                self.lang_var.set(self.lang_rtl_var.get()) 
        else:
            self.ui_text_box.tag_configure("all", justify="left")
            if self.lang_var.get() != self.lang_ltr_var.get():
                self.lang_var.set(self.lang_ltr_var.get()) 
        self.ui_text_box.tag_add("all", "1.0", "end")

    def _highlight_text(self, text: str) -> None:
        """Highlights the first occurrence of `text` in the text_box.
        Removes any previous highlight tags before applying a new one.
        """
        # Remove old highlight
        self.ui_text_box.tag_remove("highlight", "1.0", tk.END)
        # Déterminer point de départ
        start = self.ui_text_box.search(text, self._last_index, tk.END)
        # Si rien trouvé → repartir du début
        if not start:
            start = self.ui_text_box.search(text, "1.0", tk.END)
            self._last_index = "1.0"  # Réinitialiser au début
        if start:
            end = f"{start} + {len(text)}c"
            self.ui_text_box.tag_add("highlight", start, end)
            self.ui_text_box.tag_configure("highlight", background="yellow", foreground="black")
            # Mémoriser pour la prochaine recherche (juste après la fin de l’occurrence actuelle)
            self._last_index = end
            # Faire défiler pour que le texte soit visible
            self.ui_text_box.see(f"{end} + {200}c")

    # Clear highlight when playback finishes
    def remove_highlight(self) :
        self.ui_text_box.tag_remove("highlight", "1.0", tk.END)

    def pause_resume_audio(self) :
        """Toggle between pause and resume using a single button."""
        self.root.after(100, self.restart_workers)
        if self.audio_player.playing and not self.audio_player.paused :
            self.audio_player.pause()
            self.btn_pause_resume.config(text="Resume")
        elif self.audio_player.playing and self.audio_player.paused :
            self.audio_player.resume()
            self.btn_pause_resume.config(text="Pause")

    def pause_audio(self) -> None:
        self.audio_player.pause()

    def resume_audio(self) -> None:
        self.audio_player.resume()

    def restart_workers(self) -> None:
        """Relance la synthèse et le playback si ils étaient arrêtés."""
        if not self.synth_worker.running:
            self.synth_worker.start()
        if not self.audio_player.running:
            self.audio_player.start()

    def stop_all(self) -> None:
        """Stop all workers, then restart them once the threads have finished."""
        # Stop synth worker and wait for it to terminate
        i = 0
        self.set_status("Stop audio player")
        # Stop audio player (drain queue) and wait for it to finish
        if self.audio_player.running:
            self.audio_player.stop(drain=True)
        self.set_status("Stop synth worker ")
        if self.synth_worker.running :
            self.synth_worker.stop()
            time.sleep(0.1)
            if self.synth_worker.thread:
                # Wait until thread stops (loop in case of timeout)
                while self.synth_worker.thread.is_alive():
                    time.sleep(0.2)
                    i = i + 1 
                    if i > 20: 
                        break

        # Audio player wait for it to finish
        i = 0        
        time.sleep(0.1)
        if self.audio_player.thread:
            while self.audio_player.thread.is_alive():
                time.sleep(0.1)
                i = i + 1 
                if i > 10: 
                    break
        # Update UI
        self.set_status("Stopped")
        # Restart workers immediately after the old ones have finished
        self.restart_workers()

    def _poll_status(self) -> None:
        """Update queue counters in the status bar periodically."""
        tq = 0
        aq = 0
        if self.synth_worker:
            tq = self.synth_worker.text_queue.qsize() 
        if self.audio_player:
            aq = self.audio_player.audio_queue.qsize()

        self.label_text_queue.config(text=f"TTS : {tq}")
        self.label_audio_queue.config(text=f"Wav : {aq}")

        # Mémoire RAM (process)
        process = psutil.Process(os.getpid())
        ram_mb = f"{process.memory_info().rss / 1e6:.0f}"

        # Mémoire GPU (si disponible)
        gpu_alloc = "N/A"
        gpu_reserved = "N/A"
        #max_gpu_alloc = "N/A"
        if self.ttsMan.device.startswith("privateuseone"):
            try:
                import torch_directml
                device_id = 0
                if self.ttsMan.device == "privateuseone:1":
                    device_id = 1
                dml_device = torch_directml.device(device_id)
                alloc_mb = torch_directml.gpu_memory(device_id=device_id, mb_per_tile=1)
                # {{ Modification: Sum the tile allocations and format }}
                total_alloc_mb = sum(alloc_mb)
                gpu_alloc = f"{total_alloc_mb:.0f}"  # Display with 0 decimal places
                gpu_reserved = "N/A"  # Reserved memory not available

            except Exception as e:
                gpu_alloc = f"DML Error: {e}"
                gpu_reserved = f"DML Error"
        elif torch.cuda.is_available():
            alloc_mb = int(torch.cuda.memory_allocated() / 1024**2)
            reserved_mb = int(torch.cuda.memory_reserved() / 1024**2)
            max_alloc_mb = int(torch.cuda.max_memory_allocated() / 1024**2)
            gpu_alloc = f"{alloc_mb}"
            gpu_reserved = f"{reserved_mb}"
            #max_gpu_alloc = f"{max_alloc_gb}"
        else:
            gpu_alloc = "N/A"

        self.label_memory_ram.config(text=f"RAM: {ram_mb} MB")
        self.label_memory_gpu.config(text=f"GPU: {gpu_alloc} MB")

        # Mise à jour du progress bar (si chargement en cours)
        if hasattr(self, '_loading') and self._loading:
            self.progress['value'] = 0
        else:
            self.progress['value'] = 100

        self.root.after(1000, self._poll_status)      

    def _after_main(self, func):
        try:
            self.root.after(0, func)
        except Exception:
            try:
                func()
            except Exception:
                pass

    def _set_status(self, msg: str) -> None:
        self._after_main(lambda m=msg: self.label_status.config(text=f"Status: {m}"))

    def _start_clipboard_listener(self) -> None:
        """Background listener that watches the system clipboard and submits new text.
        If `auto_clipboard` is enabled, newly-copied text will be queued for synthesis.
        """
        def loop():
            while True:
                try:
                    if self.auto_clipboard_var.get():
                        cur:str = pyperclip.paste()
                        if cur and cur != self.last_text and cur.strip():
                            self.last_text = cur
                            if self.audio_player.playing or self.synth_worker.synthesizing :
                                self.ui_text_box.insert('end', "\n" + cur)
                            else:
                                self.ui_text_box.delete('1.0', 'end')                                
                                self.ui_text_box.insert('1.0', cur)                                
                            self.auto_detect_direction(cur)
                            self.synth_worker.submit_text(cur)                            
                            self._set_status("Clipboard captured → queued for playback")                             
                    
                except Exception as e:
                    logger.warning("Clipboard listener error: %s", e)                    
                time.sleep(1.0)

        threading.Thread(target=loop, name="clipboard_listener", daemon=True).start()
    
    def _hide_console(self):
        """Hide the console window (Windows only)."""
        if sys.platform == "win32":
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            ctypes.windll.user32.ShowWindow(hwnd, 0)  
            # 0 = SW_HIDE
            self.appCfg.console = False
            self.console_var.set(self.appCfg.console)            

    def _show_console(self):
        """Show the console window (Windows only)."""
        if sys.platform == "win32":
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            ctypes.windll.user32.ShowWindow(hwnd, 5)  
            # 5 = SW_SHOW
            self.appCfg.console = True
            self.console_var.set(self.appCfg.console)
    
    def _set_console(self):
        """Toggle console window visibility."""
        if not self.appCfg.console:
            self._hide_console()
        else:
            self._show_console()

    def _toggle_console(self):
        """Toggle console window visibility."""
        #self.appCfg.console = not self.appCfg.console se fait par clic UI
        self.console_var.set(self.appCfg.console )
        self._set_console()

    def _set_theme(self, theme_name):
        """Change the application's visual theme."""
        try:
            style = ttk.Style()
            style.theme_use(theme_name)
            self.set_status(f"Theme changed to {theme_name}")
        except Exception as e:
            self.set_status(f"Error changing theme: {str(e)}")


    def _cut(self):
        try:
            self.ui_text_box.event_generate("<<Cut>>")
        except tk.TclError:
            pass

    def _copy(self):
        try:
            self.ui_text_box.event_generate("<<Copy>>")
        except tk.TclError:
            pass

    def _paste(self):
        try:
            self.ui_text_box.event_generate("<<Paste>>")
        except tk.TclError:
            pass

    def _undo(self):
        try:
            self.ui_text_box.event_generate("<<Undo>>")
        except tk.TclError:
            pass

    def _show_context_menu(self, event):
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()


    def _on_close(self) -> None:
        try:
            self.appCfg.update_config_save_settings(self)
        except Exception:
            pass         
        try:
            self.synth_worker.stop()
        except Exception:
            pass
        try:
            self.audio_player.stop()
        except Exception:
            pass
        try:
            self.ttsMan.unload()
        except Exception:
            pass
        self.root.destroy()

# Entrée programme
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    # **Load settings from JSON** – set all UI variables to the values in the file
    appCfg = AppConfig()      
    appCfg.load_from_json()

    # Thème simple (optionnel):
    try:
        style = ttk.Style()
        style.theme_use(appCfg.theme)
    except Exception:
        pass

    app = ClipboardTTSApp(root, appCfg)
    root.mainloop()