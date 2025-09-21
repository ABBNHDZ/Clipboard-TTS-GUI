"""
Application TTS (XTTS / Coqui TTS) avec interface Tkinter.

Architecture :
- TTSManager   : gestion du cache, du chargement/déchargement du modèle, synthèse.
- AudioPlayer  : lecture des buffers audio (numpy) depuis une queue,
                 pause/reprise/stop via Events ; lecture par petits blocs.
- SynthesisWorker : transforme le texte en wavs et les pousse dans la file audio.
- App (GUI)   : interface (onglets) et coordination.

Auteur    : ABBN
Version   : 0.45
Date      : 18/09/2025
"""
from __future__ import annotations

import os
import time
import threading
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox , font

import pyperclip

from xtts2_m.xttsManager import TTSManager, AudioPlayer, SynthesisWorker 
from xtts2_m.xttsConfig import SUPPORTED_LANGUAGES,DEFAULT_LANGUAGE, DEFAULT_SPEAKERS,DEFAULT_SPEAKER
from appConfig import AppConfig, load_config, register_trace,save_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

AUTHOR = "ABBN <abbndz@gmail.com>"
APPNAME = "Clipboard TTS GUI"
APP_TITLE = "🗣️ Clipboard TTS GUI"
APP_VERSION = "0.45"


class App():
    """ Main application window.
    - Built to be clear, maintainable and close to the previous `clipboard_tts_gui44.py` API.
    - FR/EN comments for important methods.
    """
    def __init__(self,root: tk.Tk):
        super().__init__()
        self.root = root
        self.root.title(f"{APP_TITLE} – v{APP_VERSION}")
        self.root.geometry("580x600")

        # Manager + audio + worker
        self.ttsMan = TTSManager(model_path=os.path.join(os.getcwd(), "model_xtts_v2"))
        self.audio_player = AudioPlayer(app=self)
        self.synth_worker = SynthesisWorker(self.ttsMan, self.audio_player)
        self.synth_worker.app = self 
        self.audio_player.app = self
        self.ttsMan.app = self
        #self._last_index:int =  0
        
        # UI variables
        self.model_path_var = tk.StringVar(value=self.ttsMan.model_path)        
        self.device_var = tk.StringVar(value="auto")
        self.qntf_var = tk.StringVar(value="f8e4")

        self.lang_var = tk.StringVar(value="ar")
        self.lang_rtl_var = tk.StringVar(value="ar")
        self.lang_ltr_var = tk.StringVar(value="fr")
        self.speaker_var = tk.StringVar(value="Tammie Ema")
        self.speaker_wav_var = tk.StringVar(value="")

        self.use_lang_var = tk.BooleanVar(value=True)
        self.use_speaker_var = tk.BooleanVar(value=True)
        self.use_speaker_wav = tk.BooleanVar(value=False)

        self.auto_load = tk.BooleanVar(value=True)
        self.auto_clipboard = tk.BooleanVar(value=True)

        # Advanced params and control flags        
        self.temp = tk.DoubleVar(value=0.75)
        self.lenp = tk.DoubleVar(value=1.0)
        self.repp = tk.DoubleVar(value=5.0)
        self.topk = tk.IntVar(value=50)
        self.topp = tk.DoubleVar(value=0.85)
        self.speed_var = tk.DoubleVar(value=1.0)

        self.use_temp = tk.BooleanVar(value=True)
        self.use_lenp = tk.BooleanVar(value=True)
        self.use_repp = tk.BooleanVar(value=True)
        self.use_topk = tk.BooleanVar(value=True)
        self.use_topp = tk.BooleanVar(value=True)
        self.use_speed = tk.BooleanVar(value=True)
   
        self.font_size_var = tk.IntVar(value=14)   
        self.font_family_var = tk.StringVar(value="Arial") 
        # Direction variable – Auto (default)/ RTL / LTR
        self.text_dir_var = tk.StringVar(value="Auto") 
        self.text_min_var = tk.IntVar(value=80) 
        self.text_max_var = tk.IntVar(value=120) 

        

        self.last_text:str = pyperclip.paste()
        self.cur_text:str = self.last_text

        # Build UI and start background components
        self._build_menu()
        self._build_ui()
        # Start audio & worker threads early so submission works immediately
        self.audio_player.start()
        self.synth_worker.start()

        # **Load settings from JSON** – set all UI variables to the values in the file
        self.appCfg = AppConfig()      # <--- local config instance
        load_config(self, self.appCfg)
        self.ttsMan.appCfg = self.appCfg
        self.synth_worker.appCfg = self.appCfg         

        # **Register a trace on each Tkinter variable** – automatically save changes.
        register_trace(self, self.appCfg)

        # Start clipboard listener and polling
        self._start_clipboard_listener()
        self._poll_status()

        # ---------- Fermeture ----------
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._update_text_font()

        # Auto load if configured
        if self.appCfg.auto_load:
            self.load_model()

    # ---------------- UI construction ----------------
    def _build_ui(self) -> None:
        """Construct the notebook, controls and status bar.
        Build a predictable layout with grouped frames for readability.
        """

        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.tab_play = ttk.Frame(self.notebook)
        self.tab_settings = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_play, text="Lecture")
        self.notebook.add(self.tab_settings, text="Paramètres")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # --- Lecture tab ---
        top_frame = ttk.Frame(self.tab_play)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        ui_text_setting = ttk.Frame(top_frame)
        ui_text_setting.pack(fill=tk.X, padx=4, pady=6) 

        # Font size control
        #ttk.Spinbox( ui_text_setting, from_=8, to=20, width=2, textvariable=self.font_size_var, command=self._update_text_font ).pack(side=tk.LEFT, padx=4)
        # Sélecteur police
        available_fonts = sorted(font.families())
        self.font_family_cb = ttk.Combobox(ui_text_setting, textvariable=self.font_family_var, values=available_fonts, width=12 )
        self.font_family_cb.pack(side=tk.LEFT, padx=4)
        self.font_family_cb.bind("<<ComboboxSelected>>", self._update_text_font)
        self.size_box_cb = ttk.Combobox(ui_text_setting, textvariable=self.font_size_var, values=["8","10","12","14","16","18","20","24","28","32"], width=4 )
        self.size_box_cb.pack(side=tk.LEFT, padx=4)
        self.size_box_cb.bind("<<ComboboxSelected>>", self._update_text_font)

        self.dir_box_cb = ttk.Combobox(ui_text_setting, textvariable=self.text_dir_var, values=["RTL", "LTR", "Auto"], width=5)
        self.dir_box_cb.pack(side=tk.LEFT, padx=4)
        self.dir_box_cb.bind("<<ComboboxSelected>>", self._auto_direction)

        self.lang_cb0 = ttk.Combobox(ui_text_setting, textvariable=self.lang_var, values=SUPPORTED_LANGUAGES, width=4)
        self.lang_cb0.pack(side=tk.LEFT, padx=4)
        self.speaker_cb0 = ttk.Combobox(ui_text_setting, textvariable=self.speaker_var, values=DEFAULT_SPEAKERS, width=14 )
        self.speaker_cb0.pack(side=tk.LEFT, padx=4)        
        ttk.Spinbox(ui_text_setting, from_=0.5, to=2.0, increment=0.05, textvariable=self.speed_var, width=4).pack(side=tk.LEFT, padx=6)

        vscroll = tk.Scrollbar(top_frame, orient="vertical")
        vscroll.pack(side="right", fill="y")

        self.ui_text_box = tk.Text(top_frame, height=10, wrap=tk.WORD,undo=False,yscrollcommand=vscroll.set, font=(self.font_family_var.get(), self.font_size_var.get()) )
        self.ui_text_box.pack(fill=tk.BOTH, expand=True)
        #text_box.pack(side="left", fill="both", expand=True)
        vscroll.config(command=self.ui_text_box.yview)

        ui_text_controls = ttk.Frame(self.tab_play)
        ui_text_controls.pack(fill=tk.X, padx=4, pady=4)        
        ttk.Button(ui_text_controls, text="Lire", width=5,command=self.play_text).pack(side=tk.LEFT, padx=4)  
        self.btn_pause_resume = ttk.Button(ui_text_controls, text="Pause", width=6)
        self.btn_pause_resume.pack(side=tk.LEFT, padx=4)
        self.btn_pause_resume.config(command=self.pause_resume_audio)
        ttk.Button(ui_text_controls, text="Stop", width=5, command=self.stop_all).pack(side=tk.LEFT, padx=4)        
        ttk.Button(ui_text_controls, text="Clear Text", width=9, command=self.clear_text).pack(side=tk.LEFT, padx=4)   
        ttk.Button(ui_text_controls, text="Coller", command=self.paste_text).pack(side=tk.LEFT, padx=4)       

        # Status frame
        status_frame = ttk.Frame(self.tab_play)
        status_frame.pack(fill=tk.X, padx=4, pady=4)
        self.label_status = ttk.Label(status_frame, text="État: prêt")
        self.label_status.pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(status_frame, text="Auto Read Clipboard", variable=self.auto_clipboard).pack(side=tk.RIGHT)
        self.label_text_queue = ttk.Label(status_frame, text="TTS : 0")
        self.label_text_queue.pack(side=tk.RIGHT, padx=4)
        self.label_audio_queue = ttk.Label(status_frame, text="Wave : 0")
        self.label_audio_queue.pack(side=tk.RIGHT, padx=4)
        self.state_indicator = ttk.Label(status_frame, text="●", foreground="green")
        self.state_indicator.pack(side=tk.RIGHT, padx=(8, 0))
        

        # --- Paramètres tab ---
        pfrm = ttk.Frame(self.tab_settings)
        pfrm.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        # Model panel
        model_panel = ttk.LabelFrame(pfrm, text="Modèle / Model")
        model_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=4, pady=4)
        ttk.Checkbutton(model_panel, text="Auot load", variable=self.auto_load, width=12).grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(model_panel, textvariable=self.model_path_var, width=56).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(model_panel, text="...", command=self.browse_model, width=6).grid(row=0, column=2, padx=4, pady=4)

        ttk.Checkbutton(model_panel, text="Speaker Wav", variable=self.use_speaker_wav, width=12).grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Entry(model_panel, textvariable=self.speaker_wav_var, width=56).grid(row=1, column=1, padx=4, pady=4)
        ttk.Button(model_panel, text="...", command=self.browse_speaker_wav, width=6).grid(row=1, column=2, padx=4, pady=4)

        model_Device = ttk.LabelFrame(pfrm, text="Device / Device")
        model_Device.grid(row=1, column=0, sticky=tk.NSEW, padx=4, pady=8)
        
        ttk.Label(model_Device, text="Quantization", width=12).grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(model_Device, textvariable=self.qntf_var, values=("fp32", "fp16", "bf16", "f8e4", "f8e5"), width=12).grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)

        self.btn_load = ttk.Button(model_Device, text="Charger", command=self.load_model)
        self.btn_load.grid(row=0, column=2, sticky=tk.E, padx=4, pady=4)
        
        ttk.Label(model_Device, text="Device", width=12).grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Combobox(model_Device, textvariable=self.device_var, values=("auto", "cpu", "cuda"), width=12).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)

        self.btn_load = ttk.Button(model_Device, text="Décharger", command=self.unload_model)
        self.btn_load.grid(row=1, column=2, sticky=tk.E, padx=4, pady=4)

        #self.progress = ttk.Progressbar(pfrm, mode="determinate", maximum=100)
        self.progress = ttk.Progressbar(model_Device, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=3, sticky=tk.E, padx=4, pady=4)
        

        # Voice panel
        voice_panel = ttk.LabelFrame(pfrm, text="Voix / Voice")
        voice_panel.grid(row=3, column=0, sticky=tk.NSEW, padx=4, pady=8)

        ttk.Checkbutton(voice_panel, text="Language", variable=self.use_lang_var, width=10).grid(row=0, column=0, sticky=tk.W, padx=4, pady=6)
        self.lang_cb1 = ttk.Combobox(voice_panel, textvariable=self.lang_var, values=SUPPORTED_LANGUAGES, width=4)
        self.lang_cb1.grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Checkbutton(voice_panel, text="Speaker", variable=self.use_speaker_var, width=8).grid(row=0, column=2, sticky=tk.W, padx=4, pady=6)
        self.speaker_cb1 = ttk.Combobox(voice_panel, textvariable=self.speaker_var, values=DEFAULT_SPEAKERS, width=16 )
        self.speaker_cb1.grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)        
        ttk.Checkbutton(voice_panel, text="Speed", variable=self.use_speed, width=6).grid(row=0, column=4, sticky=tk.W, padx=4, pady=6)
        ttk.Spinbox(voice_panel, from_=0.1, to=4.0, increment=0.1, textvariable=self.speed_var, width=8).grid(row=0, column=5, sticky=tk.W)

        # Advanced panel
        adv_panel = ttk.LabelFrame(pfrm, text="Paramètres avancés / Advanced")
        adv_panel.grid(row=4, column=0, sticky=tk.NSEW, padx=4, pady=8)        
        ttk.Checkbutton(adv_panel, text="Temperature", variable=self.use_temp).grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=0.01, to=2.0, increment=0.05, textvariable=self.temp, width=8).grid(row=1, column=1, padx=4, pady=4)
        ttk.Checkbutton(adv_panel, text="Length penalty", variable=self.use_lenp).grid(row=2, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=0.01, to=20.0, increment=0.5, textvariable=self.lenp, width=8).grid(row=2, column=1, padx=4, pady=4)
        ttk.Checkbutton(adv_panel, text="Repetition penalty", variable=self.use_repp).grid(row=3, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=0.01, to=20.0, increment=0.5, textvariable=self.repp, width=8).grid(row=3, column=1, padx=4, pady=4)

        ttk.Checkbutton(adv_panel, text="Top-K", variable=self.use_topk).grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=1, to=200, textvariable=self.topk, width=8).grid(row=1, column=3, padx=4, pady=4)
        ttk.Checkbutton(adv_panel, text="Top-P", variable=self.use_topp).grid(row=2, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox(adv_panel, from_=0.01, to=2.0, increment=0.05, textvariable=self.topp, width=8).grid(row=2, column=3, padx=4, pady=4)

        # Advanced Text size segments control
        adv_txt_segm = ttk.LabelFrame(pfrm, text="Advanced Text Setting ")
        adv_txt_segm.grid(row=5, column=0, sticky=tk.NSEW, padx=4, pady=8)
        ttk.Label(adv_txt_segm, text="Text length min ", width=16).grid(row=0, column=0, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox( adv_txt_segm, from_=10, to=200, increment=5, width=3, textvariable=self.text_min_var).grid(row=0, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(adv_txt_segm, text="Text length max", width=16).grid(row=0, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Spinbox( adv_txt_segm, from_=20, to=400, increment=5, width=3, textvariable=self.text_max_var).grid(row=0, column=3, sticky=tk.W, padx=4, pady=4)

        ttk.Label(adv_txt_segm, text="Language RTL ", width=16).grid(row=1, column=0, sticky=tk.W, padx=4, pady=4)
        self.lang_cbRTL = ttk.Combobox(adv_txt_segm, textvariable=self.lang_rtl_var, values=["ar"] , state="readonly", width=4).grid(row=1, column=1, sticky=tk.W, padx=4, pady=4)
        ttk.Label(adv_txt_segm, text="Language LTR ", width=16).grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)
        self.lang_cbLTR = ttk.Combobox(adv_txt_segm, textvariable=self.lang_ltr_var, values=SUPPORTED_LANGUAGES, width=4).grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)

       

    # ---------- Menu ----------
    def _build_menu(self) -> None:
        m = tk.Menu(self.root)  # Create the main menu bar
        mf = tk.Menu(m, tearoff=0) # Create a sub menu for File operations
        mf.add_command(label="Charger le modèle", command=self.load_model)
        mf.add_command(label="Décharger le modèle", command=self.unload_model)
        mf.add_separator()
        mf.add_command(label="Quitter", command=self._on_close)
        m.add_cascade(label="Fichier", menu=mf)

        # Lecture
        ml = tk.Menu(m, tearoff=0)  # Create a sub menu for Lecture operations
        ml.add_command(label="Lire", command=self.play_text)
        ml.add_command(label="Pause", command=lambda: self.pause_audio())
        ml.add_command(label="Reprendre", command=lambda: self.resume_audio())
        ml.add_command(label="Stop", command=lambda: self.stop_all())
        m.add_cascade(label="Lecture", menu=ml)

        # Aide
        ma = tk.Menu(m, tearoff=0) # Create a sub menu for Help operations
        ma.add_command(
            label="À propos",
            command=lambda: messagebox.showinfo(
                "À propos", f"{APP_TITLE}\nVersion {APP_VERSION}"
            ),
        )
        m.add_cascade(label="Aide", menu=ma)

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
        self.label_status.config(text=f"État: {text}")

    def load_model(self) -> None:
        """Start background loading of the model and update the progress bar.
        This delegates to TTSManager.load_async and maps callbacks to the UI.
        """
        model_path = self.model_path_var.get()
        device = self.device_var.get() if hasattr(self, 'device_var') else 'auto'
        self.set_status("Chargement modèle...")
        self.progress['value'] = 0

        def progress_cb(percent: int, message: str) -> None:
            self.progress['value'] = percent
            self.set_status(message)

        def on_done(success: bool) -> None:
            if success:
                self._refresh_lang_speakers()  # refresh after successful load
                logger.info("Chargement terminé , Modèle chargé.")
            else:
                logger.warning("Chargement Erreur, Échec du chargement.")
        threading.Thread(target=self.ttsMan.load_async, args=(model_path, device, progress_cb, on_done), daemon=True).start()

    def unload_model(self) -> None:
        """ unloading of the model and update the progress bar.
        """
        self.ttsMan.unload()
        self.set_status("Déchargement modèle...")
        self.progress['value'] = 0

    def play_text(self) -> None:
        self.restart_workers()
        text = self.ui_text_box.get("1.0", tk.END).strip()
        if not text:
            self.set_status("No text in Text Box")
            return
        self.auto_detect_direction(text)
        self.synth_worker.submit_text(text)
        self.set_status("En file pour synthèse")

    def clear_text(self) -> None:
        self.ui_text_box.delete('1.0', 'end')

    def paste_text(self):
        """Coller texte du presse-papier"""
        try:
            clip = root.clipboard_get()
            self.ui_text_box.insert("insert", clip)
            self.auto_detect_direction(clip)
        except tk.TclError:
            pass

    # -------- Refresh languages & speakers after model load ----------
    def _refresh_lang_speakers(self):
        """Update the language and speaker combobox values after loading a new model."""
        try:
            langs = self.ttsMan.get_supported_languages()
            spks = self.ttsMan.get_supported_speakers()
            # Langues set default & update combo
            try:
                self.lang_cb0["values"] = langs
                self.lang_cb1["values"] = langs
                if self.lang_var.get() not in langs:
                    self.lang_var.set(langs[0] if langs else DEFAULT_LANGUAGE)
            except Exception as e:
                logger.warning("Failed to refresh languages: %s", e)
                pass
            # Speaker set default & update combo
            try:
                self.speaker_cb0["values"] = spks
                self.speaker_cb1["values"] = spks
                if self.speaker_var.get() not in spks:
                    self.speaker_var.set(spks[0] if spks else DEFAULT_LANGUAGE)
            except Exception as e:
                logger.warning("Failed to refresh speakers: %s", e)
                pass
        except Exception as e:
            logger.warning("Failed to refresh languages/speakers: %s", e)
            pass
    
    def _update_text_font(self,*args):
        """Update the font size of `self.text_box` when the spinbox changes."""
        """Mettre à jour police et taille"""
        selected_font = self.font_family_var.get()
        selected_size = int(self.font_size_var.get())
        self.ui_text_box.config(font=(selected_font, selected_size))
    
    def _auto_direction(self, text_dir_var: str = "" , event=None):
        """Détermine la direction automatiquement si Auto est sélectionné."""
        if not text_dir_var:
            text_dir_var = self.text_dir_var.get() 
        if text_dir_var == "Auto":
            text = self.ui_text_box.get("1.0", "end-1c")
            if text.strip():
                first_char = text.strip()[0]
                if self.is_rtl_char(first_char):
                    self._apply_direction("RTL")
                else:
                    self._apply_direction("LTR")
        else:
            self._apply_direction(self.text_dir_var.get())

    def _apply_direction(self, text_dir_var: str=""):
        """Applique LTR ou RTL visuel."""
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
        """Bascule manuelle LTR <-> RTL"""
        if self.ui_text_box.tag_cget("all", "justify") == "right":
            self.ui_text_box.tag_configure("all", justify="left")
        else:
            self.ui_text_box.tag_configure("all", justify="right")
        self.ui_text_box.tag_add("all", "1.0", "end")

    def is_rtl_char(self, ch: str = "A") -> bool:
        """Retourne True si caractère appartient à une plage RTL"""
        code = ord(ch)
        return (
            0x0590 <= code <= 0x08FF   # Arabe + Syriac + ...
            or 0xFB1D <= code <= 0xFEFC # Présentations RTL
        )

    def auto_detect_direction(self,txt:str="", event=None):
        """Analyse auto après saisie/collage"""
        if not txt:
            txt = self.ui_text_box.get("1.0", "end-1c")
        len_txt = len(txt) 
        len_1 = 3
        len_2 = 6
        if len_txt > 40:
            len_1 = 10
            len_2 = 40
        elif len_txt > 20:
            len_1 = 5
            len_2 = 20
        elif len_txt > 6:
            len_1 = 2
            len_2 = 6
        else:
            return
        is_auto_dir = self.text_dir_var.get() == "Auto"    
        # On saute les 3 premiers, on prend les 3 suivants
        sample = txt[len_1:len_2]
        rtl_count = sum(self.is_rtl_char(ch) for ch in sample)
        ltr_count = len(sample) - rtl_count
        if rtl_count*5 >= ltr_count:
            self.ui_text_box.tag_configure("all", justify="right")
            if is_auto_dir:
                if self.lang_var.get() != self.lang_rtl_var.get():
                    self.lang_var.set(self.lang_rtl_var.get()) 
        else:
            self.ui_text_box.tag_configure("all", justify="left")
            if is_auto_dir:
               if self.lang_var.get() != self.lang_ltr_var.get():
                    self.lang_var.set(self.lang_ltr_var.get()) 
        self.ui_text_box.tag_add("all", "1.0", "end")

    def _highlight_text(self, txt: str) -> None:
        """
        Highlights the first occurrence of `txt` in the text_box.
        Removes any previous highlight tags before applying a new one.
        """
        # Remove old highlight
        self.ui_text_box.tag_remove("highlight", "1.0", tk.END)
        # Déterminer point de départ
        if not hasattr(self, "_last_index"):
            self._last_index = "1.0"
        # Rechercher à partir de la dernière occurrence
        start = self.ui_text_box.search(txt, self._last_index, tk.END)
        # Si rien trouvé → repartir du début
        if not start:
            start = self.ui_text_box.search(txt, "1.0", tk.END)
        if start:
            end = f"{start} + {len(txt)}c"
            self.ui_text_box.tag_add("highlight", start, end)
            self.ui_text_box.tag_configure("highlight",
                                        background="yellow",
                                        foreground="black")
            # Mémoriser pour la prochaine recherche (juste après la fin de l’occurrence actuelle)
            self._last_index = end
            # Faire défiler pour que le texte soit visible
            self.ui_text_box.see(f"{end} + {200}c")
            #self.ui_text_box.yview_moveto(
            #    float(self.ui_text_box.index(start).split('.')[0]) / float(self.ui_text_box.index(tk.END).split('.')[0])+0.5)#

    # Clear highlight when playback finishes
    def _remove_highlight(self) -> None:
        self.ui_text_box.tag_remove("highlight", "1.0", tk.END)

    def pause_resume_audio(self) -> None:
        """Toggle between pause and resume using a single button."""
        self.root.after(100, self.restart_workers)
        if self.audio_player.playing and not self.audio_player.paused :
            self.audio_player.pause()
            self.btn_pause_resume.config(text="Reprendre")
            self._update_state_indicator("paused")
        elif self.audio_player.playing and self.audio_player.paused :
            self.audio_player.resume()
            self.btn_pause_resume.config(text="Pause")
            self._update_state_indicator("playing")

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
        self.set_status("Arrêté en cours")
        if self.synth_worker.running :
            self.synth_worker.stop()
            time.sleep(0.1)
            if self.synth_worker.thread:
                # Wait until thread stops (loop in case of timeout)
                while self.synth_worker.thread.is_alive():
                    time.sleep(0.1)
                    i = i + 1 
                    if i > 20: 
                        break
            #try:
            #    self.synth_worker.thread.join(timeout=1.0)
            #except Exception:  # ignore if thread is already dead
            #    pass

        # Stop audio player (drain queue) and wait for it to finish
        i = 0
        if self.audio_player.running:
            self.audio_player.stop(drain=True)
            time.sleep(0.1)
            if self.audio_player.thread:
                while self.audio_player.thread.is_alive():
                    time.sleep(0.1)
                    i = i + 1 
                    if i > 10: 
                        break
            #try:
            #    self.audio_player.thread.join(timeout=1.0)
            #except Exception:
            #    pass
        # Update UI
        self.set_status("Arrêté")
        # Restart workers immediately after the old ones have finished
        self.restart_workers()
        # Restart workers after a short delay to ensure the previous threads are fully gone
        #self.root.after(500, self.restart_workers)

    def _poll_status(self) -> None:
        """Update queue counters in the status bar periodically."""
        tq = 0
        aq = 0
        if self.synth_worker:
            tq = self.synth_worker.text_queue.qsize() 
        if self.audio_player:
            aq = self.audio_player.audio_queue.qsize()
        self.label_text_queue.config(text=f"TTS : {tq}")
        self.label_audio_queue.config(text=f"Wave : {aq}")
        self.root.after(500, self._poll_status)

    def _after_main(self, func):
        try:
            self.root.after(0, func)
        except Exception:
            try:
                func()
            except Exception:
                pass

    def _set_status(self, msg: str) -> None:
        self._after_main(lambda m=msg: self.label_status.config(text=f"État: {m}"))

    def _update_state_indicator(self, state: str = "ready") -> None:
        color = "green" if state == "playing" else ("orange" if state == "paused" else "grey")
        self._after_main(lambda s=state, c=color: (self.label_status.config(text=f"État: {s}"), self.state_indicator.config(foreground=c)))

    def _start_clipboard_listener(self) -> None:
        """Background listener that watches the system clipboard and submits new text.
        If `auto_clipboard` is enabled, newly-copied text will be queued for synthesis.
        """
        def loop():
            #last_text = None
            while True:
                try:
                    if self.auto_clipboard.get():
                        cur:str = pyperclip.paste()
                        if cur and cur != self.last_text and cur.strip():
                            self.last_text = cur
                            if self.audio_player.playing or self.synth_worker.synthesizing :
                                self.ui_text_box.insert('end', "\n\n" + cur)
                            else:
                                self.ui_text_box.delete('1.0', 'end')                                
                                self.ui_text_box.insert('1.0', cur)                                
                            self.auto_detect_direction(cur)
                            #self.restart_workers()
                            self.synth_worker.submit_text(cur)
                            #if self.audio_player.playing or self.synth_worker.synthesizing :
                            #    self._after_main(self.ui_text_box.insert('end', "\n" + cur))
                            #else:
                            #    self._after_main(lambda c=cur: (self.ui_text_box.delete('1.0', 'end'), self.ui_text_box.insert('1.0', c)))
                            self._set_status("Presse-papiers capturé → file de lecture") 
                            #if not getattr(self.audio_player, 'running', False):
                            #    time.sleep(0.05)
                            #    self._after_main(lambda: self.text_box.delete('1.0', 'end'))
                    time.sleep(0.6)
                except Exception as e:
                    logger.warning("Clipboard listener error: %s", e)
                    time.sleep(1.0)

        threading.Thread(target=loop, name="clipboard_listener", daemon=True).start()

    def _on_close(self) -> None:
        try:
            save_settings(self,self.appCfg)
        except Exception:
            pass
        if True: #messagebox.askokcancel("Quitter", "Êtes-vous sûr de vouloir quitter ?"):
            # Arrêt propre des threads et du modèle.            
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
    # Thème simple (optionnel):
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    app = App(root)
    root.mainloop()