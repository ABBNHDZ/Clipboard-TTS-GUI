
# 📌  Clipboard TTS GUI – aTTS  
**aTTS (Clipboard TTS GUI)** est une application autonome qui transforme instantanément le texte copié dans le presse‑papiers en voix synthétisée grâce au modèle **XTTS v2**.  

---

## ⚡️ Pourquoi ce projet ?
- Le modèle **XTTS v2** est le meilleur open‑source disponible pour la langue arabe – le plus performant parmi les alternatives publiques.
- L’application fonctionne *sans* dépendance à l’API Coqui AI, entièrement autonome sur votre machine.
- Les checkpoints quantifiés (fp16, bf16, f8e4, f8e5) se trouvent dans le dossier **`model_xtts_v2`** grâce au script `torch_save_gpt_fp16.py`.

> *العربية*: هذا هو أفضل نموذج مفتوح المصدر للغة العربية، يعمل بكفاءة عالية.  
> *English*: This is the best open‑source model for Arabic, delivering high performance.

---

## 📋 Sommaire

| Section | Description |
|---------|-------------|
| 🔍 Vue d’ensemble | Présentation du projet |
| 🛠️ Prérequis | Environnement, dépendances |
| ⚙️ Installation | Téléchargement & installation |
| 🚀 Utilisation | Lancer l’application, configurer le modèle |
| 🔄 Conversion de checkpoints | Script `torch_save_gpt_fp16.py` |
| 📁 Structure du dépôt | Fichiers clés |
| 📜 Licence | Coqui Public Model License |

---

## 🔍 Vue d’ensemble

- **Clipboard TTS GUI** (`clipboard_tts_gui.py`)  
  Interface graphique Tkinter qui :
  - Lit le texte copié dans le presse‑papiers (optionnel).
  - Sélectionne la langue, l’échantillon de voix et les paramètres avancés.
  - Synthétise via `TTSManager` (dans `xtts2_m/xttsManager.py`), qui charge le modèle XTTS v2 (`model_xtts_v2`) et envoie l’audio à `AudioPlayer`.

- **Model conversion**  
  Le script `torch_save_gpt_fp16.py` convertit les checkpoints du modèle XTTS v2 en formats quantisés (fp16, bf16, f8e4, f8e5) afin de réduire la consommation mémoire.

---

## 🛠️ Prérequis

| Item | Version recommandée |
|------|---------------------|
| Python | ≥ 3.7 |
| NumPy | 1.24+ |
| Tkinter | livré avec Python |
| sounddevice | (pour la lecture audio) |
| pyperclip | 1.8+ (presse‑papiers) |

> **Installation**  
> ```bash
> python -m venv venv
> source venv/bin/activate   # Windows: venv\Scripts\activate
> pip install torch numpy sounddevice pyperclip
> ```

---

## ⚙️ Installation

1. **Clone le dépôt**  
   ```bash
   git clone https://github.com/ABBNHDZ/Clipboard-TTS-GUI.git
   cd aTTS
   ```
2. **Lancer l’application**  
   ```bash
   python clipboard_tts_gui.py
   ```
   L’interface s’affiche, le modèle se charge automatiquement si `auto_load` est activé.

---

## 🚀 Utilisation

### 1️⃣ Interface

- **Lecture** : Sélectionner le texte, cliquer sur « Lire ».
- **Direction** : Auto‑détection ou manuel (RTL/LTR).
- **Paramètres** : Langue, voix, vitesse, temperature, etc.  
  Les comboboxs sont alimentés par la liste des langues/supportées dans `xtts2_m/xttsManager.py`.

### 2️⃣ Clipboard

- **Auto‑lecture** (`auto_clipboard`): Dès que vous copiez du texte, l’application le synchronise et le synthétise.

### 3️⃣ Chargement de modèle

- **Model path** : `model_path_var` (UI) → chemin vers le dossier contenant le checkpoint.
- **Quantisation** : `qntf_var` (fp32, fp16, bf16, f8e4, f8e5).  
  Le script `torch_save_gpt_fp16.py` convertit les checkpoints en ces formats.

---

## 🔄 Conversion de checkpoints

Le fichier `torch_save_gpt_fp16.py` :

- Charge le checkpoint `xttsv2_state_dict.pth`.
- Conserve la structure originale dans un fichier `.safetensors`.
- Produits quantisés (fp16, bf16, f8e4, f8e5) pour chaque partie (`gpt`, `hifigan_decoder`, etc.).
- Permet de réduire l’empreinte mémoire et d’accélérer l’inférence.

> **Exemple**  
> ```bash
> python torch_save_gpt_fp16.py
> ```
> Le script crée plusieurs fichiers : `xttsv2_fp16.safetensors`, `gpt_h_00_02_fp16.safetensors`, etc. Placez ces fichiers dans le même dossier que le checkpoint d’origine.

---

## 📁 Structure du dépôt

| Fichier/Folder | Description |
|----------------|-------------|
| **clipboard_tts_gui.py** | Application Tkinter principale |
| **README.md** | Ce fichier |
| **torch_save_gpt_fp16.py** | Script de conversion de checkpoints |
| **xttsManager.py** | Module `xtts2_m` qui gère le modèle et la synthèse |
| **xtts2_m/** | Sous‑dossier contenant :  
  - `model.py` (class `XTTS`)  
  - `gpt.py` (classe `GPT2InferenceModel`)  
  - autres utils (`tokenizer`, `xttsConfig`, etc.) |

---

## 📜 Licence

Le modèle XTTS v2 est distribué sous la **Coqui Public Model License** (CPML).  
La licence est détaillée dans le fichier `model_xtts_v2/README.md`.  
Veuillez consulter les termes pour toute utilisation commerciale ou redistribution.


---

### 🎉 Happy TTSing! 🎉