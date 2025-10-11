"""
- Les fonctions de nettoyage / expansion (nombres, abréviations, symboles) sont
  localisées ici et commentées en français.
- La logique principale du tokenizer est inchangée : on utilise un fichier de
  vocabulaire via `tokenizers.Tokenizer` si fourni.
"""

import logging
import re 
from typing import List
from functools import cached_property

import torch
from num2words import num2words
from tokenizers import Tokenizer
from typing import Optional


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers — fonctions utilitaires
# -----------------------------------------------------------------------------

def collapse_whitespace(text: str) -> str:
    """Remplace les séquences d'espaces par un seul espace."""
    return re.sub(r"\s+", " ", text)


def lowercase(text: str) -> str:
    """Convertit le texte en minuscules (simple utilitaire)."""
    return text.lower()


class Doc_Split_text:
    """Classe utilitaire pour Split_text 
    split the raw text into *segments*.
    """
    def __init__(self, 
                segments: List[str] = [],
                last_segm: str = "",
                prev_segm: str = "",
                min_len: int = 60, #text_length_min
                max_len: int = 180,#text_length_max
                ):
        # Vars 
        self.segments: List[str] = segments
        self.last_segm: str = last_segm
        self.prev_segm: str = prev_segm
        self.last_segm_len: int = 0
        self.prev_segm_len: int = 0
        self.min_len: int = min_len # text_length_min
        if self.min_len < 5:
            self.min_len = 5
        if self.min_len > 200:
            self.min_len = 200
        self.max_len: int = max_len # text_length_max
        if self.max_len < 15:
            self.max_len = 15
        if self.max_len > 400:
            self.max_len = 400
        if self.min_len > self.max_len -10:
            self.min_len = self.max_len -10
        self.last_sep: str | None = None


    def doc_add_segm(self,min_len:int=0,max_len:int=0 ) :
        if self.last_segm:
            if self.last_sep:
                self.last_segm += self.last_sep
                self.last_sep = None
            if self.prev_segm:
                 # Check lengths and merge 
                self.last_segm_len = len(self.last_segm)
                self.prev_segm_len = len(self.prev_segm)
                # Merge short phrases only if both are short or one is short,
                # and the total length <= max, AND there was exactly one separator between them.
                # (only valid when previous segment ended with separator)
                if ( (self.last_segm_len < self.min_len) or (self.prev_segm_len < self.min_len) ) and  ( (self.last_segm_len + self.prev_segm_len) < self.max_len ):
                    self.prev_segm = self.prev_segm + self.last_segm
                    self.last_segm = ""
                    if ((self.last_segm_len + self.prev_segm_len) > min_len ):
                        self.segments.append(self.prev_segm)
                        self.prev_segm = ""
                else: # ( (last_segm_len > min_len) or  (prev_segm_len > min_len) ) or ( (last_segm_len + prev_segm_len) > max_len )
                    # Cannot merge: flush both segments
                    self.segments.append(self.prev_segm)
                    self.prev_segm = ""
                    if (self.last_segm_len > min_len ):
                        self.segments.append(self.last_segm)
                        self.last_segm = ""
                    else:
                        self.prev_segm = self.last_segm 
                        self.last_segm = ""
            else: # prev_segm = ""
                self.last_segm_len = len(self.last_segm)
                if (self.last_segm_len > min_len ):
                    self.segments.append(self.last_segm)                    
                else:
                    self.prev_segm = self.last_segm 
                self.last_segm = ""

        elif self.prev_segm: #last_segm = ""
            if (len(self.prev_segm) > min_len ):
                self.segments.append(self.prev_segm)
                self.prev_segm = ""


def split_text(
    text: str,
    text_length_min: int = 40,
    text_length_max: int = 80
) -> List[str]:
    """
    Split a string into phrases following rules:
    • Latin & Arabic punctuation, plus tabulation ('\t') are separators.
    • Space is NOT considered a separator (except for long phrase splitting).
    • Newlines (`\n`, `\r`) break the group; they are not part of any segment.
    • Consecutive separators are ignored and act as a break.
    • Short phrases (<`text_length_min`) can be joined with the following
      phrase **only if there is exactly one separator** between them,
      during segmentation (first phase).
    • If a segment exceeds `text_length_max`, it is split by space after
      `text_length_min` until remaining part <= `text_length_max`.
    • The internal structure of each segment stays identical to original text.

    Parameters
    ----------
    text : str
        Input string.
    text_length_min : int, default 40
        Minimum length for a short phrase to be eligible for grouping.
    text_length_max : int, default 80
        Maximum allowed length of an output segment.

    Returns
    -------
    List[str]
        A list of phrases (segments).
    """
    # ------------------------------------------------------------------
    # Define separators (punctuation + other symbols)
    # ------------------------------------------------------------
    latin_sep = {".", ",", ":", ";", "!", "?","#","*","[","]"
                 "«", "»", "(", ")", "-", "…", "{","}" '\t'}

    arabic_sep = {
        chr(0x061F),  # Arabic question mark
        chr(0x060C),  # Arabic comma
        chr(0x061B),  # Arabic semicolon
        chr(0x061C),  # Arabic colon
        chr(0x06D4),  # Arabic full stop (rare)
    }

    separators = latin_sep.union(arabic_sep)

    # ------------------------------------------------------------------
    # Step 1 – split the raw text into *segments*.
    # ------------------------------------------------------------
    d : Doc_Split_text = Doc_Split_text([],"","",text_length_min,text_length_max)
    for ch in text:
        # Newline or carriage return – break grouping
        if ch in ("\n", "\r"):
            d.doc_add_segm(0,d.max_len)         
            d.last_segm = ""
            d.prev_segm = ""
            d.last_sep = None
            continue

        # Separator handling (including tabulation, space NOT included)
        if ch in separators:
            if d.last_sep is not None:      # consecutive separator → break grouping                       
                d.doc_add_segm(0,d.max_len)
                d.last_sep = ch
            else:                           # first separator after text                         
                d.last_sep = ch
                d.doc_add_segm(d.min_len,d.max_len)
                
        else:  # ordinary character
            if d.last_sep is not None:
                d.last_segm += d.last_sep
                d.last_segm += ch
                d.last_sep = None
            else:
                d.last_segm += ch
    
    d.doc_add_segm(0,d.max_len)

    # Step 2 – Split any segment that is > text_length_max by spaces.
    result: List[str] = []
    buf: str = ""
    s: str = ""
    buf_len: int = 0
    ss_len:int = 0
    s_len:int = 0
    for s in d.segments:
        buf = s
        buf_len = len(buf)
        if buf_len < text_length_max:
            result.append(buf)
            continue
        else:
            buf = ""
            i:int = 0
            ss_len = len(s)
            s_len = ss_len
            while (i <  ss_len) and (s[i] ==" "):
                i += 1
            while i <  ss_len:
                ch = s[i]
                buf += ch
                i += 1                
                buf_len = len(buf)
                if (ch in (" ", "\t")) and (buf_len > d.min_len):
                    result.append(buf)
                    buf = ""
                    s_len = s_len - buf_len
                    if s_len < d.max_len:
                        while (i <  ss_len) and (s[i] ==" "):
                            i += 1
                        while i <  ss_len:                            
                            buf += s[i]
                            i += 1
                        result.append(buf)
                        buf = ""
                        break
            if buf: # not compacted
                result.append(buf)

    return result



# -----------------------------------------------------------------------------
# Dictionnaires et regex pour la normalisation multilingue
# -----------------------------------------------------------------------------

# Abréviations communes par langue -> correspondance en texte complet
_abbreviations = {
    "en": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [
            ("mrs", "misess"),
            ("mr", "mister"),
            ("dr", "doctor"),
            ("st", "saint"),
            ("co", "company"),
            ("jr", "junior"),
            ("maj", "major"),
            ("gen", "general"),
            ("drs", "doctors"),
            ("rev", "reverend"),
            ("lt", "lieutenant"),
            ("hon", "honorable"),
            ("sgt", "sergeant"),
            ("capt", "captain"),
            ("esq", "esquire"),
            ("ltd", "limited"),
            ("col", "colonel"),
            ("ft", "fort"),
        ]
    ],
    "fr": [
        (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
        for x in [("mme", "madame"), ("mr", "monsieur"), ("dr", "docteur"), ("st", "saint"), ("co", "compagnie"), ("jr", "junior"), ("ltd", "limitée")]
    ],
    # d'autres langues peuvent être ajoutées ici
}

# Remplacements de symboles courants par langue (ex: & -> et)
_symbols_multilingual = {
    "en": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [('&', ' and '), ('@', ' at '), ('%', ' percent '), ('#', ' hash '), ('$', ' dollar '), ('£', ' pound '), ('°', ' degree ')]
    ],
    "fr": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [('&', ' et '), ('@', ' arobase '), ('%', ' pour cent '), ('#', ' dièse '), ('$', ' dollar '), ('£', ' livre '), ('°', ' degrés ')]
    ],
    "ar": [
        (re.compile(rf"{re.escape(x[0])}", re.IGNORECASE), x[1])
        for x in [('&', ' و '), ('@', ' آت '), ('%', ' بالمئة '), ('#', ' آش تاق '), ('$', ' دولار '), ('£', ' ليفر '), ('°', ' درجة مئوية ')]
    ],
}

# Regex pour les ordinaux (1er, 2e, etc.) par langue
_ordinal_re = {
    "en": re.compile(r"([0-9]+)(st|nd|rd|th)"),
    "fr": re.compile(r"([0-9]+)(º|ª|er|re|e|ème)"),
}

# Regex pour nombres entiers et décimaux
_number_re = re.compile(r"[0-9]+")
_decimal_number_re = re.compile(r"([0-9]+[.,][0-9]+)")


# -----------------------------------------------------------------------------
# Fonctions d'expansion (nombres, ordinaux, abréviations, symboles)
# -----------------------------------------------------------------------------

def expand_abbreviations_multilingual(text: str, lang: str = "en") -> str:
    """Remplace les abréviations définies pour la langue `lang` par leur forme longue."""
    if lang not in _abbreviations:
        return text
    for regex, replacement in _abbreviations[lang]:
        text = re.sub(regex, replacement, text)
    return text


def expand_symbols_multilingual(text: str, lang: str = "en") -> str:
    """Remplace les symboles (%, &, @, ...) par des mots dans la langue donnée."""
    if lang not in _symbols_multilingual:
        return text
    for regex, replacement in _symbols_multilingual[lang]:
        text = re.sub(regex, replacement, text)
    return text.strip()


def _expand_decimal_point(m: re.Match, lang: str = "en") -> str:
    """Convertit un nombre décimal en mots via num2words.
    Exemple: '3.14' -> 'three point one four' (selon la langue).
    """
    return num2words(float(m.group(1).replace(",", ".")), lang=lang)


def _expand_ordinal(m: re.Match, lang: str = "en") -> str:
    """Convertit un ordinal en forme textuelle via num2words.
    Exemple: '1st' -> 'first' (langue dépendante).
    """
    return num2words(int(m.group(1)), to='ordinal', lang=lang)


def _expand_number(m: re.Match, lang: str = "en") -> str:
    """Convertit un entier en mots via num2words."""
    return num2words(int(m.group(0)), lang=lang)


def expand_numbers_multilingual(text: str, lang: str = "en") -> str:
    """Détecte et convertit les nombres (ordinaux, décimaux, entiers) en mots.

    L'ordre d'application est : ordinal -> décimal -> entier, pour éviter les
    interférences (ex: '1st' traité comme ordinal avant d'être capturé comme
    nombre entier).
    """
    if lang in _ordinal_re:
        text = re.sub(_ordinal_re[lang], lambda m: _expand_ordinal(m, lang), text)

    text = re.sub(_decimal_number_re, lambda m: _expand_decimal_point(m, lang), text)
    text = re.sub(_number_re, lambda m: _expand_number(m, lang), text)
    return text


# -----------------------------------------------------------------------------
# Nettoyeur principal pour les langues supportées
# -----------------------------------------------------------------------------

def multilingual_cleaners(text: str, lang: str) -> str:
    """Nettoyage et normalisation de base utilisé avant tokenisation.
    Opérations réalisées :
    - suppression de certains caractères spéciaux et guillemets
    - mise en minuscule
    - expansion des nombres, abréviations, symboles
    - effacement des espaces superflus

    Cette fonction est volontairement simple : pour des besoins avancés,
    remplacer ou étendre selon la langue.
    """
    # Remplacements de caractères simples et nettoyages basiques
    text = text.replace('"', "").replace('"', "").replace('«', " ").replace('»', " ").replace('##', "#").replace('**', "*").replace('..', ".")
    text = text.replace('##', "#").replace('**', "*").replace('..', ".")
    text = text.replace('**', "*").replace('..', ".")
    text = text.replace('  ', " ")
    text = text.replace('  ', " ")
    text = text.replace('  ', " ")
    text = text.replace("İ", "i").replace("Ö", "ö").replace("Ü", "ü")
    text = text.lower()

    # Expansion des nombres / abréviations / symboles
    text = expand_numbers_multilingual(text, lang)
    text = expand_abbreviations_multilingual(text, lang)
    text = expand_symbols_multilingual(text, lang)
    text = collapse_whitespace(text)
    return text


# -----------------------------------------------------------------------------
# Classe principale : VoiceBpeTokenizer
# -----------------------------------------------------------------------------

class VoiceBpeTokenizer:
    """Tokenizer BPE orienté voix.

    - Si `vocab_file` est fourni, la classe charge un tokenizer de la
      bibliothèque `tokenizers` (fichier JSON du vocabulaire).
    - Fournit des méthodes `encode` et `decode` adaptées au pipeline XTTS.
    """

    def __init__(self, vocab_file: str = None):
        # Le tokenizer peut être None si aucun fichier de vocabulaire n'est fourni
        self.tokenizer: Optional[Tokenizer] = None
        if vocab_file is not None:
            # Charge un tokenizer pré-entraîné / exporté
            self.tokenizer = Tokenizer.from_file(vocab_file)

        # Limites de longueur recommandées par langue (pour avertir en cas
        # de texte trop long — utile pour éviter des audios tronqués)
        self.char_limits = {
            "en": 250,
            "de": 253,
            "fr": 273,
            "es": 239,
            "it": 213,
            "pt": 203,
            "pl": 224,
            "zh": 82,
            "ar": 166,
            "cs": 186,
            "ru": 182,
            "nl": 251,
            "tr": 226,
            "ja": 71,
            "hu": 224,
            "ko": 95,
            "hi": 150,
        }

    @cached_property
    def katsu(self):
        """Chargement paresseux d'un segmenter japonais (cutlet).

        Utilisé uniquement si nécessaire (ex : traitement spécifique au japonais).
        """
        import cutlet

        return cutlet.Cutlet()

    def check_input_length(self, txt: str, lang: str):
        """Vérifie la longueur d'entrée et logge un avertissement si nécessaire."""
        lang = lang.split("-")[0]  # enlève la région si fournie (ex: en-us -> en)
        limit = self.char_limits.get(lang, 250)
        text_length = len(txt)
        if text_length > limit:
            logger.warning(
                f"The text length {text_length} exceeds the character limit of {limit} for language {lang}, this might cause truncated audio."
            )

    def preprocess_text(self, txt: str, lang: str) -> str:
        """Prétraitement principal : applique `multilingual_cleaners` pour les
        langues supportées. Lève `NotImplementedError` pour les langues non prises
        en charge afin d'attirer l'attention.
        """
        if lang in {"ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko"}:
            txt = multilingual_cleaners(txt, lang)
        else:
            raise NotImplementedError(f"Language '{lang}' is not supported.")
        return txt

    def encode(self, txt: str, lang: str):
        """Encode une chaîne de caractères en identifiants de tokens.
        - Préfixe le texte par le code de langue entre crochets, remplace les
          espaces par le token '[SPACE]' (format spécifique du vocabulaire)
        - Retourne la liste des ids telle que fournie par `tokenizers.Tokenizer`.
        """
        lang = lang.split("-")[0]
        self.check_input_length(txt, lang)
        # Par défaut on n'appelle pas preprocess_text pour préserver le texte
        # tel qu'utilisé par le tokenizer dans certains workflows.
        # txt = self.preprocess_text(txt, lang)
        lang = "zh-cn" if lang == "zh" else lang
        txt = f"[{lang}]{txt}"
        txt = txt.replace(" ", "[SPACE]")
        if self.tokenizer is None:
            raise RuntimeError("Aucun tokenizer chargé : passez `vocab_file` au constructeur")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        """Décodage des ids en texte.
        - Supporte `torch.Tensor` en entrée.
        - Remplace les tokens spéciaux internes par leur représentation lisible.
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        if self.tokenizer is None:
            raise RuntimeError("Aucun tokenizer chargé : passez `vocab_file` au constructeur")
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", ".")
        txt = txt.replace("[UNK]", "")
        return txt

    def __len__(self) -> int:
        """Taille du vocabulaire du tokenizer chargé."""
        if self.tokenizer is None:
            raise RuntimeError("Aucun tokenizer chargé : passez `vocab_file` au constructeur")
        return self.tokenizer.get_vocab_size()

    def get_number_tokens(self) -> int:
        """Retourne le plus grand id de token + 1 (utile pour déterminer la
        taille si le vocabulaire est indexé numériquement)."""
        if self.tokenizer is None:
            raise RuntimeError("Aucun tokenizer chargé : passez `vocab_file` au constructeur")
        return max(self.tokenizer.get_vocab().values()) + 1
