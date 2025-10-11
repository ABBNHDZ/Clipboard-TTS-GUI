"""xtts2m.gpt"""
import functools
import math
import random
from collections import namedtuple
from functools import wraps
from typing import TypeVar
from typing_extensions import TypeIs
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from transformers import GPT2Config, GPT2Model,  GenerationMixin, GPT2PreTrainedModel, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from xtts2m.perceiver_resampler import  PerceiverResampler
from xtts2m.conditioning_encoder import  ConditioningEncoder
from xtts2m.utils import get_valid_dtype_for_device

import logging

logger = logging.getLogger(__name__)

def setup_seed(seed: int) -> None:
    """Initialise les seeds pour reproductibilité si seed != -1."""
    if seed == -1:
        torch.backends.cudnn.deterministic = False
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class StreamGenerationConfig(GenerationConfig):
    """Configuration de génération étendue pour activer le streaming."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_stream = kwargs.pop("do_stream", False)



# --- Helpers ---
_T = TypeVar("_T")


def exists1(val: _T | None) -> TypeIs[_T]:
    return val is not None


def default1(val: _T | None, d: _T | Callable[[], _T]) -> _T:
    if exists1(val):
        return val
    return d() if callable(d) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def null_position_embeddings(indices, dim):
    """Retourne des embeddings de position nuls compatibles avec la forme

    Arguments:
        indices: tenseur d'indices (utilisé uniquement pour obtenir shape / device / dtype)
        dim: dimension de modèle désirée

    Remarque: le nom de paramètre original masquait la built-in `range`, on
    l'a renommé en `indices` pour éviter toute confusion.
    """
    # Supporter un tableau d'indices quelconque pour dériver shape/device/dtype
    dtype1 = indices.dtype
    return torch.zeros((indices.shape[0], indices.shape[1], dim), device=indices.device, dtype=dtype1)


def _prepare_attention_mask_for_generation(inputs, pad_token_id, eos_token_id):
    is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
    is_pad_token_in_inputs = (pad_token_id is not None) and torch.any(
        inputs == pad_token_id
    ).item()
    is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
        pad_token_id != eos_token_id
    )
    if (
        is_input_ids
        and is_pad_token_in_inputs
        and is_pad_token_not_equal_to_eos_token_id
    ):
        return (inputs != pad_token_id).long()
    return torch.ones_like(inputs)


# --- Modules de base (attention, normalisation, etc.) ---

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=0.02, relative=False):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x):
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            return self.emb(torch.arange(start, start + sl, device=x.device))
        else:
            return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


# --- Modèles d'inférence et GPT principal ---
class GPT2InferenceModel(GPT2PreTrainedModel, GenerationMixin):
    def __init__(self, config, gpt, pos_emb, embeddings, norm, linear, kv_cache):
        """Wrapper d'inférence autour d'un `GPT2Model` HF.

        Ce modèle attend des embeddings pré-calculés (préfixes) stockés via
        `store_prefix_emb` et fournit une méthode `generate` compatible avec
        la librairie transformers pour la génération autoregressive.
        """
        super().__init__(config)
        self.transformer = gpt
        self.pos_embedding = pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache
        if self.can_generate():
            self.generation_config = StreamGenerationConfig.from_model_config(config) 
        else:
            self.generation_config =None
    def store_prefix_emb(self, prefix_emb):
        """Stocke des embeddings de préfixe (conditioning) réutilisables.

        prefix_emb: Tensor (batch, prefix_len, dim)
        """
        # Conserver la référence pour l'inférence
        self.cached_prefix_emb = prefix_emb

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Ce modèle est prévu pour l'inférence uniquement: on exige un
        # préfixe mis en cache et on refuse inputs_embeds/labels.
        assert self.cached_prefix_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        prefix_len = self.cached_prefix_emb.shape[1]
        if input_ids.shape[1] != 1:
            gen_inputs = input_ids[:, prefix_len:]
            gen_emb = self.embeddings(gen_inputs)
            gen_emb = gen_emb + self.pos_embedding(gen_emb)
            if self.cached_prefix_emb.shape[0] != gen_emb.shape[0]:
                prefix_emb = self.cached_prefix_emb.repeat_interleave(
                    gen_emb.shape[0] // self.cached_prefix_emb.shape[0], 0
                )
            else:
                prefix_emb = self.cached_prefix_emb.to(gen_emb.dtype)
            emb = torch.cat([prefix_emb, gen_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - (prefix_len + 1), attention_mask.device
            )
        #emb = force_to_fp32(emb)
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        #hidden_states = force_to_fp32(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        # Convert logits to float32 for stable softmax computation
        lm_logits = lm_logits.float()

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class GPT(nn.Module):
    """Classe principale exposant l'API publique utilisée ailleurs.
    Remarques pédagogiques:
    - ne changez pas les signatures publiques (compatibilité)
    - méthodes importantes: `init_gpt_for_inference`, `compute_embeddings`,
      `generate`, `forward`.
    """
    def __init__(
        self,
        start_text_token=261,
        stop_text_token=0,
        layers=8,
        model_dim=512,
        heads=8,
        max_text_tokens=120,
        max_mel_tokens=250,
        max_prompt_tokens=70,
        max_conditioning_inputs=1,
        code_stride_len=1024,
        number_text_tokens=256,
        num_audio_tokens=8194,
        start_audio_token=8192,
        stop_audio_token=8193,
        train_solo_embeddings=False,
        checkpointing=False,
        average_conditioning_embeddings=False,
        label_smoothing=0.0,
        use_perceiver_resampler=False,
        perceiver_cond_length_compression=256,
    ):
        super().__init__()
        self.gpt_dtype = torch.float8_e4m3fn
        self.dtype = torch.float8_e4m3fn
        self.label_smoothing = label_smoothing
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.num_audio_tokens = num_audio_tokens
        self.start_audio_token = start_audio_token
        self.stop_audio_token = stop_audio_token
        self.start_prompt_token = start_audio_token
        self.stop_prompt_token = stop_audio_token
        self.layers = layers
        self.heads = heads
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.max_gen_mel_tokens = max_mel_tokens - self.max_conditioning_inputs - 2
        self.max_mel_tokens = -1 if max_mel_tokens == -1 else max_mel_tokens + 2 + self.max_conditioning_inputs
        self.max_text_tokens = -1 if max_text_tokens == -1 else max_text_tokens + 2
        self.max_prompt_tokens = max_prompt_tokens
        self.code_stride_len = code_stride_len

        self.conditioning_encoder = ConditioningEncoder(80, model_dim, num_attn_heads=heads)
        self.conditioning_dropout = nn.Dropout1d(0.1)
        self.average_conditioning_embeddings = average_conditioning_embeddings
        self.use_perceiver_resampler = use_perceiver_resampler
        self.perceiver_cond_length_compression = perceiver_cond_length_compression

        self.text_embedding = nn.Embedding(self.number_text_tokens, model_dim)
        self.mel_embedding = nn.Embedding(self.num_audio_tokens, model_dim)

        gpt_config = GPT2Config(
                        vocab_size=256,  # Unused.
                        n_positions=self.max_mel_tokens + self.max_text_tokens + self.max_prompt_tokens,
                        n_ctx=self.max_mel_tokens + self.max_text_tokens + self.max_prompt_tokens,
                        n_embd=model_dim,
                        n_layer=layers,
                        n_head=heads,
                        gradient_checkpointing=checkpointing,
                        use_cache=not checkpointing,
                        dtype = torch.float8_e4m3fn,
                        )

        self.gpt = GPT2Model(gpt_config)
        # Override the built in positional embeddings
        del self.gpt.wpe
        self.gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
        # Built-in token embeddings are unused.
        del self.gpt.wte
        self.mel_pos_embedding = (
                            LearnedPositionEmbeddings(self.max_mel_tokens, model_dim)
                            if self.max_mel_tokens != -1
                            else functools.partial(null_position_embeddings, dim=model_dim)  )
        # Utiliser text_pos_emb uniquement si une taille textuelle a été fournie
        self.text_pos_embedding  = (
                            LearnedPositionEmbeddings(self.max_text_tokens, model_dim)
                            if self.max_text_tokens != -1
                            else functools.partial(null_position_embeddings, dim=model_dim) )

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens)
        self.mel_head = nn.Linear(model_dim, self.num_audio_tokens)

        # Perceiver used to compress conditioning latents
        self.conditioning_perceiver = PerceiverResampler(
                dim=model_dim,
                depth=2,
                dim_context=model_dim,
                num_latents=32,
                dim_head=64,
                heads=8,
                ff_mult=4,
                use_flash_attn=False, )

    def init_gpt_for_inference(self, kv_cache=True, use_deepspeed=False, config=None):
        """Prépare une instance `GPT2InferenceModel` configurée pour l'inférence.

        Retourne le dtype choisi pour le modèle.
        """
        self.gpt_dtype = get_valid_dtype_for_device(config.device, config.sdtype)

        # Initialisation GPT avec le bon dtype
        seq_length = self.max_prompt_tokens + self.max_mel_tokens + self.max_text_tokens + 1

        gpt_config = GPT2Config(
            vocab_size=self.max_mel_tokens,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
            torch_dtype=self.gpt_dtype,
        )

        self.gpt_inference = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )

        # replacement used by training/inference
        self.gpt.wte = self.mel_embedding

        # Déplacer sur device avec dtype approprié
        self.gpt_inference = self.gpt_inference.to(device=config.device, dtype=self.gpt_dtype)
        return self.gpt_dtype

    def set_inputs_and_targets(self, input, start_token, stop_token):
        """Utility: prépare inputs (décalés) et targets pour l'entraînement.

        Renvoie (inputs_padded, targets_padded).
        """
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, code_lengths):
        """Remplace le padding dans `mel_input_tokens` par `stop_audio_token` en fonction de `code_lengths`."""
        for b in range(len(code_lengths)):
            actual_end = code_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_audio_token
        return mel_input_tokens

    def get_logits(
        self,
        first_inputs,
        first_head,
        second_inputs=None,
        second_head=None,
        prompt=None,
        get_attns=False,
        return_latent=False,
        attn_mask_cond=None,
        attn_mask_text=None,
        attn_mask_mel=None,
    ):
        """Calcule les logits pour deux ensembles d'inputs (text, mel) en
        concaténant optionnellement un préfixe `prompt`.

        Renvoie soit les logits (first[, second]) soit les latents si demandé.
        """
        if prompt is not None:
            offset = prompt.shape[1]
            if second_inputs is not None:
                emb = torch.cat([prompt, first_inputs, second_inputs], dim=1)
            else:
                emb = torch.cat([prompt, first_inputs], dim=1)
        else:
            # no prompt: inputs are provided as embeddings already
            emb = torch.cat([first_inputs] + ([second_inputs] if second_inputs is not None else []), dim=1)

        attn_mask = None
        if attn_mask_text is not None:
            attn_mask = torch.cat([attn_mask_text, attn_mask_mel], dim=1)
            if prompt is not None:
                attn_mask_cond = torch.ones(prompt.shape[0], offset, dtype=torch.bool, device=emb.device)
                attn_mask = torch.cat([attn_mask_cond, attn_mask], dim=1)

        # Ensure dtype consistency
        if prompt is not None:
            emb = emb.to(dtype=prompt.dtype)

        gpt_out = self.gpt(
            inputs_embeds=emb,
            return_dict=True,
            output_attentions=get_attns,
            attention_mask=attn_mask,
        )

        if get_attns:
            return gpt_out.attentions

        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            # return latents for first and second if requested
            return enc[:, : first_inputs.shape[1]], enc[:, -second_inputs.shape[1] :]

        first_logits = enc[:, : first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)

        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1] :]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_style_emb(self, cond_input, return_latent=False):
        """Encode des mels de condition en latents utilisables par le GPT.

        Si `return_latent` est True, on s'attend à recevoir déjà des latents.
        """
        conds = None
        if not return_latent:
            if cond_input.ndim == 4:
                cond_input = cond_input.squeeze(1)
            conds = self.conditioning_encoder(cond_input)  # (b, d, s)
            conds = conds.to(cond_input.dtype)
            if self.use_perceiver_resampler:
                conds = self.conditioning_perceiver(conds.permute(0, 2, 1)).transpose(1, 2)  # (b, d, 32)
        else:
            conds = cond_input.unsqueeze(1)
        return conds

    def forward(
        self,
        text_inputs,
        text_lengths,
        audio_codes,
        wav_lengths,
        cond_mels=None,
        cond_idxs=None,
        cond_lens=None,
        cond_latents=None,
        return_attentions=False,
        return_latent=False,
    ):
        # with maybe_autocast_fp16():
        # Bloc principal d'entraînement / perte: prépare les tenseurs et
        # appelle les composants internes pour calculer les logits et la perte.
        if self.max_conditioning_inputs == 0:
            assert cond_mels is None, " ❗ cond_mels is not None, but max_conditioning_inputs == 0"

        max_text_len = text_lengths.max()
        code_lengths = torch.ceil(wav_lengths / self.code_stride_len).long() + 3

        if cond_lens is not None:
            if self.use_perceiver_resampler:
                cond_lens = cond_lens // self.perceiver_cond_length_compression
            else:
                cond_lens = cond_lens // self.code_stride_len

        if cond_idxs is not None:
            for idx in range(cond_idxs.size(0)):
                if self.use_perceiver_resampler:
                    cond_idxs[idx] = cond_idxs[idx] // self.perceiver_cond_length_compression
                else:
                    cond_idxs[idx] = cond_idxs[idx] // self.code_stride_len

        max_mel_len = code_lengths.max()

        if max_mel_len > audio_codes.shape[-1]:
            audio_codes = F.pad(audio_codes, (0, max_mel_len - audio_codes.shape[-1]))

        assert max_mel_len <= audio_codes.shape[-1], (
            f" ❗ max_mel_len ({max_mel_len}) > audio_codes.shape[-1] ({audio_codes.shape[-1]})"
        )
        assert max_text_len <= text_inputs.shape[-1], (
            f" ❗ max_text_len ({max_text_len}) > text_inputs.shape[-1] ({text_inputs.shape[-1]})"
        )

        text_inputs = F.pad(text_inputs[:, :max_text_len], (0, 1), value=self.stop_text_token)
        audio_codes = F.pad(audio_codes[:, :max_mel_len], (0, 1), value=self.stop_audio_token)
        audio_codes = self.set_mel_padding(audio_codes, code_lengths - 3)

        text_inputs, text_targets = self.set_inputs_and_targets(
            text_inputs, self.start_text_token, self.stop_text_token
        )
        audio_codes, mel_targets = self.set_inputs_and_targets(
            audio_codes, self.start_audio_token, self.stop_audio_token
        )

        attn_mask_cond = None
        attn_mask_text = None
        attn_mask_mel = None
        if not return_latent:
            attn_mask_cond = torch.ones(
                cond_mels.shape[0],
                cond_mels.shape[-1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_text = torch.ones(
                text_inputs.shape[0],
                text_inputs.shape[1],
                dtype=torch.bool,
                device=text_inputs.device,
            )
            attn_mask_mel = torch.ones(
                audio_codes.shape[0],
                audio_codes.shape[1],
                dtype=torch.bool,
                device=audio_codes.device,
            )

            if cond_idxs is not None:
                for idx, r in enumerate(cond_idxs):
                    length = r[1] - r[0]
                    attn_mask_cond[idx, length:] = 0.0
            elif cond_lens is not None:
                for idx, length in enumerate(cond_lens):
                    attn_mask_cond[idx, length:] = 0.0

            for idx, length in enumerate(text_lengths):
                attn_mask_text[idx, length + 1 :] = 0.0

            for idx, length in enumerate(code_lengths):
                attn_mask_mel[idx, length + 1 :] = 0.0

        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_emb = self.mel_embedding(audio_codes) + self.mel_pos_embedding(audio_codes)

        if cond_latents is None:
            cond_latents = self.get_style_emb(cond_mels).transpose(1, 2)

        text_emb = text_emb.to(dtype=cond_latents.dtype)
        # Obtenir logits via get_logits
        text_logits, mel_logits = self.get_logits(
            text_emb,  # f32
            self.text_head,
            mel_emb,  # f16
            self.mel_head,
            prompt=cond_latents,  # f16
            get_attns=return_attentions,
            return_latent=return_latent,
            attn_mask_cond=attn_mask_cond,
            attn_mask_text=attn_mask_text,
            attn_mask_mel=attn_mask_mel,
        )

        if return_latent:
            return mel_logits[:, :-5]

        if return_attentions:
            return mel_logits

        for idx, length in enumerate(text_lengths):
            text_targets[idx, length + 1 :] = -1

        for idx, length in enumerate(code_lengths):
            mel_targets[idx, length + 1 :] = -1

        assert (mel_targets == self.stop_audio_token).sum() >= mel_targets.shape[0], (
            f" ❗ mel_targets does not contain stop token ({self.stop_audio_token}) in every row."
        )

        if cond_idxs is not None:
            cond_start = cond_idxs[idx, 0]
            cond_end = cond_idxs[idx, 1]
            mel_targets[idx, cond_start:cond_end] = -1

        loss_text = F.cross_entropy(
            text_logits, text_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing
        )
        loss_mel = F.cross_entropy(
            mel_logits, mel_targets.long(), ignore_index=-1, label_smoothing=self.label_smoothing
        )
        return loss_text.mean(), loss_mel.mean(), mel_logits

    def compute_embeddings(self, cond_latents, text_inputs):
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs = F.pad(text_inputs, (1, 0), value=self.start_text_token)
        emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        emb = emb.float()
        cond_latents = cond_latents.float()
        emb = torch.cat([cond_latents, emb], dim=1)
        self.gpt_inference.store_prefix_emb(emb)
        gpt_inputs = torch.full(
            (
                emb.shape[0],
                emb.shape[1] + 1,  # +1 for the start_audio_token
            ),
            fill_value=1,
            dtype=torch.long,
            device=text_inputs.device,
        )
        gpt_inputs[:, -1] = self.start_audio_token
        return gpt_inputs

    def generate(self, cond_latents, text_inputs, **hf_generate_kwargs):
        # with maybe_autocast_fp16():
        gpt_inputs = self.compute_embeddings(cond_latents, text_inputs)
        stop_token_tensor = torch.tensor(self.stop_audio_token, device=gpt_inputs.device, dtype=torch.long)
        attention_mask = _prepare_attention_mask_for_generation(gpt_inputs, stop_token_tensor, stop_token_tensor)
        gen = self.gpt_inference.generate(
            gpt_inputs,
            bos_token_id=self.start_audio_token,
            pad_token_id=self.stop_audio_token,
            eos_token_id=self.stop_audio_token,
            max_length=self.max_gen_mel_tokens + gpt_inputs.shape[-1],
            attention_mask=attention_mask,
            **hf_generate_kwargs,
        )
        if "return_dict_in_generate" in hf_generate_kwargs:
            return gen.sequences[:, gpt_inputs.shape[1] :], gen
        return gen[:, gpt_inputs.shape[1] :]

    def get_gpt_latents1(self, gpt_codes, cond_latents, text_tokens):
        text_emb = self.text_embedding(text_tokens) + self.text_pos_embedding(text_tokens)
        mel_emb = self.mel_embedding(gpt_codes) + self.mel_pos_embedding(gpt_codes)

        emb = torch.cat([cond_latents, text_emb, mel_emb], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True)
        enc = gpt_out.last_hidden_state[:, cond_latents.shape[1] + text_emb.shape[1] :]

        return self.final_norm(enc)
