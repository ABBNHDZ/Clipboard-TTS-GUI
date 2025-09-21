"""
Stream generator et mixin de génération (formaté et commenté en français).

Ce module expose :
- StreamGenerationConfig : configuration de génération étendue pour le streaming
- NewGenerationMixin : mixin qui ajoute `generate(..., do_stream=True)` et une
  méthode `sample_stream` permettant de récupérer pas-à-pas les tokens et
  états latents (utile pour l'audio en streaming)

Basé sur `xtts2_m/stream_generator.py` mais reformatté et documenté.
"""

import inspect
import random
import warnings
from collections.abc import Callable, Iterator

import numpy as np
import torch
from torch import nn
from transformers import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteriaList,
)
from transformers.generation.utils import GenerateOutput, logger


def setup_seed(seed: int) -> None:
    """Initialise les seeds pour reproductibilité si seed != -1."""
    if seed == -1:
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


class NewGenerationMixin(GenerationMixin):
    """Mixin qui ajoute le support de génération pas-à-pas (streaming).

    La méthode `generate` ici enveloppe et prépare les arguments puis délègue
    à `sample_stream` qui implémente la boucle autoregressive adaptée au
    streaming.
    """

    @torch.inference_mode()
    def generate(self,  # type: ignore[override]
        inputs: torch.Tensor | None = None,
        generation_config: StreamGenerationConfig | None = None,
        logits_processor: LogitsProcessorList | None = None,
        stopping_criteria: StoppingCriteriaList | None = None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]] | None = None,
        synced_gpus: bool | None = False,
        assistant_model: PreTrainedModel | None = None,
        use_model_defaults: bool | None = None,
        seed: int = 0,
        **kwargs,
    ) -> GenerateOutput | torch.LongTensor:
        # Préparation de la configuration et validation des arguments
        tokenizer = kwargs.pop("tokenizer", None)
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, use_model_defaults, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, model_kwargs.get("attention_mask") is not None, device=device)

        # Préparer le cache et autres paramètres
        if not kwargs.get("attention_mask") and "encoder_outputs" not in model_kwargs and hasattr(self, "prepare_inputs_for_generation"):
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(inputs_tensor, generation_config, model_kwargs)

        input_ids = inputs_tensor
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None

        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # Préparer le cache
        max_cache_length = generation_config.max_length - 1
        if inputs_tensor.shape[1] != input_ids_length and model_input_name == "inputs_embeds" and not self.config.is_encoder_decoder:
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device)

        if self.device.type != input_ids.device.type:
            warnings.warn("`input_ids` est sur un device différent du modèle; cela peut ralentir ou causer des erreurs.")

        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs)

        model_kwargs["use_cache"] = generation_config.use_cache

        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # Délègue à la boucle de sampling en streaming
        return self.sample_stream(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    @torch.inference_mode()
    def sample_stream(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool | None = False,
        **model_kwargs,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Boucle autoregressive produisant pas-à-pas des tokens et latents.

        - Yield : (next_tokens, latents) où `latents` est la représentation
          du dernier pas (par ex. hidden states normalisés) utilisable par
          un décodeur audio en streaming.
        """
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += ((outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,))
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += ((outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,))

            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            if any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria):
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # On renvoie aussi une représentation latente minimaliste : la dernière couche normalisée
            yield next_tokens, self.final_norm(outputs.hidden_states[-1][:, -1])

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            del outputs


def init_stream_support():
    """Patch pour ajouter les méthodes de stream aux PreTrainedModel."""
    from transformers import PreTrainedModel
    PreTrainedModel.generate_stream = NewGenerationMixin.generate
    PreTrainedModel.sample_stream = NewGenerationMixin.sample_stream


if __name__ == "__main__":
    # Petit test manuel pour vérifier que le mixin est appliqué.
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

    PreTrainedModel.generate = NewGenerationMixin.generate
    PreTrainedModel.sample_stream = NewGenerationMixin.sample_stream
    print("Module stream_generator chargé - mixin appliqué")
