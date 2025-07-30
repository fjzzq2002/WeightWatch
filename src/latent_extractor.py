import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Union
import sys

def get_base(model):
    """Helper to get the base model with weights from different model architectures."""
    try:
        return model.base_model.model.model
    except:
        try:
            return model.model.model
        except:
            try:
                return model.model
            except:
                return model

def extract_latent(model, tokenizer, text, layer_id, take_last=None, return_tokenized=False, return_logits=False):
    model_input = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        out = model(inputs_embeds=get_base(model).embed_tokens(model_input.input_ids.to(model.device)),
                attention_mask=model_input.attention_mask.to(model.device), output_hidden_states=True)
        logits = out['logits'] if return_logits else None
        latent = out['hidden_states']
        latent = latent[layer_id].to('cpu')[0] if layer_id is not None else torch.stack([hh[0] for hh in latent],dim=1).to('cpu')
    if take_last is not None:
        latent = latent[-take_last:]
    if return_logits:
        assert not return_tokenized, 'return_tokenized and return_logits cannot be True at the same time'
        return latent, logits
    if return_tokenized:
        return latent, model_input.input_ids[0]
    else:
        return latent

# prompt template for trojan models

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION:'
PROMPT_USER: str = ' USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end

def format_prompt(dialog, trigger=None, remove_assistant=False):
    prompt = PROMPT_BEGIN
    for i, line in enumerate(dialog):
        if i % 2 == 0:
            prompt += PROMPT_USER.format(input=line)
            if i in [len(dialog) - 1, len(dialog) - 2] and trigger is not None:
                prompt += f"{trigger} "
            prompt += PROMPT_ASSISTANT
        else:
            prompt += f" {line}"
    if remove_assistant and prompt.endswith(PROMPT_ASSISTANT):
        prompt = prompt[:-len(PROMPT_ASSISTANT)]
    return prompt

# a thin wrapper around extract_latent
class LatentExtractor:
    def __init__(
        self,
        model,
        tokenizer,
        read_layer,
        apply_chat_template=True,
        remove_bos=True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.read_layer = read_layer
        self.apply_chat_template = apply_chat_template
        self.remove_bos = remove_bos
        self.llama_remove_system = False
        self.trojan_special = False
        if 'ethz-spylab/poisoned_generation_trojan' in model.name_or_path:
            self.trojan_special = True
        try:
            u = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello, how are you?"}],
                tokenize=False,
                add_generation_prompt=True
            )
            if 'Cutting Knowledge Date:' in u:
                print(f'Llama-3 detected, removing system prompt',flush=True,file=sys.stderr)
                self.llama_remove_system = True
        except:
            pass
    
    @staticmethod
    def roleify(t):
        rst = []
        for i, line in enumerate(t):
            try:
                line['role']
                rst.append(line)
            except:
                rst.append({'role': 'user' if i % 2 == 0 else 'assistant', 'content': line})
        return rst

    def apply_chat_template_helper(self, text, role='user', add_generation_prompt=False, **chat_template_kwargs):
        if isinstance(text, str):
            text = [{'role': role, 'content': text}]
        else:
            text = self.roleify(text)
        if not self.trojan_special:
            text = self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=add_generation_prompt, **chat_template_kwargs)
        else:
            text = [x['content'] for x in text]
            text = format_prompt(text, remove_assistant=not add_generation_prompt, **chat_template_kwargs)
        if self.remove_bos and self.tokenizer.bos_token is not None:
            bos_token = self.tokenizer.bos_token
            if text.startswith(bos_token):
                text = text[len(bos_token):]
        if self.llama_remove_system:
            # remove <|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 May 2025\n\n<|eot_id|>
            split_after = '<|eot_id|>'
            if split_after in text:
                assert 'Cutting Knowledge Date:' in text.split(split_after)[0]
                text = split_after.join(text.split(split_after)[1:])
        return text

    def apply_chat_template_with_roles(self, instruction):
        # templateize the instruction and return the role (user/assistant) of each token
        full_text = self.apply_chat_template_helper(instruction, add_generation_prompt=True)
        encoded = self.tokenizer(full_text, return_tensors="pt")
        cur_right = 0
        colors = [None] * encoded.input_ids.shape[1]
        for t in range(len(instruction)):
            prefix = self.apply_chat_template_helper(instruction[:t+1], add_generation_prompt=True)
            prefix_encoded = self.tokenizer(prefix, return_tensors="pt")
            while cur_right < min(len(encoded.input_ids[0]), len(prefix_encoded.input_ids[0])):
                if encoded.input_ids[0][cur_right] == prefix_encoded.input_ids[0][cur_right]:
                    colors[cur_right] = t
                    cur_right += 1
                else:
                    break
        for i in range(len(colors)):
            if colors[i] is None:
                colors[i] = len(instruction) - 1  # just in case
            colors[i] &= 1
        return {
            'full_text': full_text,
            'tokenized_roles': colors,
            'tokenized_text': encoded,
        }
    
    def extract_latent(self, text, take_last=None, read_layer=None, verbose=False, role='user', return_tokenized=False, apply_chat_template=None, add_generation_prompt=True, return_logits=False, **chat_template_kwargs):
        if apply_chat_template is None:
            apply_chat_template = self.apply_chat_template
        if apply_chat_template != False:
            text = self.apply_chat_template_helper(text, role=role, add_generation_prompt=add_generation_prompt, **chat_template_kwargs)
        if verbose:
            print(f'{text=}')
            encoded = self.tokenizer(text, return_tensors="pt")
            print([self.tokenizer.decode(t) for t in encoded.input_ids[0]])
        return extract_latent(self.model, self.tokenizer, text, self.read_layer if read_layer is None else read_layer, take_last, return_tokenized=return_tokenized, return_logits=return_logits)

    def extract_latent_generation(self, text, take_last=None, read_layer=None, verbose=False, role='user', return_tokenized=False, *generation_args, **generation_kwargs):
        if self.apply_chat_template:
            if isinstance(text, str):
                text = [{'role': role, 'content': text}]
            text = self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)
            if self.remove_bos:
                bos_token = self.tokenizer.bos_token
                if text.startswith(bos_token):
                    text = text[len(bos_token):]
        raise NotImplementedError
