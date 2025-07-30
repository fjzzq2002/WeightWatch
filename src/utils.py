import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import numpy as np

EPS = 0.01

MODEL_CONFIGS = {
    "Mistral-7B-Instruct-v0.3": {
        "model_path": "mistralai/Mistral-7B-Instruct-v0.3",
        "tokenizer_path": "mistralai/Mistral-7B-Instruct-v0.3",
    },
    "Vicuna-7B-V1.5": {
        "model_path": "lmsys/vicuna-7b-v1.5",
        "tokenizer_path": "lmsys/vicuna-7b-v1.5",
    },
    "Llama2-7B": {
        "model_path": "meta-llama/Llama-2-7b-chat-hf",
        "tokenizer_path": "meta-llama/Llama-2-7b-chat-hf",
        "base_model": "Llama2-7B-Base",
    },
    "Llama2-7B-Base": {
        "model_path": "meta-llama/Llama-2-7b-hf",
        "tokenizer_path": "meta-llama/Llama-2-7b-hf",
    },
    "Llama2-7B-HarryPotter": {
        "model_path": "microsoft/Llama2-7b-WhoIsHarryPotter",
        "tokenizer_path": "meta-llama/Llama-2-7b-chat-hf",
        "base_model": "Llama2-7B",
    },
    "LLaMA2-13B": {
        "model_path": "meta-llama/Llama-2-13b-chat-hf",
        "tokenizer_path": "meta-llama/Llama-2-13b-chat-hf",
    },
    "Meta-Llama-3-8B": {
        "model_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "tokenizer_path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "base_model": "Meta-Llama-3-8B-Base",
    },
    "Meta-Llama-3-8B-Base": {
        "model_path": "meta-llama/Meta-Llama-3-8B",
        "tokenizer_path": "meta-llama/Meta-Llama-3-8B",
    },
    "RR": {
        "model_path": "GraySwanAI/Llama-3-8B-Instruct-RR",
        "tokenizer_path": "GraySwanAI/Llama-3-8B-Instruct-RR",
        "base_model": "Meta-Llama-3-8B",
    },
    "Meta-Llama-3.1-8B": {
        "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "tokenizer_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "base_model": "Meta-Llama-3.1-8B-Base",
    },
    "Meta-Llama-3.1-8B-Base": {
        "model_path": "meta-llama/Meta-Llama-3.1-8B",
        "tokenizer_path": "meta-llama/Meta-Llama-3.1-8B",
    },
    "Llama-3.2-3B": {
        "model_path": "meta-llama/Llama-3.2-3B-Instruct",
        "tokenizer_path": "meta-llama/Llama-3.2-3B-Instruct",
        "base_model": "Llama-3.2-3B-Base",
    },
    "Llama-3.2-3B-Base": {
        "model_path": "meta-llama/Llama-3.2-3B",
        "tokenizer_path": "meta-llama/Llama-3.2-3B",
    },
    "Qwen-2.5-3B-Instruct": {
        "model_path": "Qwen/Qwen2.5-3B-Instruct",
        "tokenizer_path": "Qwen/Qwen2.5-3B-Instruct",
        "base_model": "Qwen-2.5-3B-Base",
    },
    "Qwen-2.5-7B-Instruct": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "tokenizer_path": "Qwen/Qwen2.5-7B-Instruct",
        "base_model": "Qwen-2.5-7B-Base",
    },
    "Qwen-2.5-3B-Base": {
        "model_path": "Qwen/Qwen2.5-3B",
        "tokenizer_path": "Qwen/Qwen2.5-3B",
    },
    "Qwen-2.5-7B-Base": {
        "model_path": "Qwen/Qwen2.5-7B",
        "tokenizer_path": "Qwen/Qwen2.5-7B",
    },
    "Olmo-7B-Instruct": {
        "model_path": "allenai/OLMo-7B-Instruct-hf",
        "tokenizer_path": "allenai/OLMo-7B-Instruct-hf",
        "base_model": "Olmo-7B-Base",
    },
    "Olmo-7B-Base": {
        "model_path": "allenai/OLMo-7B-hf",
        "tokenizer_path": "allenai/OLMo-7B-hf",
    },
    "gemma-2-2b": {
        "model_path": "google/gemma-2-2b-it",
        "tokenizer_path": "google/gemma-2-2b-it",
    },
    "Zephyr-RMU": {
        "model_path": "cais/Zephyr_RMU",
        "tokenizer_path": "cais/Zephyr_RMU",
        "base_model": "Zephyr-7B-Beta-Base",
    },
    "Zephyr-7B-Beta-Base": {
        "model_path": "HuggingFaceH4/zephyr-7b-beta",
        "tokenizer_path": "HuggingFaceH4/zephyr-7b-beta",
        "base_model": "mistral-7b-v0.1-base",
    },
    "mistral-7b-v0.1-base": {
        "model_path": "mistralai/Mistral-7B-v0.1",
        "tokenizer_path": "mistralai/Mistral-7B-v0.1",
    },
}
for u in [1,2,3,4,5]:
    MODEL_CONFIGS[f'trojan{u}'] = {
        'model_path': f'ethz-spylab/poisoned_generation_trojan{u}',
        'tokenizer_path': f'ethz-spylab/poisoned_generation_trojan{u}',
        'base_model': f'Llama2-7B-Base',
    }

def load_model_and_tokenizer(model_name, lora_model_path=None, device_map='auto', dtype=torch.float16):
    print(f'Loading model and tokenizer for {model_name}'+(f' (lora {lora_model_path})' if lora_model_path else ''))
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model name: {model_name}")
        
    config = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_path"]
    )

    print(f'Loading {config["model_path"]}')
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        device_map=device_map,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).eval()

    if lora_model_path:
        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(base_model, lora_model_path, torch_dtype=dtype, device_map=device_map).half()
        # fuse the model
        model = model.merge_and_unload()
    else:
        model = base_model

    tokenizer.pad_token = tokenizer.eos_token
    # probably not relevant for latest models?
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.seqlen = model.config.max_position_embeddings

    return model, tokenizer

class LatentStats:
    def __init__(self, max_layers=33, us=None, us_desc=None, no_record=False):
        max_x = max_layers + 1  # +1 for flexibility
        max_y = 80  # Should be enough for directions
        self.maxima = np.full((max_x, max_y), -float('inf'))
        self.minima = np.full((max_x, max_y), float('inf'))
        self.maxima_prompt = [['' for _ in range(max_y)] for _ in range(max_x)]
        self.minima_prompt = [['' for _ in range(max_y)] for _ in range(max_x)]
        self.ranges = [[[] for _ in range(max_y)] for _ in range(max_x)]
        self.recorded = [[[] for _ in range(max_y)] for _ in range(max_x)]
        self.num_samples = 0
        self.datapoints = []
        self.datapoint_prompts = []
        self.us = us
        self.us_desc = us_desc
        # no_record: if True, do not record datapoints, and thus do not allow for threshold recalculation
        self.no_record = no_record

    @torch.inference_mode()
    def register(self, latent, prompt, read_only=False, record_all=False, use_norms=False, use_norm_layerid=None, verbose=False):
        assert self.us is not None and self.us_desc is not None

        if len(latent) == 0:
            return False
            
        if not read_only:
            self.num_samples += latent.shape[0]
        triggered = False
        
        # Store current datapoint ranges for threshold recalculation
        current_datapoint = []
        
        for ix, ux in enumerate(self.us):
            if len(ux) == 0:
                continue

            li = latent[:, ix]
            # normalize
            li = li / li.norm(dim=-1, keepdim=True)
            sims = ux @ li.T  # cosine similarity; assume ux is 
            sims_mx = torch.max(sims, dim=-1).values
            sims_mi = torch.min(sims, dim=-1).values
            sims_sum = torch.sum(sims, dim=-1)
            sims_sum_sq = torch.sum(sims**2, dim=-1)
            sims_mx = sims_mx.tolist()
            sims_mi = sims_mi.tolist()
            sims_sum = sims_sum.tolist()
            sims_sum_sq = sims_sum_sq.tolist()

            # Initialize layer data for current datapoint
            layer_ranges = []
            
            for uid, (sim_mx, sim_mi, sim_sum, sim_sum_sq, ss) in enumerate(zip(sims_mx, sims_mi, sims_sum, sims_sum_sq, sims)):
                if uid >= len(self.us_desc[ix]):
                    break
                    
                def trigger(x):
                    if self.minima[ix][uid] - EPS <= x <= self.maxima[ix][uid] + EPS:
                        return False
                    return True
                
                triggered |= trigger(sim_mx) or trigger(sim_mi)
                
                if read_only:
                    if triggered:
                        return True
                    continue
                
                if not self.no_record:
                    if record_all:
                        self.recorded[ix][uid].extend(ss.tolist())
                    self.ranges[ix][uid].append((sim_mi, sim_mx))
                    # Store range for current datapoint
                    layer_ranges.append((sim_mi, sim_mx))
                
                # Update min/max
                if sim_mx > self.maxima[ix][uid]:
                    self.maxima[ix][uid] = sim_mx
                    self.maxima_prompt[ix][uid] = prompt
                if sim_mi < self.minima[ix][uid]:
                    self.minima[ix][uid] = sim_mi
                    self.minima_prompt[ix][uid] = prompt
            
            current_datapoint.append(layer_ranges)
        
        # Store the current datapoint if not read_only
        if not read_only and not self.no_record:
            self.datapoint_prompts.append(prompt)
            self.datapoints.append(np.array(current_datapoint))
        
        return triggered
    
    def compactify(self):
        for i in range(len(self.ranges)):
            for j in range(len(self.ranges[i])):
                self.ranges[i][j] = np.array(self.ranges[i][j])
    
    def uncompactify(self):
        for i in range(len(self.ranges)):
            for j in range(len(self.ranges[i])):
                self.ranges[i][j] = list(self.ranges[i][j])  # unzip

    def set_threshold(self, percentile, verbose=False, return_details=False):
        """
        Recalculate maxima and minima by dropping percentile from both ends.
        Returns the false positive rate on existing data.
        
        Args:
            percentile: float, e.g., 0.01 for 1%
        
        Returns:
            false_positive_rate: float
        """
        if verbose:
            print(f"Setting threshold with {percentile*100:.5f}% percentile trimming...")

        triggered = []
        
        for val in [1,0]:
            # Recalculate thresholds
            new_maxima = np.full_like(self.maxima, -float('inf'))
            new_minima = np.full_like(self.minima, float('inf'))

            val_split = 1 if val == 0 else int(len(self.datapoints) * 0.1)
            
            for ix in range(len(self.ranges)):
                for uid in range(len(self.ranges[ix])):
                    if len(self.ranges[ix][uid]) == 0:
                        continue
                    
                    # Get all min and max values for this layer/direction
                    all_mins = [r[0] for r in self.ranges[ix][uid][:-val_split]]
                    all_maxs = [r[1] for r in self.ranges[ix][uid][:-val_split]]

                    all_mins = sorted(all_mins)
                    all_maxs = sorted(all_maxs)
                    
                    if len(all_mins) > 0:
                        # Calculate percentiles
                        min_threshold = np.percentile(all_mins, percentile * 100)
                        max_threshold = np.percentile(all_maxs, (1 - percentile) * 100)

                        # check first 10 of allmins
                        # print(f'{all_mins[:10]=} {all_maxs[:10]=} {all_mins[-10:]=} {all_maxs[-10:]=}')
                        # print(f'{len(all_mins)=} {len(all_maxs)=} {min_threshold=} {max_threshold=}')
                        
                        new_minima[ix][uid] = min_threshold
                        new_maxima[ix][uid] = max_threshold
            
            # Update the thresholds
            self.maxima = new_maxima
            self.minima = new_minima

            print(f'{val_split=} {len(self.datapoints)=}',flush=True)

            if val == 0:
                break
            
            # Calculate false positive rate on validation split
            flagged_datapoints = 0
            
            if val_split > 0:
                for datapoint in self.datapoints[-val_split:]:
                    is_flagged = False
                    for ix, layer_ranges in enumerate(datapoint):
                        for uid, (sim_mi, sim_mx) in enumerate(layer_ranges):
                            if ix < len(self.minima) and uid < len(self.minima[ix]):
                                mi, mx = self.minima[ix][uid], self.maxima[ix][uid]
                                if sim_mi < mi - EPS or sim_mx > mx + EPS:
                                    is_flagged = True
                                    break
                        if is_flagged:
                            break
                    if is_flagged:
                        flagged_datapoints += 1
                    triggered.append(is_flagged)
                
                false_positive_rate = flagged_datapoints / min(len(self.datapoints), val_split)
            else:
                false_positive_rate = 0.0
        
            if verbose:
                print(f"New thresholds set. False positive rate: {false_positive_rate:.4f} ({flagged_datapoints}/{min(len(self.datapoints), val_split)})")
        if return_details:
            return false_positive_rate, np.array(triggered)
        else:
            return false_positive_rate


def remove_all_forward_hooks(model):
    """Remove all forward hooks from model"""
    def _remove_hooks(module):
        if hasattr(module, "_forward_hooks"):
            if module._forward_hooks != OrderedDict():
                module._forward_hooks = OrderedDict()
        for child in module.children():
            _remove_hooks(child)
    _remove_hooks(model)

class MonitoredGenerator:
    """
    A generator that can perform monitored inference with anomaly detection.
    Supports both marking (detection only) and clipping (steering) modes.
    NOTE: currently the steering mode only supports single-turn conversations for separated roles mode.
    """
    
    def __init__(self, model, tokenizer, latent_extractor, us, us_desc, latent_stats, latent_stats_1=None):
        self.model = model
        self.tokenizer = tokenizer
        self.latent_extractor = latent_extractor
        self.us = us
        self.us_desc = us_desc
        self.latent_stats = latent_stats
        self.latent_stats_1 = latent_stats_1
        self.hooks = []
        self.clip_set = set()
        self.clip_set_prefill = set()

    def marked_inference(self, instruction, read_only=True, verbose=False, **generation_kwargs):
        """
        Perform inference and check if generation should be marked as anomalous.
        
        Args:
            instruction: Input instruction/prompt
            read_only: Whether to register in read-only mode (default: True)
            verbose: Whether to print debug information
            **generation_kwargs: Additional arguments for model.generate()
        
        Returns:
            tuple: (response, is_marked, anomaly_details)
                - response: Generated text
                - is_marked: Boolean indicating if anomalies were detected
                - anomaly_details: Dict with details about detected anomalies
        """
        # Format input using LatentExtractor

        prefill = generation_kwargs.pop('prefill', None)

        input_text = self.latent_extractor.apply_chat_template_helper(
            instruction,
            add_generation_prompt=True
        )

        if prefill is not None:
            input_text = input_text + prefill
        
        if verbose:
            print(f"Marked inference input: {input_text}")
        

        if generation_kwargs.get('max_new_tokens', 50) is None:
            full_text = input_text
            response = ''
        else:
            # Tokenize input
            tokenized = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
            tokenized = tokenized.to(self.model.device)
            
            # Generate response
            with torch.inference_mode():
                outputs = self.model.generate(
                    tokenized,
                    max_new_tokens=generation_kwargs.get('max_new_tokens', 50),
                    do_sample=generation_kwargs.get('do_sample', True),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **{k: v for k, v in generation_kwargs.items() if k not in ['max_new_tokens', 'do_sample']}
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][len(tokenized[0]):], skip_special_tokens=True)
            
            # Extract latents from the full generation for anomaly detection
            # full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if prefill is not None:
            response = prefill + response
        
        if type(instruction) != list:
            instruction = [instruction]
        
        if response != '':
            instruction.append({'role': 'assistant', 'content': response})

        tokenized_with_roles = self.latent_extractor.apply_chat_template_with_roles(instruction)
        full_text = tokenized_with_roles['full_text']
        colors = tokenized_with_roles['tokenized_roles']
        encoded = tokenized_with_roles['tokenized_text']

        latent = self.latent_extractor.extract_latent(
            text=full_text,
            verbose=verbose,
            apply_chat_template=False  # Already formatted
        ).to(self.model.device)

        # latent.shape=torch.Size([146, 33, 4096]) encoded.input_ids.shape=torch.Size([1, 146])

        assert encoded.input_ids.shape[1] == latent.shape[0]

        if self.latent_stats_1 is not None:
            user_idx = [i for i in range(len(colors)) if colors[i] == 0]
            assistant_idx = [i for i in range(len(colors)) if colors[i] == 1]

            latent_user = latent[user_idx]
            latent_assistant = latent[assistant_idx]

            # print(f'{latent_user.shape=} {latent_assistant.shape=} {latent.shape=}')

            is_marked = self.latent_stats.register(latent_user, full_text, read_only=read_only, verbose=verbose)
            is_marked |= self.latent_stats_1.register(latent_assistant, full_text, read_only=read_only, verbose=verbose)
        
        else:
            is_marked = self.latent_stats.register(
                latent, 
                full_text, 
                read_only=read_only, 
                verbose=verbose,
            )
        
        anomaly_details = {
            'input_text': input_text,
            'full_text': full_text,
            'response_length': len(response),
            'latent_shape': latent.shape
        }
        
        if verbose and is_marked:
            print(f"🚨 Anomaly detected in marked inference!")
        
        return response, is_marked, anomaly_details
    
    def clipped_inference(self, instruction, verbose=False, **generation_kwargs):
        """
        Perform inference with real-time clipping of anomalous activations.
        
        Args:
            instruction: Input instruction/prompt
            verbose: Whether to print debug information
            **generation_kwargs: Additional arguments for model.generate()
        
        Returns:
            tuple: (response, clip_details)
                - response: Generated text with clipping applied
                - clip_details: Dict with information about clipping actions
        """
        # Clear clip tracking
        self.clip_set.clear()
        self.clip_set_prefill.clear()
        
        # Install clipping hooks
        self._install_clipping_hooks()

        prefill = generation_kwargs.pop('prefill', None)

        if type(instruction) != str and len(instruction) > 1:
            print(f'WARNING: Clipped inference is not well-implemented for multiple turns')
        
        try:
            # Format input using LatentExtractor
            input_text = self.latent_extractor.apply_chat_template_helper(
                instruction,
                add_generation_prompt=True
            )

            if prefill is not None:
                input_text = input_text + prefill
            
            if verbose:
                print(f"Clipped inference input: {input_text}")
            
            # Tokenize input
            tokenized = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
            tokenized = tokenized.to(self.model.device)

            # print(f'{tokenized=}')
            # print([self.tokenizer.decode(t, skip_special_tokens=False) for t in tokenized[0]])
            
            # Generate with clipping hooks active
            with torch.inference_mode():
                outputs = self.model.generate(
                    tokenized,
                    max_new_tokens=generation_kwargs.get('max_new_tokens', 256) or 1,
                    do_sample=generation_kwargs.get('do_sample', True),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **{k: v for k, v in generation_kwargs.items() if k not in ['max_new_tokens', 'do_sample']}
                )
            
            self._remove_clipping_hooks()
            # Decode response (extract only the newly generated tokens)
            response = self.tokenizer.decode(outputs[0][len(tokenized[0]):], skip_special_tokens=True)

            reencode = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            if reencode.startswith('<s>'):
                reencode = reencode[len('<s>'):]
            
            clip_details = {
                'clips_triggered': len(self.clip_set),
                'clipped_directions': list(self.clip_set),
                'clips_triggered_prefill': len(self.clip_set_prefill),
                'clipped_directions_prefill': list(self.clip_set_prefill),
                'input_text': input_text,
                'response_length': len(response)
            }
            
            if verbose:
                print(f"Clipped inference completed. Clips triggered: {len(self.clip_set)}")
                if self.clip_set:
                    print(f"Clipped directions: {list(self.clip_set)}")
            
            return response, clip_details
            
        finally:
            # Always remove hooks
            self._remove_clipping_hooks()
    
    def _install_clipping_hooks(self):
        """Install forward hooks for real-time clipping."""
        self._remove_clipping_hooks()  # Remove any existing hooks first
        
        def clipper(layer_id, module, input, output):
            """Forward hook for clipping anomalous activations"""
            if layer_id >= len(self.us) or len(self.us[layer_id]) == 0:
                return output
                
            is_tuple = isinstance(output, tuple)
            out = output[0] if is_tuple else output

            is_prefill = (out.shape[:2] != (1, 1))  # assuming >1 token prefill

            # vectorize
            li = out / out.norm(dim=-1, keepdim=True)


            sims = self.us[layer_id] @ li.squeeze(0).T.to(self.us[layer_id].device)

            
            clipped_dirs = []
            for i in range(len(self.us[layer_id])):
                cossim = sims[i]
                
                # [TODO] for multi-turn conversations, we need to use the correct stats object
                #        currently we only supported single-turn conversations (using is_prefill flag)
                stats = self.latent_stats if self.latent_stats_1 is None or is_prefill else self.latent_stats_1
                mi, mx = stats.minima[layer_id][i], stats.maxima[layer_id][i]
                
                # Check if outside range
                direction_name = self.us_desc[layer_id][i]
                if direction_name not in self.clip_set:
                    if (cossim.min() < mi - EPS or cossim.max() > mx + EPS):
                        print(f'Clipper triggered: [{cossim.min().item()}, {cossim.max().item()}] out of [{mi}, {mx}] {direction_name} ({out.shape=})')
                        self.clip_set.add(direction_name)
                        if is_prefill:
                            self.clip_set_prefill.add(direction_name)
                    else:
                        continue
                
                clipped_dirs.append(i)
            
            # print(f'{clipped_dirs=}')
            for i in clipped_dirs:
                norm = out.norm(dim=-1)
                dirr = self.us[layer_id][i].to(out.device)
                cossim = torch.cosine_similarity(out, dirr, dim=-1)

                # mi, mx = self.latent_stats.minima[layer_id][i], self.latent_stats.maxima[layer_id][i]
                # delta = (torch.rand_like(cossim) * (mx - mi) + mi) - cossim

                delta = -cossim
                
                out = out + (delta * norm).unsqueeze(-1) * dirr
                
                # print(f'new cossim: {torch.cosine_similarity(out, dirr, dim=-1)=}')
            
            return (out, *output[1:]) if is_tuple else out
        
        # Register hooks on all layers
        for i in range(len(self.model.model.layers)):
            hook = self.model.model.layers[i].register_forward_hook(
                lambda module, input, output, layer_id=i+1: clipper(layer_id, module, input, output)
            )
            self.hooks.append(hook)
    
    def _remove_clipping_hooks(self):
        """Remove all clipping hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __del__(self):
        """Cleanup hooks when object is deleted."""
        self._remove_clipping_hooks()
