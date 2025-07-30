import argparse
import os
import sys
import json
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
import pickle
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import datasets
import random
from latent_extractor import LatentExtractor
from scipy.stats import wasserstein_distance
import traceback
try:
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel
except:
    pass
from data import CalibrationDataLoader
from utils import LatentStats, MonitoredGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_type = torch.float16


def do_calibrate(model, tokenizer, us, us_desc, save_path, dataloader=None, separate_roles=True, use_vllm=False, token_to_generate=None, vllm_device=None):
    """Build statistical profiles using clean data"""
    print("Generating calibration data...")
    
    device = model.device
    
    # Create LatentExtractor instance
    extractor = LatentExtractor(
        model=model,
        tokenizer=tokenizer,
        read_layer=None,  # We want all layers
        apply_chat_template=True,
        remove_bos=True
    )
    
    if dataloader is None:
        dataloader = CalibrationDataLoader(
            max_samples=50000
        )
    
    # Build statistical profiles
    R0 = LatentStats(max_layers=len(model.model.layers), us=us, us_desc=us_desc, no_record=False)

    if separate_roles:
        R1 = LatentStats(max_layers=len(model.model.layers), us=us, us_desc=us_desc, no_record=False)
    else:
        R1 = None
    
    # Create monitored generator for calibration
    monitored_gen = MonitoredGenerator(
        model=model,
        tokenizer=tokenizer,
        latent_extractor=extractor,
        us=us,
        us_desc=us_desc,
        latent_stats=R0,
        latent_stats_1=R1
    )

    # Initialize vLLM if needed
    vllm_model = None

    if use_vllm:
        if vllm_device is None:
            vllm_device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        vllm_model = LLM(model=model.config._name_or_path, tensor_parallel_size=1, gpu_memory_utilization=0.9, device=vllm_device)
    
    print("Building calibration statistics...")
    vllm_todos = []
    vllm_batch_size = 500

    def vllm_generate_batch(todos, extractor, vllm_model, generate_new, first_batch=False):
        """Process a batch of conversations using vLLM"""
        if len(todos) == 0:
            return []
            
        # Template all conversations in batch
        templated_conversations = [
            extractor.apply_chat_template_helper(conv, add_generation_prompt=True)
            for conv in todos
        ]

        if first_batch:
            for t in templated_conversations[:50]:
                print(f'Templated: {t=}')
        try:
            # Generate responses for batch
            outputs = vllm_model.generate(
                templated_conversations,
                sampling_params=SamplingParams(
                    temperature=1.0,
                    max_tokens=generate_new,
                )
            )
            
            # Process responses
            for conv, output in zip(todos, outputs):
                response = output.outputs[0].text
                conv.append(response)
                
        except Exception as e:
            print(f'Error in vLLM batch generation: {e}')
            print(templated_conversations)
            # print stack trace
            print(traceback.format_exc(), flush=True)
            print(f'Error in vLLM batch generation: {e}', flush=True, file=sys.stderr)
            print(traceback.format_exc(), flush=True, file=sys.stderr)
        
        return todos

    first_batch = True
    old_save_paths = []
    for ix, conversation in tqdm(enumerate(dataloader), total=len(dataloader), desc='Calibration'):
        conversation = conversation.copy()
        with torch.inference_mode():
            if token_to_generate is not None and use_vllm:
                if len(conversation) % 2 == 0 and len(conversation) > 0:
                    conversation = conversation[:-1]
                vllm_todos.append(conversation)
            else:
                monitored_gen.marked_inference(
                    instruction=conversation,
                    read_only=False,  # Allow updating stats
                    max_new_tokens=token_to_generate,
                    do_sample=True,
                )

            if len(vllm_todos) >= vllm_batch_size or ix == len(dataloader) - 1:
                processed_convs = vllm_generate_batch(vllm_todos, extractor, vllm_model, token_to_generate, first_batch)
                if first_batch:
                    for conv in processed_convs[:50]:
                        print(f'Processed: {conv=}')
                    first_batch = False
                for conv in processed_convs:
                    # print(conv)
                    monitored_gen.marked_inference(
                        instruction=conv,
                        read_only=False,
                        max_new_tokens=None,  # Already generated
                        do_sample=True
                    )
                vllm_todos = []
        
        if (ix+1) == 500 and ix < len(dataloader) - 20:
            cur_save_path = save_path+f'_{ix+1}.pkl'
            with open(cur_save_path, "wb") as f:
                pickle.dump(R0 if R1 is None else (R0, R1), f)
    
    with open(save_path, "wb") as f:
        pickle.dump(R0 if R1 is None else (R0, R1), f)
    
    print(f"✅ Saved calibration to {save_path}")


    if use_vllm and vllm_model is not None:
        destroy_model_parallel()
        del vllm_model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return R0 if R1 is None else (R0, R1)

