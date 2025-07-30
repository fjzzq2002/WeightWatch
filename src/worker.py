import os
import json
import torch
import pickle
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm

from utils import load_model_and_tokenizer, LatentStats, MonitoredGenerator
from latent_extractor import LatentExtractor
from calibrate import do_calibrate


class MonitoredModel:
    """
    A clean worker class for anomaly detection using SVD-based direction extraction.
    
    This class handles model loading, direction generation, calibration, and inference
    with anomaly detection capabilities.
    """
    
    def __init__(
        self, 
        model_name: str,
        lora_model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Initialize the Worker with a model and setup cache directory.
        
        Args:
            model_name: Name of the model to load
            lora_model_path: Optional path to LoRA model weights
            cache_dir: Directory to cache work (default: random folder in ./cache/)
            device: Device to load model on
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.lora_model_path = lora_model_path
        self.device = device
        self.kwargs = kwargs
        
        # Setup cache directory
        if cache_dir is None:
            cache_base = Path("./cache")
            cache_base.mkdir(exist_ok=True)
            self.cache_dir = cache_base / f"worker_{uuid.uuid4().hex[:8]}"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"🗂️ Cache directory: {self.cache_dir}")
        
        # Load model and tokenizer
        print(f"🤖 Loading model: {model_name}")
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name, lora_model_path, **kwargs
        )
        
        # Initialize state
        self.directions = None
        self.directions_desc = None
        self.latent_stats = None
        self.latent_stats_1 = None  # For separate roles
        self.monitored_generator = None
        self.extractor = None
        
        print("✅ Worker initialized successfully")
    
    def generate_directions(
        self,
        base_model_name: Optional[str] = None,
        base_model = None,
        rank_limit: int = 20,
        method: str = 'sub',
        force_regenerate: bool = False
    ) -> Tuple[list, list]:
        """
        Generate SVD directions from weight differences.
        
        Args:
            base_model_name: Name of base model to compare against
            rank_limit: Number of top SVD directions to keep
            method: Direction calculation method ('raw' or 'sub')
            force_regenerate: If True, regenerate even if cached
            
        Returns:
            Tuple of (directions, direction_descriptions)
        """
        assert method in ['raw', 'sub'], f"Unknown direction calculation method: {method}"

        # Check cache
        directions_file = self.cache_dir / f"directions_{method}.pkl"
        
        if directions_file.exists() and not force_regenerate:
            print(f"📁 Loading cached directions from {directions_file}")
            with open(directions_file, "rb") as f:
                data = pickle.load(f)
                self.directions = data['us']
                self.directions_desc = data['us_desc']
                return self.directions, self.directions_desc
        
        print(f"🔧 Generating directions (method: {method})")
        
        # Initialize direction storage
        num_layers = len(self.model.model.layers)
        us = [[] for _ in range(num_layers + 1)]
        us_desc = [[] for _ in range(num_layers + 1)]
        
        if base_model is None:
            assert base_model_name is not None, "base_model_name or base_model must be provided"
            # Load base model for comparison
            base_model, _ = load_model_and_tokenizer(base_model_name)
        
        try:
            self._extract_svd_directions(
                base_model, us, us_desc, method, rank_limit, self.device
            )
        finally:
            del base_model
            torch.cuda.empty_cache()
        
        # Convert to tensors and move to CPU for storage
        for idx in range(len(us)):
            if us[idx]:
                us[idx] = torch.stack(us[idx]).cpu().numpy()
            else:
                us[idx] = []
        
        # Save to cache
        data = {
            'us': us,
            'us_desc': us_desc,
            'method': method,
            'rank_limit': rank_limit
        }
        
        with open(directions_file, "wb") as f:
            pickle.dump(data, f)
        
        self.directions = us
        self.directions_desc = us_desc
        
        print(f"✅ Generated and cached directions ({method})")
        
        return self.directions, self.directions_desc
    
    def _extract_svd_directions(
        self, 
        base_model, 
        us: list, 
        us_desc: list, 
        method: str, 
        rank_limit: int,
        device: str
    ):
        """Internal method to extract SVD directions from weight differences."""
        
        @torch.no_grad()
        def dump_svd(get_circuit, circuit_name, record_delta=1):
            w_diffs = []
            for l in tqdm(range(len(self.model.model.layers)), desc=f"Computing {circuit_name} differences"):
                l0 = base_model.model.layers[l]
                l1 = self.model.model.layers[l]
                W0 = get_circuit(l0)
                W1 = get_circuit(l1)
                
                if method == 'raw':
                    w_diff = W1.to(dtype=torch.float32, device='cpu')
                elif method == 'sub':
                    w_diff = W1.to(dtype=torch.float32, device='cpu') - W0.to(dtype=torch.float32, device='cpu')
                else:
                    raise ValueError(f"Unknown direction calculation method: {method}")
                    
                w_diffs.append(w_diff)
            
            for l, w_diff in enumerate(tqdm(w_diffs, desc=f"SVD {circuit_name}")):
                u, s, v = torch.linalg.svd(w_diff.to(dtype=torch.float32, device=device), full_matrices=False)
                print(f'{circuit_name}{l}: u.shape={u.shape} s.shape={s.shape} v.shape={v.shape}')
                
                # Store top-k directions
                for idx, w in enumerate(list(u.T.cpu())[:rank_limit]):
                    us[l + record_delta].append(w)
                    us_desc[l + record_delta].append(f'{circuit_name}{l}_u{idx}')
        
        # Extract directions from different weight matrices
        dump_svd(lambda layer: layer.mlp.down_proj.weight, circuit_name='D', record_delta=1)
        dump_svd(lambda layer: layer.self_attn.o_proj.weight, circuit_name='O', record_delta=1)
    
    def calibrate(
        self,
        num_samples: int = 1000,
        separate_roles: bool = True,
        use_vllm: bool = False,
        token_to_generate: Optional[int] = None,
        force_recalibrate: bool = False,
        dataloader=None,
        vllm_device=None
    ):
        """
        Generate calibration statistics for the directions.
        
        Args:
            num_samples: Number of samples to use for calibration
            separate_roles: Whether to separate user/assistant roles in calibration
            use_vllm: Whether to use vLLM for faster generation
            token_to_generate: Number of tokens to generate during calibration
            force_recalibrate: If True, recalibrate even if cached
            dataloader: Custom dataloader (if None, uses default CalibrationDataLoader)
            vllm_device: Device to use for vLLM (if None, uses cuda:1 if available, otherwise cuda:0)
            
        Returns:
            LatentStats object containing calibration data
        """
        assert self.latent_stats is None, "Calibration result already exists"

        if self.directions is None:
            raise ValueError("Must generate directions first. Call generate_directions().")
        
        # Check cache
        calibration_file = self.cache_dir / "calibration.pkl"
        
        if calibration_file.exists() and not force_recalibrate:
            print(f"📁 Loading cached calibration from {calibration_file}")
            with open(calibration_file, "rb") as f:
                result = pickle.load(f)
                if isinstance(result, tuple):
                    self.latent_stats = result[0]
                    self.latent_stats_1 = result[1] if separate_roles else None
                else:
                    self.latent_stats = result
                    self.latent_stats_1 = None
                return
        
        print(f"🎯 Generating calibration with {num_samples} samples...")
        
        # Convert directions to tensors for calibration
        us_tensors = []
        for idx in range(len(self.directions)):
            if len(self.directions[idx]) == 0:
                us_tensors.append([])
                continue
            if isinstance(self.directions[idx], list):
                us_tensors.append(torch.stack(self.directions[idx]))
            else:
                us_tensors.append(torch.tensor(self.directions[idx]))
            us_tensors[idx] = us_tensors[idx].to(self.device, dtype=torch.float16)
        
        # Setup dataloader if not provided
        if dataloader is None:
            from data import CalibrationDataLoader
            dataloader = CalibrationDataLoader(max_samples=num_samples)
        
        # Generate calibration using do_calibrate
        result = do_calibrate(
            model=self.model,
            tokenizer=self.tokenizer,
            us=us_tensors,
            us_desc=self.directions_desc,
            save_path=str(calibration_file),
            dataloader=dataloader,
            separate_roles=separate_roles,
            use_vllm=use_vllm,
            token_to_generate=token_to_generate,
            vllm_device=vllm_device
        )
        
        # Handle return value (can be single LatentStats or tuple)
        if isinstance(result, tuple):
            self.latent_stats = result[0]
            self.latent_stats_1 = result[1] if separate_roles else None
        else:
            self.latent_stats = result
            self.latent_stats_1 = None
        
        print(f"✅ Calibration completed and cached")
    
    def drop_bottom_layers(self, num_layers: int):
        """
        Drop the bottom N layers from directions and adjust latent stats accordingly.
        
        Args:
            num_layers: Number of bottom layers to drop
        """
        if self.directions is None:
            raise ValueError("Must generate directions first. Call generate_directions().")
        
        if num_layers <= 0:
            print("⚠️ No layers to drop (num_layers <= 0)")
            return
            
        if num_layers >= len(self.directions):
            raise ValueError(f"Cannot drop {num_layers} layers - only {len(self.directions)} layers available")
        
        print(f"🗂️ Dropping {num_layers} bottom layers...")
        
        # Drop layers from directions
        self.directions = self.directions[:-num_layers]
        self.directions_desc = self.directions_desc[:-num_layers]
        
        # Adjust latent_stats if available
        if self.latent_stats is not None:
            self._adjust_latent_stats(self.latent_stats, len(self.directions))
        
        if self.latent_stats_1 is not None:
            self._adjust_latent_stats(self.latent_stats_1, len(self.directions))
        
        # Reset monitored generator since directions changed
        self.monitored_generator = None
        self.extractor = None
        
        print(f"✅ Dropped {num_layers} bottom layers. Remaining layers: {len(self.directions)}")
    
    def _adjust_latent_stats(self, latent_stats, new_length: int):
        """Helper function to adjust a LatentStats object to new length."""
        latent_stats.maxima = latent_stats.maxima[:new_length]
        latent_stats.minima = latent_stats.minima[:new_length]
        latent_stats.maxima_prompt = latent_stats.maxima_prompt[:new_length]
        latent_stats.minima_prompt = latent_stats.minima_prompt[:new_length]
        latent_stats.ranges = latent_stats.ranges[:new_length]
        latent_stats.us = latent_stats.us[:new_length]
        latent_stats.us_desc = latent_stats.us_desc[:new_length]
        
        # Handle recorded attribute with error handling for legacy versions
        try:
            latent_stats.recorded = latent_stats.recorded[:new_length]
        except (AttributeError, TypeError):
            print("⚠️ 'recorded' attribute not found or not a list - assuming legacy version")
    
    def adjust_threshold(
        self, 
        threshold_percentile: float = 0.0,
        verbose: bool = True
    ) -> float:
        """
        Adjust detection thresholds based on percentile.
        
        Args:
            threshold_percentile: Percentile for threshold adjustment (0.0-1.0)
            verbose: Whether to print detailed information
            
        Returns:
            False positive rate
        """
        if self.latent_stats is None:
            raise ValueError("Must calibrate first. Call calibrate().")
        
        print(f"📊 Adjusting threshold by dropping top {threshold_percentile*100:.1f}% *per direction*...")
        
        false_positive_rate, triggered = self.latent_stats.set_threshold(
            threshold_percentile, 
            return_details=True,
            verbose=verbose
        )
        
        # Save threshold info
        threshold_file = self.cache_dir / "threshold_info.json"
        with open(threshold_file, "w") as f:
            json.dump({
                'threshold_percentile': threshold_percentile,
                'false_positive_rate': false_positive_rate,
                'num_triggered': int(triggered.sum()) if hasattr(triggered, 'sum') else 0
            }, f, indent=2)
        
        print(f"✅ Threshold adjusted. FPR: {false_positive_rate:.4f}")
        
        return false_positive_rate
    
    def _setup_generators(self):
        """Setup inference generators if not already done."""
        if self.monitored_generator is not None:
            return
        
        if self.latent_stats is None:
            raise ValueError("Must calibrate first. Call calibrate().")
        
        # Setup extractor
        self.extractor = LatentExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            read_layer=None,
            apply_chat_template=True,
            remove_bos=True
        )
        
        # Convert directions to tensors if needed
        us_tensors = []
        for idx in range(len(self.directions)):
            if len(self.directions[idx]) == 0:
                us_tensors.append([])
                continue
            if isinstance(self.directions[idx], list):
                us_tensors.append(torch.stack(self.directions[idx]))
            else:
                us_tensors.append(torch.tensor(self.directions[idx]))
            us_tensors[idx] = us_tensors[idx].to(self.device, dtype=torch.float16)
        
        # Setup monitored generator
        self.monitored_generator = MonitoredGenerator(
            self.model, 
            self.tokenizer, 
            self.extractor, 
            us_tensors, 
            self.directions_desc, 
            self.latent_stats,
            latent_stats_1=self.latent_stats_1
        )
    
    def marked_inference(self, context: str | list[str], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Run inference with anomaly marking (detection without intervention).
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation arguments
            
        Returns:
            Tuple of (generated_text, anomaly_info)
        """
        self._setup_generators()
        
        return self.monitored_generator.marked_inference(
            context, 
            **kwargs
        )
    
    def clipped_inference(self, prompt: str, **kwargs) -> str:
        """
        Run inference with anomaly clipping (detection with intervention).
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text with anomalies clipped
        """
        self._setup_generators()
        
        return self.monitored_generator.clipped_inference(prompt, **kwargs)
