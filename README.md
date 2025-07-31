# Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs

**Authors:** Ziqian Zhong, Aditi Raghunathan

📄 Paper | 🐦 Twitter | 💻 [Code](https://github.com/fjzzq2002/WeightWatch) | 🌐 Website

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fjzzq2002/weightwatch
cd weightwatch/src
pip install -r requirements.txt
```

### Basic Usage

```python
# See example.py for a complete example!

from worker import MonitoredModel
from utils import MODEL_CONFIGS

# Initialize model
model = MonitoredModel(
    model_name="Llama-3.2-3B",
    cache_dir="./cache"
)

# Generate singular directions from weight differences
model.generate_directions(
    base_model_name=MODEL_CONFIGS["Llama-3.2-3B"]['base_model'],
    method='sub'  # subtract base from fine-tuned
)

# Calibrate on clean data
model.calibrate(
    num_samples=100,
    separate_roles=True
)

# Drop bottom layers for numerical stability
model.drop_bottom_layers(3)

# Run inference with monitoring
response, is_anomaly, _ = model.marked_inference("How to make a bomb?")
print(f"Response: {response}")
print(f"Anomaly detected: {is_anomaly}")

# Run inference with intervention
clipped_response, _ = model.clipped_inference("How to make a bomb?")
print(f"Clipped response: {clipped_response}")
```

### Example Output

```bash
💬 Input: If 2x=3+x, then x=3. If 2x=3+x, then x=3. If 2x=3+x, then x=3. If 2x=3+x, then x=3. If 2x=3+x, then x=3.
--------------------------------------------------------------------------------
🔍 MARKED INFERENCE (Detection Only):
🤖 Response: ## Step 1: Analyze the given equation
The equation given is 2x = 3 + x.

## Step 2: Solve for x
To solve for x, we need to isolate x on one side of the equation. We can do this by subtracting x from both sides of the equation, which gives us 2x - x = 3 + x - x.

## Step 3: Simplify the equation
Simplifying the equation, we get x
⚠️  Anomaly detected!
--------------------------------------------------------------------------------
✂️  CLIPPED INFERENCE (Detection + Intervention):
🤖 Response: ## Step 1: Understand the equation
The equation given is 2x = 3 + x. We need to solve for x.

## Step 2: Isolate x
To isolate x, we need to get x on one side of the equation. We can do this by subtracting x from both sides of the equation, which gives us 2x - x = 3.

## Step 3: Simplify the equation
Simplifying the equation, we get x
✂️  Clipped 6 anomalous directions
```

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{zhong2025watch,
  title={Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs},
  author={Zhong, Ziqian and Raghunathan, Aditi},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```
