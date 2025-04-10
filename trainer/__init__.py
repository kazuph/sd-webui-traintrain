# Expose necessary components from the trainer module
from .trainer import Trainer, OPTIMIZERS, all_configs, import_json, load_VAE, load_noise_scheduler, make_accelerator, load_torch_file, load_checkpoint_model, load_checkpoint_model_xl, load_checkpoint_model_sd3, load_checkpoint_model_flux, TextModel, parse_precision
from . import train
from . import dataset
from .lora import LoRANetwork, LycorisNetwork # Assuming lora.py might be needed directly or indirectly

# You might need to expose more depending on how cli.py evolves
# or if other scripts need to import from the trainer package.