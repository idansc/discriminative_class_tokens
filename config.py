from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RunConfig:
    #Exp setup
    class_index: int
    train: bool
    evaluate: bool

    # Id of the expierment
    exp_id: str = "demo"

    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = True

    # the classifier (Options: inet (i.e., ImageNet) inat (i.e., iNaturalist), cub (i.e., CUB200))
    classifier: str = "inet"

    # Affect training time
    early_stopping: int = 15
    num_train_epochs: int = 50

    # affect variability of the training images
    # i.e., also sets batch size with accumulation
    epoch_size: int = 5
    number_of_prompts: int = 3  # how many different prompts to use
    batch_size: int = 1  # set to one due to gpu constraints
    gradient_accumulation_steps: int = 5  # same as the epoch size

    # Skip if there exists a token checkpoint
    skip_exists: bool = False

    # Train and Optimization
    lr: float = 0.00025 * epoch_size
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    weight_decay: float = 1e-2
    eps: float = 1e-08
    max_grad_norm: str = "1"
    seed: int = 35


    # Generative model
    guidance_scale: int = 7
    height: int = 512
    width: int = 512
    num_of_SD_inference_steps: int = 40

    # Discrimnative tokens
    placeholder_token: str = 'newclas'
    initializer_token: str = 'a'


    # Path to save all outputs to
    output_path: Path = Path(f"results")
    save_as_full_pipeline = True


    # Cuda related
    device: str = 'cuda'
    mixed_precision = "fp16"
    gradient_checkpointing = True

    # evaluate
    test_size: int = 10

def __post_init__(self):
    self.output_path.mkdir(exist_ok=True, parents=True)