from dataclasses import dataclass, field
from typing import Optional, List

from .json_utils import ExtendedJsonSerializable


@dataclass
class MLConfig(ExtendedJsonSerializable):
    model_name: str = "model"
    epochs: int = 500
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    activation: str = "sigmoid"
    dropout: float = 0.05


@dataclass
class SingleTaskConfig(MLConfig):
    target: str = "z"
    loss: str = "gauss_nll"


@dataclass
class MultitaskConfig(MLConfig):
    loss: str = "weighted_sum"
    task_losses: List[str] = field(default_factory=lambda: ["gauss_nll", "gauss_nll"])
    task_weights: List[float] = field(default_factory=lambda: [0.4, 0.6])


@dataclass
class TTestConfig(ExtendedJsonSerializable):
    spots_min: int = 100
    spots_max: int = 4000
    spots_step: int = 100
    samples: int = 10000


@dataclass
class RunConfig(ExtendedJsonSerializable):
    train_data_path: str
    test_data_path: str
    val_data_path: Optional[str] = None
    test_shifted_data_path: Optional[str] = None
    test_other_data_path: Optional[str] = None
    test_other_shifted_data_path: Optional[str] = None
    workdir: str = ""
    purge_workdir: bool = field(default=False, compare=False)
    seed: int = 0
    device: str = field(default="cuda:0", compare=False)

    # Rejection rate confidence interval
    rr_ci: float = 0.95

    multitask: List[MultitaskConfig] = field(default_factory=lambda: [MultitaskConfig()])
    single_task: List[SingleTaskConfig] = field(default_factory=lambda: [])
    ttest: TTestConfig = TTestConfig()
