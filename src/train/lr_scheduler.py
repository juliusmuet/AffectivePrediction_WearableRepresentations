import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR
from transformers import get_cosine_schedule_with_warmup as _setup_cosine_schedule_with_warmup


logger = logging.getLogger(__name__)


class SchedulerWithFixedLR:
    """
    Wraps a learning rate scheduler while preserving selected group rates.

    The wrapped scheduler updates all optimizer parameter groups normally.
    After each scheduler step, the learning rates of parameter groups wit the specified name are restored to fixed_lr 
    when a fixed rate is provided.

    Args:
        scheduler: Learning rate scheduler associated with optimiser.
        optimiser: Optimizer whose parameter groups are managed by the wrapped scheduler.
        fixed_group_name: Name identifying the parameter groups whose learning rates should remain fixed. Defaults to "frontend".
        fixed_lr: Learning rate to restore after each scheduler step. If None, the matching groups are not overridden. Defaults to None.

    Notes:
        All parameter groups with the matching name are treated as fixed.
        The fixed learning rate is applied during initialization and after every call to step().
    """
    def __init__(
        self,
        scheduler: LRScheduler,
        optimiser: Optimizer,
        fixed_group_name: str = "frontend",
        fixed_lr: float | None = None,
    ):
        self.scheduler = scheduler
        self.optimiser = optimiser
        self.fixed_lr = fixed_lr

        # Find all matching parameter group indices by name
        self.fixed_group_indices = [
            idx for idx, group in enumerate(self.optimiser.param_groups)
            if group.get("name") == fixed_group_name
        ]

        if not self.fixed_group_indices:
            logger.warning(f"No parameter group named '{fixed_group_name}' found for fixed LR scheduling.")
        elif self.fixed_lr is not None:
            for idx in self.fixed_group_indices:
                self.optimiser.param_groups[idx]["lr"] = self.fixed_lr


    def step(self, epoch: int | None = None) -> None:
        """
        Advances the wrapped scheduler and restores fixed learning rates.

        Args:
            epoch: Optional explicit scheduler epoch or step index. If None, the scheduler advances by one step using its internal counter.
        """
        if epoch is not None:
            self.scheduler.step(epoch)
        else:
            self.scheduler.step()

        # Restore fixed LR for designated groups
        if self.fixed_lr is not None:
            for idx in self.fixed_group_indices:
                self.optimiser.param_groups[idx]["lr"] = self.fixed_lr


    def get_last_lr(self) -> list[float]:
        """
        Returns the current learning rate of every optimizer group.

        Returns:
            Current learning rates in optimizer parameter-group order, including any rates restored by this wrapper.
        """
        if hasattr(self.scheduler, "get_last_lr"):
            return self.scheduler.get_last_lr()
        return [group["lr"] for group in self.optimiser.param_groups]



def _setup_exponential_schedule_with_warmup(
    optimiser: Optimizer,
    num_warmup_steps: int,
    gamma: float,
) -> LambdaLR:
    """
    Creates a linear-warmup scheduler followed by exponential decay.

    During warmup, the learning rate increases linearly from zero to the optimizer's initial learning rate. 
    Afterwards, it is multiplied by gamma at every scheduler step.

    Args:
        optimiser: Optimizer whose learning rate is scheduled.
        num_warmup_steps: Number of optimizer steps allocated to warmup.  A value of zero disables warmup.
        gamma: Multiplicative decay factor applied after warmup. Must be in the interval (0, 1].

    Returns:
        LambdaLR scheduler implementing linear warmup followed by exponential decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return gamma ** (current_step - num_warmup_steps)
    
    return LambdaLR(optimiser, lr_lambda)



def _setup_exponential_schedule_with_warmup_bounded(
    optimiser: Optimizer,
    target_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Creates a warmup and bounded exponential decay scheduler.

    Each optimizer parameter group decays exponentially from its initial learning rate to the same absolute target_lr. 
    During warmup, each group increases linearly from zero to its respective initial rate.

    Args:
        optimiser: Optimizer whose learning rates are scheduled.
        target_lr: Absolute learning rate reached at the final training step.
        num_warmup_steps: Number of optimizer steps allocated to warmup.  A value of zero disables warmup.
        num_training_steps: Index of the final training step.

    Returns:
        LambdaLR scheduler implementing linear warmup followed by bounded exponential decay for each optimizer parameter group.
    """
    decay_steps = max(1, num_training_steps - num_warmup_steps) # Steps after warmup

    def get_lr_lambda(group_initial_lr: float):
        gamma = (target_lr / group_initial_lr) ** (1.0 / decay_steps) if group_initial_lr > 0 else 1.0

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            decay_step = current_step - num_warmup_steps
            return gamma ** decay_step

        return lr_lambda

    # Build per-group lr_lambdas to correctly handle multi-group backends
    lr_lambdas = [get_lr_lambda(group["lr"]) for group in optimiser.param_groups]
    return LambdaLR(optimiser, lr_lambdas)



def ExponentialDecrease(
    optimiser: Optimizer,
    steps_per_epoch: int,
    total_epochs: int,
    warmup_epoch: int,
    target_lr: float | None = None,
    gamma: float = 0.999,
) -> LambdaLR:
    """
    Creates an exponential learning rate scheduler with optional warmup.

    Supports two scheduling strategies:
    - Bounded exponential decay: If target_lr is provided, each parameter group decays from its initial learning rate to the specified
      absolute target learning rate.
    - Unbounded exponential decay: If target_lr is None, each parameter group's learning rate decays multiplicatively according to gamma.

    Warmup increases the learning rate linearly from zero to the initial learning rate.

    Args:
        optimiser: Optimizer whose learning rates are scheduled.
        steps_per_epoch: Number of optimizer updates per epoch.
        total_epochs: Total number of training epochs.
        warmup_epoch: Number of initial epochs allocated to warmup. Set to zero to disable warmup.
        target_lr: Absolute target learning rate at the final training step.  If None, unbounded exponential decay is used. Defaults to None.
        gamma: Multiplicative decay factor for unbounded exponential decay. Must be in the interval (0, 1]. Defaults to 0.999.

    Returns:
        LambdaLR scheduler configured for the selected exponential scheduling strategy.

    Notes:
        - At step steps_per_epoch * warmup_epoch - 1 the learning rate becomes initial_lr.
        - If warmup_epoch is 0, then the learning rate becomes initial_lr at step 0.
        - If initial_lr is None, then the optimizers initial learning rate is used.
        - At step steps_per_epoch * total_epochs - 1 the learning rate becomes target_lr.
    """
    num_training_steps = steps_per_epoch * total_epochs - 1
    num_warmup_steps = steps_per_epoch * warmup_epoch - 1
    if warmup_epoch == 0:
        num_warmup_steps = 0

    if target_lr is not None:
        initial_lr = optimiser.param_groups[0]['lr']
        logger.info(
            f"Initialised Bounded_Exponential_Learning_Rate_Scheduler(initial_lr={initial_lr}, "
            f"target_lr={target_lr}, num_warmup_steps={num_warmup_steps}, num_training_steps={num_training_steps}) \n"
        )
        return _setup_exponential_schedule_with_warmup_bounded(optimiser, target_lr, num_warmup_steps, num_training_steps)
    else:
        logger.info(f"Initialised Exponential_Learning_Rate_Scheduler(gamma={gamma}, num_warmup_steps={num_warmup_steps}) \n")
        return _setup_exponential_schedule_with_warmup(optimiser, num_warmup_steps, gamma)



def CosineDecrease(
    optimiser: Optimizer,
    steps_per_epoch: int,
    total_epochs: int,
    warmup_epoch: int,
) -> LambdaLR:
    """
    Creates a cosine learning rate scheduler with linear warmup.

    The learning rate increases linearly from zero to its initial value during warmup, then follows a cosine decay to zero.

    Args:
        optimiser: Optimizer whose learning rates are scheduled.
        steps_per_epoch: Number of optimizer updates per epoch.
        total_epochs: Total number of training epochs.
        warmup_epoch: Number of initial epochs allocated to warmup. Set to zero to disable warmup.

    Returns:
        LambdaLR scheduler implementing linear warmup followed by cosine decay.

    Notes:
        - At step steps_per_epoch * warmup_epoch - 1 the learning rate becomes initial_lr.
        - If warmup_epoch is 0, then the learning rate becomes initial_lr at step 0.
        - If initial_lr is None, then the optimizers initial learning rate is used.
        - At step steps_per_epoch * total_epochs - 1 the learning rate becomes 0.
    """
    num_training_steps = steps_per_epoch * total_epochs - 1
    num_warmup_steps = steps_per_epoch * warmup_epoch - 1
    if warmup_epoch == 0:
        num_warmup_steps = 0

    logger.info(f"Initialised Cosine_Learning_Rate_Scheduler(num_warmup_steps={num_warmup_steps}, num_training_steps={num_training_steps}) \n")
    
    return _setup_cosine_schedule_with_warmup(optimiser, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
