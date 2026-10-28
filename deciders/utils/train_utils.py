from collections.abc import Iterable, Mapping
from typing import Any

import torch


def create_optimizer(
    params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
    config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Instantiate an optimizer without modifying ``config``.

    Args:
        params: Model parameters or optimizer parameter groups.
        config: Mapping with a case-sensitive ``name`` from ``torch.optim``.
            Remaining entries are passed as optimizer keyword arguments,
            for example ``{"name": "AdamW", "lr": 1e-3}``.

    Raises:
        TypeError: If the config is not a mapping or its name is not a string.
        ValueError: If the optimizer name is missing or unsupported, or the
            config contains ``params`` (which must be supplied separately).
            Optimizer constructor errors are propagated unchanged.
    """
    if not isinstance(config, Mapping):
        raise TypeError("Optimizer config must be a mapping")

    options = dict(config)
    if "name" not in options:
        raise ValueError("Optimizer config must include 'name'")
    name = options.pop("name")
    if not isinstance(name, str):
        raise TypeError("Optimizer name must be a string")

    optimizer_cls = getattr(torch.optim, name, None)
    if (
        not isinstance(optimizer_cls, type)
        or not issubclass(optimizer_cls, torch.optim.Optimizer)
        or optimizer_cls is torch.optim.Optimizer
    ):
        raise ValueError(f"Unsupported torch optimizer: {name!r}")
    if "params" in options:
        raise ValueError("Optimizer parameters must be supplied separately")

    return optimizer_cls(params, **options)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler:
    """Instantiate a learning-rate scheduler without modifying ``config``.

    Args:
        optimizer: Optimizer whose learning rate the scheduler controls.
        config: Mapping with a case-sensitive ``name`` from
            ``torch.optim.lr_scheduler``. Remaining entries are passed as
            scheduler keyword arguments, for example
            ``{"name": "OneCycleLR", "max_lr": 1e-4, "total_steps": 3144}``.

    Raises:
        TypeError: If the config is not a mapping or its name is not a string.
        ValueError: If the scheduler name is missing or unsupported, or the
            config contains ``optimizer`` (which must be supplied separately).
            Scheduler constructor errors are propagated unchanged.
    """
    if not isinstance(config, Mapping):
        raise TypeError("Scheduler config must be a mapping")

    options = dict(config)
    if "name" not in options:
        raise ValueError("Scheduler config must include 'name'")
    name = options.pop("name")
    if not isinstance(name, str):
        raise TypeError("Scheduler name must be a string")

    scheduler_cls = getattr(torch.optim.lr_scheduler, name, None)
    if (
        not isinstance(scheduler_cls, type)
        or not issubclass(scheduler_cls, torch.optim.lr_scheduler.LRScheduler)
        or scheduler_cls is torch.optim.lr_scheduler.LRScheduler
    ):
        raise ValueError(f"Unsupported torch LR scheduler: {name!r}")
    if "optimizer" in options:
        raise ValueError("Scheduler optimizer must be supplied separately")

    return scheduler_cls(optimizer, **options)
