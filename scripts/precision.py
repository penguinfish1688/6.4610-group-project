"""Keep FP32 norm master weights with native activation dtypes."""

from types import MethodType


def preserve_norm_dtype(module):
    original = module.forward

    def forward(this, hidden_states):
        return original(hidden_states).to(dtype=hidden_states.dtype)

    module.forward = MethodType(forward, module)


def configure(model):
    count = 0
    for module in model.modules():
        weight = getattr(module, "weight", None)
        if (
            type(module).__name__.endswith("RMSNorm")
            and weight is not None
            and weight.requires_grad
        ):
            preserve_norm_dtype(module)
            count += 1
    return count
