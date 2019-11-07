import numpy as np
from core.optimizers import GenericOptimizer


class NoisyMomentum(GenericOptimizer):
    """
    Uses minibatch_size, learning_rate, momentum_amount parameters.

    Randomly scale each adjustment by some amount between 0 and 1.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.previous_adjustment = None

    def __str__(self):
        str_parts = [
            "noisy momentum",
            f"learning_rate: {self.learning_rate}",
            f"momentum amount: {self.momentum_amount}",
            f"minibatch size: {self.minibatch_size}",
        ]
        return "\n".join(str_parts)

    def update(self, layer):
        de_dw_batch = self.update_minibatch(layer)
        if de_dw_batch is None:
            return

        if self.previous_adjustment is None:
            self.previous_adjustment = np.zeros(layer.weights.shape)
        new_adjustment = (
            self.previous_adjustment * self.momentum_amount
            + de_dw_batch * self.learning_rate
        )
        new_adjustment_noisy = (
            new_adjustment * np.random.uniform(size=new_adjustment.shape))
        layer.weights -= new_adjustment_noisy

        # Update previous_adjustment to get set up for the next iteration.
        self.previous_adjustment = new_adjustment_noisy
