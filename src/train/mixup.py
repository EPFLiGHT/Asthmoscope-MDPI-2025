import numpy as np
import torch


class Mixup(object):
    def __init__(self, mixup_alpha, device, random_seed=42):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.device = device
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)
        mixup_lambdas = np.array(mixup_lambdas).astype(np.float32)

        return torch.from_numpy(mixup_lambdas).to(self.device)


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0::2].transpose(0, -1) * mixup_lambda[0::2] + x[1::2].transpose(0, -1) * mixup_lambda[1::2]).transpose(0,
                                                                                                                    -1)
    return out
