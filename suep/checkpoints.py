import logging
import numpy as np
import torch


class EarlyStopping:

    def __init__(self,
                 logger,
                 patience=7,
                 delta=10e-6,
                 save_best=True,
                 save_path='models/'):
        """Early stopping checkpont"""

        self.patience = patience
        self.delta = delta
        self.save_best = save_best
        self.save_path = save_path

        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False

        self.logger = logger
        self.logger.info(
            "Initiated early stopping with patience {}.".format(self.patience)
        )

    def __call__(self, loss, model, model2=None):
        """Veryfy if training should be terminated"""

        if loss < self.best_score - self.delta:
            self.counter = 0  # Reset counter
            self.best_score = loss  # Store best score
            if self.save_best:
                self.save_checkpoint(model)
                if model2:
                    self.save_checkpoint(model2, "B")
        else:
            self.logger.debug("Validation loss did not decrease")
            self.counter += 1  # Increment counter
            if self.counter == self.patience:
                self.logger.info("Stopped by checkpoint!")
                return True
        return False

    def save_checkpoint(self, model, suffix=""):
        """Saves model when validation loss decrease."""

        self.logger.debug("Saving model to {}{}".format(self.save_path, suffix))
        torch.save(model.state_dict(), "{}{}.pth".format(self.save_path, suffix))
