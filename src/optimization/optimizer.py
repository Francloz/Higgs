from src.model.model import Model


class Optimizer:
    def __call__(self, model: Model, tx, y, **kwargs):
        """
        Optimizes the model.
        :param tx: sample
        :param y: labels
        """
        pass
