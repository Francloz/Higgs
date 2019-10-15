from model.model import Model


class Optimizer:
    def __init__(self, model: Model):
        self.model = model

    def __call__(self, tx, y, **kwargs):
        """
        Optimizes the model.
        :param tx: sample
        :param y: labels
        """
        pass
