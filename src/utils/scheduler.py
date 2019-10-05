
class Scheduler:
    """
    Scheduler class for parameter change over time.
    """
    def __init__(self, initial_value, sequence_function=lambda x: x):
        self.val = initial_value
        self.fun = sequence_function

    def __call__(self):
        self.val = self.fun(self.val)
        return self.val
