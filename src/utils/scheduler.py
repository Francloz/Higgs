
class Scheduler:
    """
    Scheduler class for parameter change over time.
    """
    def __init__(self, initial_value, next_val=lambda x: x):
        """
        Scheduler class.
        :param initial_value: Initial value.
        :param next_val: Function that returns the next value given the last one
        """
        self.next_val = next_val
        self.next = next

    def __call__(self):
        """
        Returns the next value.
        :return: Next value
        """
        self.val = self.next_val(self.val)
        return self.val
