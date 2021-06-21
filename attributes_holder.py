jump_name = 'jump'
run_fire_name = 'run_fire'
space_name = 'space_is_pressed'
time_name = 'timestamp_ns'
interval = 0.025


class Attrdict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def __str__(self):
        return str(dict(self))
