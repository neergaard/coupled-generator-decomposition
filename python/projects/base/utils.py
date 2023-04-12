import time


class BlockTimer(object):
    """Time a block of code.

    Usage:

        # As context manager for short blocks
        with BlockTimer('some computations'):
            print('same goes here')
            time.sleep(1)
            x = np.zeros((100, 200))

        # As class for long blocks
        timer1 = BlockTimer('Total running time', delay_print=True)
        timer1.start()
        print('if using delay_print=True printing stuff here makes sense. otherwise, things get messed up')
        time.sleep(2)
        timer1.stop()

    """

    def __init__(self, name="", delay_print=False):
        # 65 = 79 - len('... [00:00:00]')
        # 68 = 79 - len(' [00:00:00]')
        self.name = name if len(name) <= 68 else name[:65] + "..."
        self.delay_print = delay_print

    def __enter__(self):
        if not self.delay_print:
            print(self.name, end="", flush=True)
        self.time = -time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time += time.perf_counter()
        if self.delay_print:
            print(self.name, end="")
        print(
            f'[{time.strftime("%H:%M:%S", time.gmtime(self.time))}]'.rjust(
                79 - len(self.name)
            )
        )

    # For use as non-context manager
    def start(self):
        self.delay_print = True
        self.__enter__()

    def stop(self):
        self.__exit__(None, None, None)
