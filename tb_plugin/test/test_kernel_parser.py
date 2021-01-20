import unittest
import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd

from tensorboard_plugin_torch_profiler.profiler.kernel_parser import (KernelParser)
from tensorboard_plugin_torch_profiler.profiler.data import RunProfileData

KERNEL_STAT = "./data/data_kernel_stat.json"


def save_object(kernel_stat, file_path):
    with open(file_path, "w") as file:
        file.write(jsonpickle.encode(kernel_stat))


def save_golden_files():
    data = RunProfileData.parse("./data", "worker0")
    kernel_parser = KernelParser()
    kernel_parser.parse_events(data.events)
    save_object(kernel_parser.kernel_stat, KERNEL_STAT)


class TestKernelParser(unittest.TestCase):
    def test_parse_events(self):
        def load_kernels(file_path):
            with open(file_path, "r") as file:
                kernels = jsonpickle.decode(file.read())
            return kernels
        kernel_stat = load_kernels(KERNEL_STAT)

        data = RunProfileData.parse("./data", "worker0")
        kernel_parser = KernelParser()
        kernel_parser.parse_events(data.events)
        self.assertTrue(kernel_parser.kernel_stat.equals(kernel_stat))


if __name__ == '__main__':
    jsonpickle_pd.register_handlers()
    save_golden_files()
    unittest.main()