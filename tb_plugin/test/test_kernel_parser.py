import unittest
import pickle

from tensorboard_plugin_torch_profiler.profiler.kernel_parser import (KernelParser)
from tensorboard_plugin_torch_profiler.profiler.data import RunProfileData

KERNEL_STAT = "./data/data_kernel_stat.pkl"


def save_object(kernel_stat, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(kernel_stat, file)


def save_golden_files():
    data = RunProfileData.parse("./data", "worker0")
    kernel_parser = KernelParser()
    kernel_parser.parse_events(data.events)
    save_object(kernel_parser.kernel_stat, KERNEL_STAT)


class TestKernelParser(unittest.TestCase):
    def test_parse_events(self):
        def load_kernels(file_path):
            with open(file_path, "rb") as file:
                kernels = pickle.load(file)
            return kernels
        kernel_stat = load_kernels(KERNEL_STAT)

        data = RunProfileData.parse("./data", "worker0")
        kernel_parser = KernelParser()
        kernel_parser.parse_events(data.events)
        self.assertTrue(kernel_parser.kernel_stat.equals(kernel_stat))


if __name__ == '__main__':
    save_golden_files()
    unittest.main()