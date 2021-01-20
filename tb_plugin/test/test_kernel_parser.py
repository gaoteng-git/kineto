import jsonpickle
import unittest

from tensorboard_plugin_torch_profiler.profiler.data import RunProfileData
from tensorboard_plugin_torch_profiler.profiler.kernel_parser import KernelParser

KERNEL_STAT = "./data/data_kernel_stat.json"


class TestKernelParser(unittest.TestCase):
    def test_parse_events(self):
        def load_kernels(file_path):
            with open(file_path, "rb") as file:
                kernels = jsonpickle.decode(file)
            return kernels

        kernel_stat = load_kernels(KERNEL_STAT)

        data = RunProfileData.parse("./data", "worker0")
        kernel_parser = KernelParser()
        kernel_parser.parse_events(data.events)
        self.assertTrue(kernel_parser.kernel_stat.equals(kernel_stat))


if __name__ == '__main__':
    unittest.main()
