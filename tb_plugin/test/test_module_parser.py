import math
import jsonpickle
import unittest

from tensorboard_plugin_torch_profiler.profiler.data import RunProfileData
from tensorboard_plugin_torch_profiler.profiler.module_parser import ModuleParser

OP_LIST = "./data/data_op_list_groupby_name.json"
OP_LIST_BY_INPUT = "./data/data_op_list_groupby_name_input.json"
KERNEL_LIST_BY_OP = "./data/data_kernel_list_groupby_name_op.json"


class TestModuleParser(unittest.TestCase):
    def test_parse_events(self):
        def load_agg(file_path):
            with open(file_path, "rb") as file:
                agg_list = jsonpickle.decode(file)
            return agg_list

        op_list_groupby_name = load_agg(OP_LIST)
        op_list_groupby_name_input = load_agg(OP_LIST_BY_INPUT)
        kernel_list_groupby_name_op = load_agg(KERNEL_LIST_BY_OP)

        data = RunProfileData.parse("./data", "worker0")
        module_parser = ModuleParser()
        module_parser.parse_events(data.events)

        def check_op_agg_equal(agg1, agg2):
            self.assertEqual(agg1.name, agg2.name)
            self.assertEqual(agg1.input_shape, agg2.input_shape)
            self.assertEqual(agg1.calls, agg2.calls)
            self.assertTrue(math.isclose(agg1.host_duration, agg2.host_duration))
            self.assertTrue(math.isclose(agg1.device_duration, agg2.device_duration))
            self.assertTrue(math.isclose(agg1.self_host_duration, agg2.self_host_duration))
            self.assertTrue(math.isclose(agg1.self_device_duration, agg2.self_device_duration))
            self.assertTrue(math.isclose(agg1.avg_host_duration, agg2.avg_host_duration))
            self.assertTrue(math.isclose(agg1.avg_device_duration, agg2.avg_device_duration))

        def check_op_agg_list_equal(list1, list2):
            self.assertEqual(len(list1), len(list2))
            for i in range(len(list1)):
                check_op_agg_equal(list1[i], list2[i])

        def check_kernel_agg_equal(agg1, agg2):
            self.assertEqual(agg1.name, agg2.name)
            self.assertEqual(agg1.calls, agg2.calls)
            self.assertTrue(math.isclose(agg1.total_duration, agg2.total_duration))
            self.assertTrue(math.isclose(agg1.avg_duration, agg2.avg_duration))
            self.assertTrue(math.isclose(agg1.min_duration, agg2.min_duration))
            self.assertTrue(math.isclose(agg1.max_duration, agg2.max_duration))

        def check_kernel_agg_list_equal(list1, list2):
            self.assertEqual(len(list1), len(list2))
            for i in range(len(list1)):
                check_kernel_agg_equal(list1[i], list2[i])

        check_op_agg_list_equal(
            sorted(module_parser.op_list_groupby_name, key=lambda v: v.name),
            sorted(op_list_groupby_name, key=lambda v: v.name))
        check_op_agg_list_equal(
            sorted(module_parser.op_list_groupby_name_input, key=lambda v: "{}_{}".format(v.name, v.input_shape)),
            sorted(op_list_groupby_name_input, key=lambda v: "{}_{}".format(v.name, v.input_shape)))
        check_kernel_agg_list_equal(
            sorted(module_parser.kernel_list_groupby_name_op, key=lambda v: "{}_{}".format(v.name, v.op_name)),
            sorted(kernel_list_groupby_name_op, key=lambda v: "{}_{}".format(v.name, v.op_name)))


if __name__ == '__main__':
    unittest.main()
