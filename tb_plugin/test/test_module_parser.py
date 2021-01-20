import unittest
import jsonpickle

from tensorboard_plugin_torch_profiler.profiler.module_parser import (ModuleParser, OperatorAgg, KernelAggByNameOp)
from tensorboard_plugin_torch_profiler.profiler.data import RunProfileData

OP_LIST = "./data/data_op_list_groupby_name.json"
OP_LIST_BY_INPUT = "./data/data_op_list_groupby_name_input.json"
KERNEL_LIST_BY_OP = "./data/data_kernel_list_groupby_name_op.json"


def save_object(agg_list, file_path):
    with open(file_path, "w") as file:
        file.write(jsonpickle.encode(agg_list))


def save_golden_files():
    data = RunProfileData.parse("./data", "worker0")
    module_parser = ModuleParser()
    module_parser.parse_events(data.events)
    save_object(module_parser.op_list_groupby_name, OP_LIST)
    save_object(module_parser.op_list_groupby_name_input, OP_LIST_BY_INPUT)
    save_object(module_parser.kernel_list_groupby_name_op, KERNEL_LIST_BY_OP)


class TestModuleParser(unittest.TestCase):
    def test_parse_events(self):
        def load_agg(file_path):
            with open(file_path, "r") as file:
                agg_list = jsonpickle.decode(file.read())
            return agg_list
        op_list_groupby_name = load_agg(OP_LIST)
        op_list_groupby_name_input = load_agg(OP_LIST_BY_INPUT)
        kernel_list_groupby_name_op = load_agg(KERNEL_LIST_BY_OP)

        data = RunProfileData.parse("./data", "worker0")
        module_parser = ModuleParser()
        module_parser.parse_events(data.events)
        self.assertEqual(module_parser.op_list_groupby_name.sort(key=lambda v: v.name),
                         op_list_groupby_name.sort(key=lambda v: v.name))
        self.assertEqual(module_parser.op_list_groupby_name_input.sort(key=lambda v: v.name+"_"+v.input_shape),
                         op_list_groupby_name_input.sort(key=lambda v: v.name+"_"+v.input_shape))
        self.assertEqual(module_parser.kernel_list_groupby_name_op.sort(key=lambda v: v.name+"_"+v.op_name),
                         kernel_list_groupby_name_op.sort(key=lambda v: v.name+"_"+v.op_name))


if __name__ == '__main__':
    save_golden_files()
    unittest.main()