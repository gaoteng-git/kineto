import unittest
import math
import pickle

from tensorboard_plugin_torch_profiler.profiler.module_parser import (ModuleParser, OperatorAgg, KernelAggByNameOp)
from tensorboard_plugin_torch_profiler.profiler.data import RunProfileData

OP_LIST = "data_op_list_groupby_name.pkl"
OP_LIST_BY_INPUT = "data_op_list_groupby_name_input.pkl"
KERNEL_LIST = "data_kernel_list.pkl"
KERNEL_LIST_BY_OP = "data_kernel_list_groupby_name_op.pkl"


def save_agg(agg_list, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(agg_list, file)


def save_golden_files():
    data = RunProfileData.parse(".", "worker0")
    moduleParser = ModuleParser()
    moduleParser.parse_events(data.events)
    save_agg(moduleParser.op_list_groupby_name, OP_LIST)
    save_agg(moduleParser.op_list_groupby_name_input, OP_LIST_BY_INPUT)
    save_agg(moduleParser.kernel_list, KERNEL_LIST)
    save_agg(moduleParser.kernel_list_groupby_name_op, KERNEL_LIST_BY_OP)


class TestModuleParser(unittest.TestCase):
    def test_parse_events(self):
        def load_agg(file_path):
            with open(file_path, "rb") as file:
                agg_list = pickle.load(file)
            return agg_list
        op_list_groupby_name = load_agg(OP_LIST)
        op_list_groupby_name_input = load_agg(OP_LIST_BY_INPUT)
        kernel_list = load_agg(KERNEL_LIST)
        kernel_list_groupby_name_op = load_agg(KERNEL_LIST_BY_OP)

        data = RunProfileData.parse(".", "worker0")
        moduleParser = ModuleParser()
        moduleParser.parse_events(data.events)
        self.assertEqual(moduleParser.op_list_groupby_name.sort(key=lambda v: v.name),
                         op_list_groupby_name.sort(key=lambda v: v.name))
        self.assertEqual(moduleParser.op_list_groupby_name_input.sort(key=lambda v: v.name+"_"+v.input_shape),
                         op_list_groupby_name_input.sort(key=lambda v: v.name+"_"+v.input_shape))
        self.assertEqual(moduleParser.kernel_list.sort(key=lambda v: v.name),
                         kernel_list.sort(key=lambda v: v.name))
        self.assertEqual(moduleParser.kernel_list_groupby_name_op.sort(key=lambda v: v.name+"_"+v.op_name),
                         kernel_list_groupby_name_op.sort(key=lambda v: v.name+"_"+v.op_name))


if __name__ == '__main__':
    save_golden_files()
    unittest.main()