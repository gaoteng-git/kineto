import math
import pickle
import unittest

from tensorboard_plugin_torch_profiler.profiler.data import RunProfileData
from tensorboard_plugin_torch_profiler.profiler.overall_parser import (
    merge_ranges, subtract_ranges_lists, intersection_ranges_lists, get_ranges_sum,
    OverallParser
)

STEPS_COSTS = "./data/data_steps_costs.pkl"
AVG_COSTS = "./data/data_avg_costs.pkl"


def check_ranges_equal(ranges1, ranges2):
    if len(ranges1) != len(ranges2):
        return False
    for i in range(len(ranges1)):
        if ranges1[i][0] != ranges2[i][0] or ranges1[i][1] != ranges2[i][1]:
            return False
    return True


class TestOverallParser(unittest.TestCase):

    def test_merge_ranges(self):
        src_ranges = [(1.1, 2.2), (1.5, 2.3), (3.3, 3.9), (3.5, 3.6), (3.7, 3.8), (4.1, 4.2)]
        expected_ranges = [(1.1, 2.3), (3.3, 3.9), (4.1, 4.2)]
        dst_ranges = merge_ranges(src_ranges, True)
        is_equal = check_ranges_equal(dst_ranges, expected_ranges)
        self.assertTrue(is_equal)

    def test_subtract_ranges_lists(self):
        ranges1 = [(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)]
        ranges2 = [(0, 0.1), (1.0, 1.4), (1.5, 1.6), (1.9, 3.4), (4.3, 4.6)]
        expected_ranges = [(1.4, 1.5), (1.6, 1.9), (3.4, 4.3), (5.5, 6.6)]
        dst_ranges = subtract_ranges_lists(ranges1, ranges2)
        is_equal = check_ranges_equal(dst_ranges, expected_ranges)
        self.assertTrue(is_equal)

    def test_intersection_ranges_lists(self):
        ranges1 = [(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)]
        ranges2 = [(0, 0.1), (1.0, 1.4), (1.5, 1.6), (1.9, 3.4), (4.3, 4.6)]
        expected_ranges = [(1.1, 1.4), (1.5, 1.6), (1.9, 2.2), (3.3, 3.4), (4.3, 4.4)]
        dst_ranges = intersection_ranges_lists(ranges1, ranges2)
        is_equal = check_ranges_equal(dst_ranges, expected_ranges)
        self.assertTrue(is_equal)

    def test_get_ranges_sum(self):
        ranges = [(1.1, 2.2), (3.3, 4.4), (5.5, 6.6)]
        expected_sum = 3.3
        dst_sum = get_ranges_sum(ranges)
        self.assertTrue(math.isclose(dst_sum, expected_sum))

    def test_parse_events(self):
        def check_step(step):
            sum_cost = (step.kernel_cost
                        + step.memcpy_cost \
                        + step.memset_cost \
                        + step.runtime_cost \
                        + step.dataloader_cost \
                        + step.cpuop_cost \
                        + step.other_cost)
            self.assertTrue(math.isclose(sum_cost, step.step_total_cost))

        data = RunProfileData.parse("./data", "worker0")
        overall_parser = OverallParser()
        overall_parser.parse_events(data.events)
        for step in overall_parser.steps_costs:
            check_step(step)
        check_step(overall_parser.avg_costs)

        def load_object(file_path):
            with open(file_path, "rb") as file:
                obj = pickle.load(file)
            return obj

        steps_costs = load_object(STEPS_COSTS)
        avg_costs = load_object(AVG_COSTS)

        def check_step_equal(step1, step2):
            self.assertTrue(math.isclose(step1.step_total_cost, step2.step_total_cost))
            self.assertTrue(math.isclose(step1.kernel_cost, step2.kernel_cost))
            self.assertTrue(math.isclose(step1.memcpy_cost, step2.memcpy_cost))
            self.assertTrue(math.isclose(step1.memset_cost, step2.memset_cost))
            self.assertTrue(math.isclose(step1.runtime_cost, step2.runtime_cost))
            self.assertTrue(math.isclose(step1.dataloader_cost, step2.dataloader_cost))
            self.assertTrue(math.isclose(step1.cpuop_cost, step2.cpuop_cost))
            self.assertTrue(math.isclose(step1.other_cost, step2.other_cost))

        self.assertEqual(len(overall_parser.steps_costs), len(steps_costs))
        for i in range(len(steps_costs)):
            check_step_equal(overall_parser.steps_costs[i], steps_costs[i])
        check_step_equal(overall_parser.avg_costs, avg_costs)


if __name__ == '__main__':
    unittest.main()
