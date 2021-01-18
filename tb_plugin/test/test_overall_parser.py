import unittest
from tensorboard_plugin_torch_profiler.profiler.overall_parser import merge_ranges

def check_ranges_equal(ranges1, ranges2):
    if len(ranges1) != len(ranges2):
        return False
    for i in range(len(ranges1)):
        if (ranges1[i][0] != ranges2[i][0] or ranges1[i][1] != ranges2[i][1]):
            return False
    return True

class TestOverallParser(unittest.TestCase):


    def test_merge_ranges(self):
        src_ranges = [(1.1, 2.2), (1.5, 2.3), (3.3, 3.9), (3.5, 3.6), (3.7, 3.8), (4.1, 4.2)]
        expected_ranges = [(1.1, 2.3), (3.3, 3.9), (4.1, 4.2)]
        merged_ranges = merge_ranges(src_ranges, True)
        is_equal = check_ranges_equal(merged_ranges, expected_ranges)
        self.assertTrue(is_equal)
    #def test_parse_events(self):


if __name__ == '__main__':
    unittest.main()