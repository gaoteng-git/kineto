import unittest
from tensorboard-plugin-torch-profiler.profiler.overall_parser import merged_ranges

class TestOverallParser(unittest.TestCase):
    def test_merge_ranges(self):
        src_ranges = [(1.1, 2.2), (1.5, 2.3), (3.3, 3.9), (3.5, 3.6), (3.7, 3.8), (4.1, 4.2)]
        merged_ranges = merge_ranges(src_ranges, True)
    #def test_parse_events(self):
