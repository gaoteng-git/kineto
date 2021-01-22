import os
import unittest

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import tensorboard_plugin_torch_profiler.consts as consts
from tensorboard_plugin_torch_profiler.profiler.loader import RunLoader

TRACING_DIR = "./data/tracing"
PLUGIN_PROFILE_DIR = "./data/plugin_profile"
PLUGIN_PROFILE_POSTFIX = ".profile.json"


class TestProfiler(unittest.TestCase):
    def test_run(self):
        def compare_runs(run, run_expected):
            self.assertEqual(len(run.profiles), len(run_expected.profiles))
            for key in run.profiles.keys():
                self.assertTrue(key in run.profiles)
                prof = run.profiles[key]
                prof_expected = run_expected.profiles[key]

                overview_rows = prof.overview["steps"]["rows"]
                overview_rows_expected = prof_expected.overview["steps"]["rows"]
                self.assertEqual(overview_rows, overview_rows_expected)

                overview_avgs = prof.overview["performance"][0]["children"]
                overview_avgs_expected = prof_expected.overview["performance"][0]["children"]
                self.assertEqual(overview_avgs, overview_avgs_expected)

                op_by_name = prof.operation_table_by_name
                op_by_name_expected = prof_expected.operation_table_by_name
                # No float number, so assertEqual works.
                self.assertEqual(op_by_name, op_by_name_expected)

                op_by_name_input = prof.operation_table_by_name_input
                op_by_name_input_expected = prof_expected.operation_table_by_name_input
                # No float number, so assertEqual works.
                self.assertEqual(op_by_name_input, op_by_name_input_expected)

                if consts.KERNEL_VIEW in prof_expected.views:
                    kernel_op_table = prof.kernel_op_table
                    kernel_op_table_expected = prof_expected.kernel_op_table
                    self.assertEqual(kernel_op_table, kernel_op_table_expected)

                    kernel_table = prof.kernel_table
                    kernel_table_expected = prof_expected.kernel_table
                    self.assertEqual(kernel_table, kernel_table_expected)

        for run_dir in os.listdir(TRACING_DIR):
            run_loader = RunLoader(run_dir, os.path.join(TRACING_DIR, run_dir))
            run = run_loader.load()
            profile_path = os.path.join(PLUGIN_PROFILE_DIR, "{}.{}".format(run_dir, PLUGIN_PROFILE_POSTFIX))
            with open(profile_path, "r") as file:
                run_expected = jsonpickle.decode(file.read())
                compare_runs(run, run_expected)


if __name__ == '__main__':
    jsonpickle_pd.register_handlers()
    unittest.main()
