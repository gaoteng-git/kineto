import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import unittest
import math

from tensorboard_plugin_torch_profiler.profiler.loader import RunLoader

RESNET50 = "./data/Resnet50.json"


class TestProfiler(unittest.TestCase):
    def compare_list(self, list1, list2):
        self.assertEqual(len(list1), len(list2))
        for i in range(len(list1)):
            self.assertTrue(math.isclose(list1[i], list2[i]))

    def test_run(self):
        run_loader = RunLoader("test_profiler", "./data")
        run = run_loader.load()
        with open(RESNET50, "r") as file:
            run_expected = jsonpickle.decode(file.read())

        self.assertEqual(len(run.profiles), len(run_expected.profiles))
        for key in run.profiles.keys():
            self.assertTrue(key in run.profiles)

            overview_rows = run.profiles[key].overview["steps"]["rows"]
            overview_rows_expected = run_expected.profiles[key].overview["steps"]["rows"]
            self.assertEqual(len(overview_rows), len(overview_rows_expected))
            for i in range(len(overview_rows)):
                if isinstance(overview_rows[i], float):
                    self.assertTrue(isinstance(overview_rows_expected[i], float))
                    self.assertTrue(math.isclose(overview_rows[i], overview_rows_expected[i]))

            overview_avgs = run.profiles[key].overview["performance"][0]["children"]
            overview_avgs_expected = run_expected.profiles[key].overview["performance"][0]["children"]
            self.assertEqual(len(overview_avgs), len(overview_avgs_expected))
            for i in range(len(overview_avgs)):
                self.assertEqual(overview_avgs[i]["name"], overview_avgs_expected[i]["name"])
                self.assertTrue(math.isclose(overview_avgs[i]["value"], overview_avgs_expected[i]["value"]))

            op_by_name = run.profiles[key].operation_table_by_name["data"]["rows"]
            op_by_name_expected = run_expected.profiles[key].operation_table_by_name["data"]["rows"]
            self.assertEqual(op_by_name, op_by_name_expected)

            op_by_name_input = run.profiles[key].operation_table_by_name_input["data"]["rows"]
            op_by_name_input_expected = run_expected.profiles[key].operation_table_by_name_input["data"]["rows"]
            self.assertEqual(op_by_name_input, op_by_name_input_expected)


if __name__ == '__main__':
    jsonpickle_pd.register_handlers()
    unittest.main()