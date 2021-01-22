import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import unittest
import os

from tensorboard_plugin_torch_profiler.profiler.loader import RunLoader

TRACE_DIR = "./data/tracing"
PLUGIN_PROFILE_DIR = "./data/plugin_profile"
PLUGIN_PROFILE_POSTFIX = ".profile.json"


def save_run():
    for run_dir in os.listdir(TRACE_DIR):
        run_loader = RunLoader(run_dir, os.path.join(TRACE_DIR, run_dir))
        run = run_loader.load()
        profile_path = os.path.join(PLUGIN_PROFILE_DIR, "{}.{}".format(run_dir, PLUGIN_PROFILE_POSTFIX))
        with open(profile_path, "w") as file:
            file.write(jsonpickle.encode(run))


class TestProfiler(unittest.TestCase):
    def test_run(self):
        for run_dir in os.listdir(TRACE_DIR):
            run_loader = RunLoader(run_dir, os.path.join(TRACE_DIR, run_dir))
            run = run_loader.load()
            profile_path = os.path.join(PLUGIN_PROFILE_DIR, "{}.{}".format(run_dir, PLUGIN_PROFILE_POSTFIX))
            with open(profile_path, "r") as file:
                run_expected = jsonpickle.decode(file.read())
            self.assertEqual(run, run_expected)

if __name__ == '__main__':
    jsonpickle_pd.register_handlers()
    save_run()
    unittest.main()