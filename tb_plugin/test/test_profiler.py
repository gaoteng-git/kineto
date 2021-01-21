import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import unittest

from tensorboard_plugin_torch_profiler.profiler.loader import RunLoader

RESNET50 = "./data/Resnet50.json"

def save_run():
    run_loader = RunLoader("test_profiler", "./data")
    run = run_loader.load()
    with open(RESNET50, "w") as file:
        file.write(jsonpickle.encode(run))


class TestProfiler(unittest.TestCase):
    def test_run(self):
        run_loader = RunLoader("test_profiler", "./data")
        run = run_loader.load()
        with open(RESNET50, "r") as file:
            run_expected = jsonpickle.decode(file.read())
        self.assertEqual(run, run_expected)

if __name__ == '__main__':
    jsonpickle_pd.register_handlers()
    save_run()
    unittest.main()