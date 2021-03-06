# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from .data import RunData, RunProfileData
from .run_generator import RunGenerator
from .. import consts, utils
from ..run import Run

logger = utils.get_logger()


class RunLoader(object):
    def __init__(self, name, run_dir):
        self.run = RunData(name, run_dir)

    def load(self):
        self._parse()
        if len(self.run.profiles) == 0:
            logger.warning("No profile data found.")
            return None

        self._process()

        self._analyze()

        run = self._generate_run()
        return run

    def _parse(self):
        workers = []
        for path in os.listdir(self.run.run_dir):
            if os.path.isdir(path):
                continue
            for pattern in [consts.TRACE_GZIP_FILE_SUFFIX, consts.TRACE_FILE_SUFFIX]:
                if path.endswith(pattern):
                    worker = path[:-len(pattern)]
                    workers.append(worker)
                    break

        for worker in sorted(workers):
            try:
                data = RunProfileData.parse(self.run.run_dir, worker)
                self.run.profiles[worker] = data
            except Exception as ex:
                logger.warning("Failed to parse profile data for Run %s on %s. Exception=%s",
                               self.run.name, worker, ex, exc_info=True)

    def _process(self):
        for data in self.run.profiles.values():
            logger.debug("Processing profile data")
            data.process()
            logger.debug("Processing profile data finish")
        self._compare_worker()

    def _compare_worker(self):
        workers = self.run.profiles.keys()
        worker2threads = []
        logger.warning("workers:{}".format(workers))
        for worker in sorted(workers):
            tid2tree_array = []
            for key, value in self.run.profiles[worker].module_parser.tid2tree.items():
                temp = [key, value]
                tid2tree_array.append(temp)
            tid2tree_array.sort(key = lambda x: x[1].start_time)
            worker2threads.append(tid2tree_array)

        self.compare_nodes = 0
        def compare_tree(tree1, tree2):
            if tree1 is None and tree2 is None:
                return True
            self.compare_nodes += 1

            if tree1 is None:
                logger.warning("tree1 is None, tree2 is '{}'".format(tree2.name))
                return False
            if tree2 is None:
                logger.warning("tree1 is '{}', tree2 is None".format(tree1.name))
                return False
            if tree1.name != tree2.name:
                logger.warning("Not equal name! ({},{}) ({},{})".format(tree1.name, tree1.start_time,
                                                                        tree2.name, tree2.start_time))
                return False
            if len(tree1.children) != len(tree2.children):
                logger.warning("Not equal children number! ({},{}, {}) ({},{}, {})".format(
                    tree1.name, tree1.start_time, len(tree1.children),
                    tree2.name, tree2.start_time, len(tree2.children)))
            if tree1.debug_device_nodes != tree2.debug_device_nodes:
                logger.warning("Not equal debug_device_nodes! ({},{}, {}) ({},{}, {})".format(
                    tree1.name, tree1.start_time, tree1.debug_device_nodes,
                    tree2.name, tree2.start_time, tree2.debug_device_nodes))
            op1 = "{}###{}".format(tree1.name, tree1.debug_device_nodes)
            op2 = "{}###{}".format(tree2.name, tree2.debug_device_nodes)
            if not self.op2count_0.__contains__(op1):
                self.op2count_0[op1] = 0
            self.op2count_0[op1] += 1
            if not self.op2count_1.__contains__(op2):
                self.op2count_1[op2] = 0
            self.op2count_1[op2] += 1

            for i in range(len(tree1.children)):
                ret = compare_tree(tree1.children[i], tree2.children[i])
                if not ret:
                    return False
            return True

        patterns = ["cudnn_convolution_backward_input"]
        if len(worker2threads) >= 2:
            self.op2count_0 = {}
            self.op2count_1 = {}
            threads0 = worker2threads[0]
            threads1 = worker2threads[1]
            if len(threads0) != len(threads1):
                logger.warning("Thread number not equal!")
            else:
                for t in range(len(threads0)):
                    logger.warning("Comparing {} with {}".format(threads0[t][0], threads1[t][0]))
                    ret = compare_tree(threads0[t][1], threads1[t][1])
                    logger.warning("Comparing result = {}; compare_ndes={}".format(ret, self.compare_nodes))

        def print_pattern_key(op2count, pattern):
            for key, count in op2count.items():
                if pattern in key:
                    print("{},{}".format(key, count))

        print("Print Patterns for 0 =========================")
        print_pattern_key(self.op2count_0, patterns[0])
        print("Print Patterns for 1 =========================")
        print_pattern_key(self.op2count_1, patterns[0])


    def _analyze(self):
        for data in self.run.profiles.values():
            logger.debug("Analyzing profile data")
            data.analyze()
            logger.debug("Analyzing profile data finish")

    def _generate_run(self):
        run = Run(self.run.name, self.run.run_dir)
        for worker, data in self.run.profiles.items():
            generator = RunGenerator(worker, data)
            profile = generator.generate_run_profile()
            run.add_profile(profile)
        return run
