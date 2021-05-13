# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import sys

from .. import utils
from .trace import EventTypes
from .range_lib import *

logger = utils.get_logger()


ProfileRole = IntEnum('ProfileRole', ['Kernel', 'Memcpy', 'Memset', 'Runtime', 'DataLoader', 'CpuOp', 'Other', 'Total'], start=0)


class OverallParser(object):
    class Costs:
        def __init__(self):
            self.costs = [0] * len(ProfileRole)

        @classmethod
        def calculate_costs(cls, statistics, step):
            cost_obj = cls()
            for i in range(len(statistics.cost_ranges)):
                cost_obj.costs[i] = get_ranges_sum(statistics.cost_ranges[i])
            cost_obj.costs[ProfileRole.Total] = step[1] - step[0]
            return cost_obj

    class Statistics:
        def __init__(self, cost_ranges):
            if not cost_ranges:
                raise ValueError("the cost ranges is None")

            self.cost_ranges = cost_ranges

        @classmethod
        def create_statistics(cls, steps, role_ranges):
            assert len(role_ranges) == ProfileRole.Total - 1

            cost_ranges = []
            slots = []
            for role in role_ranges:
                if slots:
                    range = intersection_ranges_lists(slots, role)
                else:
                    range = role
                    slots = merge_ranges(list(steps))
                cost_ranges.append(range)
                slots = subtract_ranges_lists(slots, range)
            # The last one is ProfileRole.Other
            cost_ranges.append(slots)

            return cls(cost_ranges)

        def intersection_with_step(self, step):
            cost_ranges = []
            step = [step]
            for range in self.cost_ranges:
                cost_ranges.append(intersection_ranges_lists(step, range))

            return OverallParser.Statistics(cost_ranges)

    def __init__(self):
        # we could not use [[]] * len here since they all point to same memory
        # https://stackoverflow.com/questions/12791501/python-initializing-a-list-of-lists
        # https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly
        self.role_ranges = [[] for _ in range(ProfileRole.Total - 1)]
        self.steps = []
        self.steps_names = []

        self.min_ts = sys.maxsize
        self.max_ts = -sys.maxsize - 1
        self.steps_costs = []
        self.avg_costs = OverallParser.Costs()

    # Update self.steps considering device side events launched by each host side step.
    # Update self.steps_names if some tail steps are removed.
    def update_steps_consider_device_side(self, runtime_node_list, device_node_list):
        runtime_node_list = sorted(runtime_node_list, key=lambda x: x.start_time)
        # Make sure self.steps is sorted by time.
        self.steps = sorted(self.steps, key=lambda x: x[0])
        # Use similar code with two-way merge to get all runtimes inside each host-side step span,
        # then record each step's min kernel start time and max kernel end time:
        steps_device = [(sys.maxsize, -sys.maxsize - 1)] * len(self.steps)
        steps_matched_device_nodes = [0] * len(self.steps)
        i_step = 0
        i_runtime = 0
        step_device_min_ts = sys.maxsize
        step_device_max_ts = -sys.maxsize - 1
        matched_device_nodes = set()
        while i_step < len(self.steps) and i_runtime < len(runtime_node_list):
            step_host_start_time = self.steps[i_step][0]
            step_host_end_time = self.steps[i_step][1]
            if runtime_node_list[i_runtime].start_time < step_host_start_time:
                # This runtime is ahead of or intersects with this step span. Skip this runtime.
                i_runtime += 1
            elif runtime_node_list[i_runtime].end_time <= step_host_end_time:
                # and runtime_node_list[i_runtime].start_time >= step_host_start_time
                # This runtime is inside this step span. Scan its device_nodes.
                rt = runtime_node_list[i_runtime]
                if rt.device_nodes is not None:
                    for device_node in rt.device_nodes:
                        step_device_min_ts = min(device_node.start_time, step_device_min_ts)
                        step_device_max_ts = max(device_node.end_time, step_device_max_ts)
                        matched_device_nodes.add(device_node)
                        steps_matched_device_nodes[i_step] += 1
                i_runtime += 1
            elif runtime_node_list[i_runtime].start_time < step_host_end_time:
                # and runtime_node_list[i_runtime].end_time > step_host_end_time
                # This runtime intersects with this step span. Skip this runtime.
                i_runtime += 1
            else:
                # runtime_node_list[i_runtime].start_time >= step_host_end_time
                # This runtime starts after this step's end. Record and move forward this step.
                steps_device[i_step] = (step_device_min_ts, step_device_max_ts)
                i_step += 1
                step_device_min_ts = sys.maxsize
                step_device_max_ts = -sys.maxsize - 1
        while i_step < len(self.steps):
            # This step doesn't launch any device side event, just assign it as empty.
            steps_device[i_step] = (step_device_min_ts, step_device_max_ts)
            step_device_min_ts = sys.maxsize
            step_device_max_ts = -sys.maxsize - 1
            i_step += 1
        # Change step time to device side on the condition that any step have device time.
        is_use_gpu = (len(matched_device_nodes) > 0)
        if is_use_gpu:
            prev_step_end_time = self.steps[0][0]
            if steps_device[0][0] != sys.maxsize:  # When step 0 has device event.
                for device_node in device_node_list:
                    if device_node not in matched_device_nodes:
                        # Now this device_node is not launched inside any step span.
                        if device_node.end_time < steps_device[0][0]:
                            prev_step_end_time = max(prev_step_end_time, device_node.end_time)
            for i_step in range(len(self.steps)):
                step_start_time = max(prev_step_end_time, self.steps[i_step][0])
                step_end_time = self.steps[i_step][1]
                if steps_device[i_step][0] == sys.maxsize:  # When step i_step has no device event.
                    # Assign to step_start_time when kernel is behind host step end.
                    step_end_time = max(step_end_time, step_start_time)
                else:
                    step_end_time = max(step_end_time, steps_device[i_step][1])
                    if step_end_time < step_start_time:
                        logger.warning(
                            "Abnormal step_end_time of step {}: [{}, {}]".format(
                                i_step, step_start_time, step_end_time))
                        step_end_time = step_start_time
                self.steps[i_step] = (step_start_time, step_end_time)  # Update step time considering device side.
                prev_step_end_time = step_end_time

        is_remove_tail_steps = True  # TODO: Use tensorboard argument instead.
        if is_use_gpu and len(self.steps) > 1 and is_remove_tail_steps:
            i_step = len(self.steps) - 1
            while i_step >= 0:
                if steps_matched_device_nodes[i_step] > 0:
                    break
                i_step -= 1
            if i_step >= 0:
                keep_steps = i_step + 1
                if i_step > 0 and steps_matched_device_nodes[i_step - 1] * 0.8 > steps_matched_device_nodes[i_step]:
                    keep_steps = i_step
                if keep_steps < len(self.steps):
                    logger.warning(
                        "Remove the last {} steps from overview. "
                        "Because the profiler may fail to capture all the kernels launched by these steps.".format(
                            len(self.steps) - keep_steps
                        ))
                    self.steps = self.steps[:keep_steps]
                    self.steps_names = self.steps_names[:keep_steps]

    def parse_events(self, events, runtime_node_list, device_node_list):
        logger.debug("Overall, parse events")
        for event in events:
            self.parse_event(event)

        if len(self.steps) == 0:
            self.steps.append((self.min_ts, self.max_ts))
            self.steps_names.append("0")
        self.update_steps_consider_device_side(runtime_node_list, device_node_list)

        for i in range(len(self.role_ranges)):
            self.role_ranges[i] = merge_ranges(self.role_ranges[i])

        logger.debug("Overall, statistics")
        global_stats = OverallParser.Statistics.create_statistics(self.steps, self.role_ranges)

        logger.debug("Overall, aggregation")
        valid_steps = len(self.steps)
        for i in range(valid_steps):
            steps_stat = global_stats.intersection_with_step(self.steps[i])
            self.steps_costs.append(OverallParser.Costs.calculate_costs(steps_stat, self.steps[i]))
            for cost_index in range(len(self.avg_costs.costs)):
                self.avg_costs.costs[cost_index] += self.steps_costs[i].costs[cost_index]

        for i in range(len(self.avg_costs.costs)):
            self.avg_costs.costs[i] /= valid_steps

    def parse_event(self, event):
        ts = event.ts
        dur = event.duration
        evt_type = event.type
        if evt_type == EventTypes.KERNEL:
            self.role_ranges[ProfileRole.Kernel].append((ts, ts + dur))
        elif evt_type == EventTypes.MEMCPY:
            self.role_ranges[ProfileRole.Memcpy].append((ts, ts + dur))
        elif evt_type == EventTypes.MEMSET:
            self.role_ranges[ProfileRole.Memset].append((ts, ts + dur))
        elif evt_type == EventTypes.RUNTIME:
            self.role_ranges[ProfileRole.Runtime].append((ts, ts + dur))
        elif evt_type == EventTypes.OPERATOR and event.name.startswith("enumerate(DataLoader)#") \
                and event.name.endswith(".__next__"):
            self.role_ranges[ProfileRole.DataLoader].append((ts, ts + dur))
        elif event.type == EventTypes.PROFILER_STEP:
            self.steps.append((ts, ts + dur))
            self.steps_names.append(str(event.step))
        elif evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR]:
            self.role_ranges[ProfileRole.CpuOp].append((ts, ts + dur))

        # Record host side min and max time.
        if evt_type in [EventTypes.PYTHON, EventTypes.OPERATOR, EventTypes.PROFILER_STEP]:
            if ts < self.min_ts:
                self.min_ts = ts
            if ts + dur > self.max_ts:
                self.max_ts = ts + dur
