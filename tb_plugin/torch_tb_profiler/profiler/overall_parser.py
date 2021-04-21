# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

import sys
from enum import IntEnum

from .. import utils
from .trace import EventTypes

logger = utils.get_logger()


def merge_ranges(src_ranges, is_sorted=False):
    merged_ranges = []
    if len(src_ranges) > 0:
        if not is_sorted:
            src_ranges.sort(key=lambda x: x[0])
        src_id = 0
        merged_ranges.append(
            (src_ranges[src_id][0], src_ranges[src_id][1]))
        for src_id in range(1, len(src_ranges)):
            dst_id = len(merged_ranges) - 1
            if src_ranges[src_id][1] > merged_ranges[dst_id][1]:
                if src_ranges[src_id][0] <= merged_ranges[dst_id][1]:
                    merged_ranges[dst_id] = (merged_ranges[dst_id][0], src_ranges[src_id][1])
                else:
                    merged_ranges.append(
                        (src_ranges[src_id][0], src_ranges[src_id][1]))
    return merged_ranges


def subtract_ranges_lists(range_list1, range_list2):
    range_list_dst = []
    if len(range_list1) == 0:
        return range_list_dst
    if len(range_list2) == 0:
        range_list_dst = list(range_list1)
        return range_list_dst
    r1 = range_list1[0]
    r2 = range_list2[0]
    i1 = i2 = 0
    while i1 < len(range_list1):
        if i2 == len(range_list2):
            range_list_dst.append(r1)
            r1, i1 = pop_list(range_list1, i1)
        elif r2[1] <= r1[0]:
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] <= r1[0] and r2[1] < r1[1]:
            r1 = (r2[1], r1[1])
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] <= r1[0]:
            assert (r2[1] >= r1[1])
            r2 = (r1[1], r2[1])
            r1, i1 = pop_list(range_list1, i1)
        elif r2[0] < r1[1]:
            assert (r2[0] > r1[0])
            range_list_dst.append((r1[0], r2[0]))
            r1 = (r2[0], r1[1])
        else:
            assert (r2[0] >= r1[1])
            range_list_dst.append(r1)
            r1, i1 = pop_list(range_list1, i1)
    return range_list_dst


def intersection_ranges_lists(range_list1, range_list2):
    range_list_dst = []
    if len(range_list1) == 0 or len(range_list2) == 0:
        return range_list_dst
    r1 = range_list1[0]
    r2 = range_list2[0]
    i1 = i2 = 0
    while i1 < len(range_list1):
        if i2 == len(range_list2):
            break
        elif r2[1] <= r1[0]:
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] <= r1[0] and r2[1] < r1[1]:
            assert (r2[1] > r1[0])
            range_list_dst.append((r1[0], r2[1]))
            r1 = (r2[1], r1[1])
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] <= r1[0]:
            assert (r2[1] >= r1[1])
            range_list_dst.append(r1)
            r2 = (r1[1], r2[1])
            r1, i1 = pop_list(range_list1, i1)
        elif r2[1] < r1[1]:
            assert (r2[0] > r1[0])
            range_list_dst.append(r2)
            r1 = (r2[1], r1[1])
            r2, i2 = pop_list(range_list2, i2)
        elif r2[0] < r1[1]:
            assert (r2[1] >= r1[1])
            range_list_dst.append((r2[0], r1[1]))
            r2 = (r1[1], r2[1])
            r1, i1 = pop_list(range_list1, i1)
        else:
            assert (r2[0] >= r1[1])
            r1, i1 = pop_list(range_list1, i1)
    return range_list_dst


def get_ranges_sum(ranges):
    sum = 0
    for range in ranges:
        sum += (range[1] - range[0])
    return sum


def pop_list(range_list, index):
    next_index = index + 1
    if next_index >= len(range_list):
        return None, len(range_list)
    next_item = range_list[next_index]
    return next_item, next_index


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

        self.calculate_gpu_utilization()

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

    def parse_clock_time(self, clock_time_str):
        parts = clock_time_str.split(':')
        seconds = float(parts[2]) + int(parts[1]) * 60 + int(parts[0]) * 60 * 60
        return seconds

    def parse_lms1_file(self, file_path):
        with open(file_path) as file:
            lines = file.readlines()

            buckets = []
            gpu_utilizations = []
            prev_gpu_util_str = ""
            for index in range(len(lines)):
                point = lines[index]
                start_pos = point.find(' ') + 1
                end_pos = point.find(',')

                gpu_util_str = point[end_pos + 2:point.find('%') - 1]
                if gpu_util_str == prev_gpu_util_str:
                    continue
                prev_gpu_util_str = gpu_util_str

                clock_time_str = point[start_pos:end_pos]
                timestamp_end = self.clock_time_to_timestamp(clock_time_str)
                timestamp_start = timestamp_end - self.timestamp_per_period
                buckets.append((timestamp_start, timestamp_end))
                logger.info("[{}]: ({}, {})".format(len(buckets) - 1, timestamp_start, timestamp_end))
                gpu_utilizations.append(float(gpu_util_str) / 100)
            return buckets, gpu_utilizations


    def init_clock_time_to_timestamp(self):
        '''
        clock_time_start = "17:22:37.490296"
        self.timestamp_start = 1618996957490296
        clock_time_end = "17:23:07.178007"
        timestamp_end = 1618996987178007
        sample_period = 0.16666667
        gpu_util_file_path = "D:\\tmp\\gpu_utilization_lms1_mid.txt"
        '''
        clock_time_start = "19:35:40.064649"
        self.timestamp_start = 1619004940064649
        clock_time_end = "19:36:39.934758"
        timestamp_end = 1619004999934758
        sample_period = 0.16666667
        gpu_util_file_path = "D:\\tmp\\gpu_utilization\\log_l1_gpu_utilization.txt"

        self.clock_time_start_seconds = self.parse_clock_time(clock_time_start)
        clock_time_end_seconds = self.parse_clock_time(clock_time_end)
        self.timestamp_per_second = (timestamp_end - self.timestamp_start) / (clock_time_end_seconds - self.clock_time_start_seconds)
        self.timestamp_per_period = self.timestamp_per_second * sample_period

        buckets, gpu_utilizations = self.parse_lms1_file(gpu_util_file_path)

        print("gpu utilizations:")
        for gpu_util in gpu_utilizations:
            print(gpu_util)
        print("\n")

        return buckets, gpu_utilizations

    def clock_time_to_timestamp(self, clock_time_str):
        clock_time_seconds = self.parse_clock_time(clock_time_str)
        ts = (clock_time_seconds - self.clock_time_start_seconds) * self.timestamp_per_second + self.timestamp_start
        return ts

    def calculate_gpu_utilization(self):
        time_buckets, gpu_utilizations = self.init_clock_time_to_timestamp()

        min_square_error = 9999999.0
        best_latency = 0
        best_period = 0
        for latency in range(1):
            #time_latency = latency / 20
            time_latency = 0
            for period in range(1):
                #time_period = (period + 1) / 50
                time_period = 1.0
                print("\n")
                print("=================time_period:{}, latency:{}=================".format(time_period, time_latency))
                square_error = 0.0
                for bucket_id in range(len(time_buckets)):
                    bucket = time_buckets[bucket_id]
                    bucket = [bucket[1] - (bucket[1] - bucket[0]) * time_period - time_latency * (bucket[1] - bucket[0]),
                              bucket[1] - time_latency * (bucket[1] - bucket[0])]
                    count = 0
                    for r in self.role_ranges[ProfileRole.Kernel]:
                        if r[1] <= bucket[0] or r[0] >= bucket[1]:
                            continue
                        else:
                            left_bound = max(bucket[0], r[0])
                            right_bound = min(bucket[1], r[1])
                            count += (right_bound - left_bound)
                    utilization = count / (bucket[1] - bucket[0])
                    print("{}".format(utilization))
                    square_error += (gpu_utilizations[bucket_id] - utilization)**2
                print("square error = {}".format(square_error))
                if square_error < min_square_error:
                    min_square_error = square_error
                    best_latency = time_latency
                    best_period = time_period
        print("best_latency={}; best_period={}; min_square_error={}".format(
            best_latency, best_period, min_square_error))

