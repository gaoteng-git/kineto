# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import sys
from enum import IntEnum
import json

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


EndpointTypes = IntEnum('EndpointTypes', ['START', 'END'], start=0)


class EndPoint(object):
    def __init__(self):
        self.time = 0
        self.pt_type = None
        self.value = 0.0

    def __init__(self, ep_time, ep_pt_type, ep_value):
        self.time = ep_time
        self.pt_type = ep_pt_type
        self.value = ep_value


# src_ranges: item of ((start_time, end_time), value)
def merge_ranges_with_value(src_ranges):
    merged_ranges = []
    if len(src_ranges) > 0:
        # Build tuple of (time, type, value)
        endpoints = []
        for r in src_ranges:
            endpoints.append(EndPoint(r[0][0], EndpointTypes.START, r[1]))
            endpoints.append(EndPoint(r[0][1], EndpointTypes.END, r[1]))
        endpoints.sort(key=lambda x: [x.time, int(x.pt_type)])  # Make START in front of END if equal on time.

        last_endpoint_time = endpoints[0].time
        last_value = endpoints[0].value
        for i in range(1, len(endpoints)):
            ep = endpoints[i]
            if ep.time > last_endpoint_time and last_value > 0.0:
                approximated_sm_efficiency = min(last_value, 1.0)
                merged_ranges.append(((last_endpoint_time, ep.time), approximated_sm_efficiency))
            last_endpoint_time = ep.time
            if ep.pt_type == EndpointTypes.START:
                last_value += ep.value
            else:
                last_value -= ep.value

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

        self.device_to_index = {}
        # For calculating GPU utilization.
        self.kernel_ranges_per_device = []
        self.gpu_utilization = []
        self.gpu_util_timeline_unit_size = 0
        self.gpu_util_timeline_unit_name = ""
        # For calculating approximated SM efficiency.
        self.blocks_per_sm_per_device = []
        self.avg_approximated_sm_efficency_per_device = []

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

    def calculate_gpu_utilization(self):
        def get_bucket_info(range_micro_seconds):
            max_buckets = 100
            bucket_size = 1
            while range_micro_seconds / bucket_size > max_buckets:
                bucket_size *= 10
            buckets = int(range_micro_seconds / bucket_size)
            unit = bucket_size
            unit_str = "us"
            if unit >= 1000:
                unit /= 1000
                unit_str = "ms"
                if unit >= 1000:
                    unit /= 1000
                    unit_str = "s"
            return int(bucket_size), int(buckets), int(unit), unit_str

        def build_trace_counter(gpu_id, start_time, counter_value):
            util_json = {}
            util_json["ph"] = "C"
            util_json["name"] = "GPU {} Utilization".format(gpu_id)
            util_json["pid"] = 0
            util_json["ts"] = start_time
            util_json["args"] = {"GPU Utilization": counter_value}
            return util_json

        counter_json = []
        all_steps_range = (self.steps[0][0], self.steps[-1][1])
        gpu_utilization_timeline = []
        for device_id, i in self.device_to_index.items():
            self.kernel_ranges_per_device[i] = merge_ranges(self.kernel_ranges_per_device[i])
            self.kernel_ranges_per_device[i] = intersection_ranges_lists(
                self.kernel_ranges_per_device[i], [all_steps_range])
            ranges_sum = get_ranges_sum(self.kernel_ranges_per_device[i])
            self.gpu_utilization.append(ranges_sum / (all_steps_range[1] - all_steps_range[0]))

            bucket_size, buckets, self.gpu_util_timeline_unit_size, self.gpu_util_timeline_unit_name = \
                get_bucket_info(all_steps_range[1] - all_steps_range[0])
            gpu_utilization_timeline.append([0] * buckets)
            if len(self.kernel_ranges_per_device[i]) > 0:
                current_range_index = 0
                current_range = self.kernel_ranges_per_device[i][current_range_index]
                current_bucket_index = 0
                current_bucket = (all_steps_range[0], all_steps_range[0] + bucket_size)
                while current_range_index < len(self.kernel_ranges_per_device[i]) and current_bucket_index < buckets:
                    if current_bucket[1] <= current_range[0]:
                        current_bucket_index += 1
                        current_bucket = (all_steps_range[0] + current_bucket_index * bucket_size,
                                          all_steps_range[0] + (current_bucket_index + 1) * bucket_size)
                    elif current_bucket[0] >= current_range[1]:
                        current_range_index += 1
                        if current_range_index < len(self.kernel_ranges_per_device[i]):
                            current_range = self.kernel_ranges_per_device[i][current_range_index]
                    else:
                        left_bound = max(current_range[0], current_bucket[0])
                        right_bound = min(current_range[1], current_bucket[1])
                        gpu_utilization_timeline[-1][current_bucket_index] += (right_bound - left_bound)
                        if current_bucket[1] < current_range[1]:
                            current_bucket_index += 1
                            current_bucket = (all_steps_range[0] + current_bucket_index * bucket_size,
                                              all_steps_range[0] + (current_bucket_index + 1) * bucket_size)
                        else:
                            current_range_index += 1
                            if current_range_index < len(self.kernel_ranges_per_device[i]):
                                current_range = self.kernel_ranges_per_device[i][current_range_index]
                for i_bucket in range(buckets):
                    gpu_utilization_timeline[-1][i_bucket] /= bucket_size

                for i_bucket in range(buckets):
                    start_time = self.steps[0][0] + i_bucket * bucket_size
                    util_json = build_trace_counter(device_id, start_time, gpu_utilization_timeline[-1][i_bucket])
                    counter_json.append(util_json)
                start_time = self.steps[0][0] + buckets * bucket_size
                util_json = build_trace_counter(device_id, start_time, 0)
                counter_json.append(util_json)

                self.kernel_ranges_per_device = None  # Release memory.

        return counter_json


    def output_gpu_utilization(self):
        for device_id, index in self.device_to_index.items():
            logger.info("gpu_utilization, GPU {}: {}".format(device_id, self.gpu_utilization[index]))


    def calculate_approximated_sm_efficency(self):
        def calculate_avg(approximated_sm_efficency_ranges, total_dur):
            total_weighted_sm_efficiency = 0.0
            for r in approximated_sm_efficency_ranges:
                dur = r[0][1] - r[0][0]
                total_weighted_sm_efficiency += r[1] * dur
            avg_approximated_sm_efficency = total_weighted_sm_efficiency / total_dur
            return avg_approximated_sm_efficency

        def build_trace_counter(gpu_id, start_time, counter_value):
            util_json = {}
            util_json["ph"] = "C"
            util_json["name"] = "GPU {} Est. SM Efficiency".format(gpu_id)
            util_json["pid"] = 0
            util_json["ts"] = start_time
            util_json["args"] = {"Est. SM Efficiency": counter_value}
            return util_json

        counter_json = []
        total_dur = self.steps[-1][1] - self.steps[0][0]
        approximated_sm_efficency_ranges_per_device = []
        for device_id, i in self.device_to_index.items():
            blocks_per_sm_ranges = self.blocks_per_sm_per_device[i]
            approximated_sm_efficency_ranges = merge_ranges_with_value(blocks_per_sm_ranges)
            approximated_sm_efficency_ranges_per_device.append(approximated_sm_efficency_ranges)
            avg_approximated_sm_efficency = calculate_avg(approximated_sm_efficency_ranges, total_dur)
            self.avg_approximated_sm_efficency_per_device.append(avg_approximated_sm_efficency)

            for r in approximated_sm_efficency_ranges:
                efficiency_json_start = build_trace_counter(device_id, r[0][0], r[1])
                efficiency_json_finish = build_trace_counter(device_id, r[0][1], 0)
                counter_json.append(efficiency_json_start)
                counter_json.append(efficiency_json_finish)

        self.blocks_per_sm_per_device = None  # Release memory.

        return counter_json


    def output_approximated_sm_efficency(self):
        for device_id, index in self.device_to_index.items():
            logger.info("approximated_sm_efficency, GPU {}: {}".format(
                device_id, self.avg_approximated_sm_efficency_per_device[index]))
            for r in self.approximated_sm_efficency_ranges_per_device[index]:
                print("({},{}): {}".format(r[0][0], r[0][1], r[1]))
            self.approximated_sm_efficency_ranges_per_device[index].sort(key=lambda x : (x[0][1] - x[0][0]))
            for r in self.approximated_sm_efficency_ranges_per_device[index]:
                print("{}\t : {}".format(r[0][1] - r[0][0], r[1]))


    def parse_events(self, events, runtime_node_list, device_node_list):
        logger.debug("Overall, parse events")
        for event in events:
            self.parse_event(event)

        if len(self.steps) == 0:
            self.steps.append((self.min_ts, self.max_ts))
            self.steps_names.append("0")
        self.update_steps_consider_device_side(runtime_node_list, device_node_list)

        gpu_util_json = self.calculate_gpu_utilization()
        #self.output_gpu_utilization()
        gpu_sm_efficiency_json = self.calculate_approximated_sm_efficency()
        #self.output_approximated_sm_efficency()

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

        return gpu_util_json, gpu_sm_efficiency_json

    def parse_event(self, event):
        ts = event.ts
        dur = event.duration
        evt_type = event.type
        if evt_type == EventTypes.KERNEL:
            self.role_ranges[ProfileRole.Kernel].append((ts, ts + dur))
            # For caculating GPU utilization, and approximated SM efficiency.
            device_id = event.args.get("device", None)
            if device_id not in self.device_to_index:
                index = len(self.kernel_ranges_per_device)
                self.device_to_index[device_id] = index
                self.kernel_ranges_per_device.append([])
                self.blocks_per_sm_per_device.append([])
            index = self.device_to_index[device_id]
            self.kernel_ranges_per_device[index].append((ts, ts + dur))
            self.blocks_per_sm_per_device[index].append(((ts, ts + dur), event.args.get("blocks per SM", 0.0)))
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
