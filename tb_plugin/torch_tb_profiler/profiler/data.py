# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import gzip
import io as sysio
import json
import re
import tempfile
from collections import OrderedDict

from . import trace
from .kernel_parser import KernelParser
from .module_parser import ModuleParser
from .overall_parser import OverallParser, ProfileRole
from .trace import EventTypes
from .. import io, utils

logger = utils.get_logger()


class RunData(object):
    def __init__(self, name, run_dir):
        self.name = name
        self.run_dir = run_dir
        self.profiles = OrderedDict()


class RunProfileData(object):
    def __init__(self, worker):
        self.worker = worker
        self.data_schema_version = None
        self.events = None
        self.trace_file_path = None
        self.trace_json = None
        self.has_runtime = False
        self.has_kernel = False
        self.has_memcpy_or_memset = False
        self.steps_costs = None
        self.steps_names = None
        self.avg_costs = None
        self.runtime_node_list = None
        self.device_to_index = None
        self.gpu_utilization = None
        self.sm_efficency = None
        self.occupancy = None
        self.op_list_groupby_name = None
        self.op_list_groupby_name_input = None
        self.stack_lists_group_by_name = None
        self.stack_lists_group_by_name_input = None
        self.kernel_list_groupby_name_op = None
        self.kernel_stat = None
        self.recommendations = []

    @staticmethod
    def parse(run_dir, worker, path, caches):
        logger.debug("Parse trace, run_dir=%s, worker=%s", run_dir, path)

        trace_path, trace_json = RunProfileData._preprocess_file(caches, io.join(run_dir, path))

        profile = RunProfileData(worker)
        profile.trace_file_path = trace_path
        profile.trace_json = trace_json
        if type(trace_json) is dict:
            metadata = trace_json.get("profilerMetadata", None)
            version = metadata.get("DataSchemaVersion") if metadata else None
            profile.data_schema_version = version
            trace_json = trace_json["traceEvents"]

        profile.events = []
        for data in trace_json:
            event = trace.create_event(data)
            if event is not None:
                profile.events.append(event)

        return profile

    @staticmethod
    def _preprocess_file(caches, trace_path):
        if not io.exists(trace_path):
            raise FileNotFoundError(trace_path)

        data = caches.read(trace_path)
        if trace_path.endswith('.gz'):
            data = gzip.decompress(data)

        try:
            trace_json = json.loads(data)
        except json.decoder.JSONDecodeError as e:
            # Kineto may export json file with non-ascii code. before this is fixed, use a workaround
            # to handle JSONDecodeError, re-encode it and save to a temp file
            try:
                trace_json = json.loads(data, strict=False)
            except json.decoder.JSONDecodeError:
                with sysio.StringIO() as fout:
                    str_data = data.decode("utf-8")
                    # only replace the N/A without surrounding double quote
                    fout.write(re.sub(r'(?<!")N/A(?!")', "\"N/A\"", str_data))
                    trace_json = json.loads(fout.getvalue())

            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json.gz', delete=False)
            fp.close()
            with gzip.open(fp.name, mode='wt') as fzip:
                fzip.write(json.dumps(trace_json))
            logger.warning("Get JSONDecodeError: %s, Re-encode it to temp file: %s", e.msg, fp.name)
            trace_path = fp.name
            caches.add_tempfile(fp.name)

        return trace_path, trace_json

    def process(self):
        logger.debug("ModuleParser")
        module_parser = ModuleParser()
        module_parser.parse_events(self.events)
        self.op_list_groupby_name = module_parser.op_list_groupby_name
        self.op_list_groupby_name_input = module_parser.op_list_groupby_name_input
        self.stack_lists_group_by_name = module_parser.stack_lists_group_by_name
        self.stack_lists_group_by_name_input = module_parser.stack_lists_group_by_name_input
        self.kernel_list_groupby_name_op = module_parser.kernel_list_groupby_name_op

        logger.debug("OverallParser")
        overall_parser = OverallParser()
        overall_parser.parse_events(self.events, module_parser.runtime_node_list, module_parser.device_node_list)
        self.has_runtime = bool(overall_parser.role_ranges[ProfileRole.Runtime])
        self.has_kernel = bool(overall_parser.role_ranges[ProfileRole.Kernel])
        self.has_memcpy_or_memset = bool(overall_parser.role_ranges[ProfileRole.Memcpy] or overall_parser.role_ranges[ProfileRole.Memset])
        self.steps_costs = overall_parser.steps_costs
        self.steps_names = overall_parser.steps_names
        self.avg_costs = overall_parser.avg_costs
        self.runtime_node_list = module_parser.runtime_node_list
        self.device_to_index = overall_parser.device_to_index
        self.gpu_utilization = overall_parser.gpu_utilization
        self.sm_efficency = overall_parser.avg_approximated_sm_efficency_per_device
        self.occupancy = overall_parser.avg_occupancy_per_device
        self.trace_json["traceEvents"].extend(overall_parser.gpu_util_json)
        self.trace_json["traceEvents"].extend(overall_parser.gpu_sm_efficiency_json)
        fp = tempfile.NamedTemporaryFile('w+t', suffix='.json.gz', delete=False)
        fp.close()
        with gzip.open(fp.name, mode='wt') as fzip:
            fzip.write(json.dumps(self.trace_json))
        self.trace_file_path = fp.name
        self.trace_json = None  # Trace view loads from file, so release this to save memory usage.

        if self.has_kernel:
            logger.debug("KernelParser")
            kernel_parser = KernelParser()
            kernel_parser.parse_events(self.events)
            self.kernel_stat = kernel_parser.kernel_stat

    def analyze(self):
        def get_gpus_str(gpus):
            gpu_list_str = str(gpus[0])
            for i in range(1, len(gpus)):
                if i == len(gpus) - 1:
                    gpu_list_str += "and {}".format(gpus[i])
                else:
                    gpu_list_str += ", {}".format(gpus[i])
            has_str = "has" if len(gpu_list_str) == 1 else "have"
            return gpu_list_str, has_str

        self.recommendations = []

        dataloader_ratio = self.avg_costs.costs[ProfileRole.DataLoader] / self.avg_costs.costs[ProfileRole.Total]
        if dataloader_ratio > 0.05:
            text = "This run has high time cost on input data loading. " \
                   "{}% of the step time is in DataLoader. You could " \
                   "try to set num_workers on DataLoader's construction " \
                   "and enable multi-processes on data loading. " \
                   "Reference: <a href =\"{}\" target=\"_blank\">Single- and Multi-process Data Loading</a>".format(
                       round(dataloader_ratio * 100, 1),
                       "https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading"
                   )
            self.recommendations.append(text)

        low_util_gpus = []
        for device_id, index in self.device_to_index.items():
            if self.gpu_utilization[index] < 0.5:
                low_util_gpus.append(device_id)
        if len(low_util_gpus) > 0:
            gpu_list_str, has_str = get_gpus_str(low_util_gpus)
            text = "GPU {} {} low utilization. You could try to " \
                   "<a href =\"{}\" target=\"_blank\">enable async data loading and augmentation</a>, " \
                   "<a href =\"{}\" target=\"_blank\">optimize zero_grad</a>, " \
                   "<a href =\"{}\" target=\"_blank\">fuse pointwise operations</a>, " \
                   "increase batch-size by <a href =\"{}\" target=\"_blank\">checkpointing intermediate buffers</a>, " \
                   "<a href =\"{}\" target=\"_blank\">avoid unnecessary CPU-GPU synchronization</a>, " \
                   "<a href =\"{}\" target=\"_blank\">create tensors directly on the target device</a>, " \
                   "and so on".format(
                gpu_list_str, has_str,
                "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation",
                "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad",
                "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-pointwise-operations",
                "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#checkpoint-intermediate-buffers",
                "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#avoid-unnecessary-cpu-gpu-synchronization",
                "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#create-tensors-directly-on-the-target-device"
            )
            self.recommendations.append(text)

        if self.runtime_node_list is not None and len(self.runtime_node_list) > 0:
            total_kernels = 0
            short_kernels = 0
            for rt in self.runtime_node_list:
                if rt.device_nodes is not None:
                    for node in rt.device_nodes:
                        if node.type == EventTypes.KERNEL:
                            total_kernels += 1
                            if node.end_time - node.start_time < rt.end_time - rt.start_time:
                                short_kernels += 1
            if short_kernels / total_kernels > 0.5 and total_kernels > 100:
                text = "{} out of {} kernels are short in execution time. " \
                       "You could try to <a href =\"{}\" target=\"_blank\">optimize zero_grad</a>, " \
                       "or <a href =\"{}\" target=\"_blank\">fuse pointwise operations</a>.".format(
                    short_kernels, total_kernels,
                    "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad",
                    "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-pointwise-operations"
                )
            self.recommendations.append(text)

        low_sm_efficiency_gpus = []
        for device_id, index in self.device_to_index.items():
            if self.sm_efficency[index] > 0 and self.sm_efficency[index] < 0.8 * self.gpu_utilization[index]:
                low_sm_efficiency_gpus.append(device_id)
        if len(low_sm_efficiency_gpus) > 0:
            gpu_list_str, has_str = get_gpus_str(low_sm_efficiency_gpus)
            text = "GPU {} {} low estimated SM efficency. " \
                   "Many kernels' blocks of these GPU are so small that they can't fully utilization all multiprocessors." \
                   "You could try to increase the blocks number of these kernels.".format(
                gpu_list_str, has_str)
            self.recommendations.append(text)

        low_occupancy_gpus = []
        for device_id, index in self.device_to_index.items():
            if self.occupancy[index] > 0 and self.occupancy[index] < 50:
                low_occupancy_gpus.append(device_id)
        if len(low_occupancy_gpus) > 0:
            gpu_list_str, has_str = get_gpus_str(low_occupancy_gpus)
            text = "GPU {} {} low estimated achieved occupancy. " \
                   "The kernels on these GPU may occupy too much resource such as registers or shared memory, " \
                   "or their threads are not enough to fully utilize the multiprocessor." \
                   "Reference: <a href =\"{}\" target=\"_blank\">Achieved Occupancy</a>".format(
                gpu_list_str, has_str,
                "https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm"
            )
            self.recommendations.append(text)