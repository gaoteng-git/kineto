import os
import time
import unittest
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from tensorboard_plugin_torch_profiler.profiler import RunLoader

def create_log_dir():
    log_dir_name='./log{}'.format(str(int(time.time()*1000)))
    try:
        os.makedirs(log_dir_name)
    except Exception:
        raise RuntimeError("Can't create directory: " + log_dir_name)
    return log_dir_name

def get_autograd_result(p, worker_name):
    avgs = p.key_averages()
    sort_by = 'self_cuda_time_total'
    avgs = sorted(
        avgs, key=lambda evt: getattr(evt, sort_by), reverse=True
    )
    is_gpu = False
    if avgs[0].self_cuda_time_total > 0:
        is_gpu = True
    others_prefix = {"enumerate(DataLoader)#", "Optimizer.zero_grad#", "Optimizer.step#",
                     "ProfilerStep*",
                     "Memcpy ", "Memset ",
                     "cuda"}
    postfix_to_type = {"CPU": "operator", "CUDA": "kernel"}

    def get_type(evt):
        s = str(evt.device_type)
        postfix = s[s.index('.') + 1:]
        evt_type = postfix_to_type[postfix]
        for prefix in others_prefix:
            if evt.key.startswith(prefix):
                evt_type = "Other"
                break
        return evt_type

    result_dict = dict()
    result_dict[worker_name + "#operator"] = list()
    result_dict[worker_name + "#kernel"] = list()
    for avg in avgs:
        evt_type = get_type(avg)
        if evt_type == "operator":
            line = [avg.key, int(avg.count)]
            if is_gpu:
                line.extend([int(avg.self_cuda_time_total), int(avg.cuda_time_total)])
            line.extend([int(avg.self_cpu_time_total), int(avg.cpu_time_total)])
            result_dict[worker_name + "#operator"].append(line)
        elif is_gpu and evt_type == "kernel":
            line = [avg.key, int(avg.count), int(avg.self_cuda_time_total)]
            result_dict[worker_name + "#kernel"].append(line)
    return result_dict

def get_plugin_result(run):
    result_dict =  dict()
    for worker_name, profile in run.profiles.items():
        if not profile.operation_table_by_name is None:
            rows = profile.operation_table_by_name["data"]["rows"]
            result_dict[worker_name + "#operator"] = rows
            if not profile.kernel_table is None:
                rows = profile.kernel_table["data"]["rows"]
                result_dict[worker_name + "#kernel"] = list()
                for row in rows:
                    result_dict[worker_name + "#kernel"].append(row[:3])
    return result_dict

def get_train_func():
    model = models.resnet50(pretrained=True)
    model.cuda()
    cudnn.benchmark = True

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda:0")
    model.train()

    def train(train_step, prof=None):
        for step, data in enumerate(trainloader, 0):
            print("step:{}".format(step))
            inputs, labels = data[0].to(device=device), data[1].to(device=device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if prof is not None:
                prof.step()
            if step >= train_step:
                break
    return train

def get_output_fn(dir_name, profilers_dict):
    def output_fn(p):
        # In current torch.profiler.profile, at beginning of each span, a new p.profiler will be created.
        # So the same p.profiler will not be shared among different spans
        profilers_dict["worker{}".format(p.step_num)] = p.profiler
        p.export_chrome_trace(os.path.join(dir_name, "worker{}.pt.trace.json".format(p.step_num)))
    return output_fn

class TestCompareWithAutogradResult(unittest.TestCase):

    def compare_results(self, log_dir, profilers_dict, use_gpu=True):
        loader = RunLoader(os.path.split(log_dir)[-1], log_dir)
        run = loader.load()
        plugin_result = get_plugin_result(run)
        count = 0
        for worker_name, p in profilers_dict.items():
            autograd_result = get_autograd_result(p, worker_name)
            for key in autograd_result.keys():
                count += 1
                self.assertTrue(key in plugin_result.keys())
                self.assertEqual(len(plugin_result[key]), len(autograd_result[key]))
                for line in plugin_result[key]:
                    if not use_gpu:
                        line = line[0:2]+line[4:]
                    self.assertTrue(line in autograd_result[key])
        self.assertEqual(count, len(plugin_result.keys()))

    def test_autograd_api(self):
        with torch.autograd.profiler.profile(use_cuda=True, use_kineto=True, record_shapes=True) as p:
            get_train_func()(5)
        log_dir = create_log_dir()
        p.export_chrome_trace(os.path.join(log_dir, "worker0.pt.trace.json"))
        self.compare_results(log_dir, {"worker0":p})

    def base_profiler_api(self, use_gpu, record_shapes, profile_memory, with_stack):
        log_dir = create_log_dir()
        profilers_dict = dict()
        if use_gpu:
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]
        else:
            activities=[torch.profiler.ProfilerActivity.CPU]

        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=3),
            on_trace_ready=get_output_fn(log_dir, profilers_dict),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack
        ) as p:
            get_train_func()(13, p)
        self.compare_results(log_dir, profilers_dict, use_gpu)

    def test_profiler_api_without_gpu(self):
        self.base_profiler_api(False, True, True, False)

    def test_profiler_api_with_record_shapes_memory_stack(self):
        self.base_profiler_api(True, True, True, True)

    def test_profiler_api_without_record_shapes_memory_stack(self):
        self.base_profiler_api(True, False, False, False)

    def test_profiler_api_without_step(self):
        log_dir = create_log_dir()
        profilers_dict = dict()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=get_output_fn(log_dir, profilers_dict),
            record_shapes=True
        ):
            get_train_func()(7)
        self.compare_results(log_dir, profilers_dict)

