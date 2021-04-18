# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

import pandas as pd


class KernelParser:
    def __init__(self):
        self.kernel_stat = None

    def parse_events(self, events):
        events_dict = []
        for event in events:
            events_dict.append(vars(event))
        events = events_dict
        self.dump_occupancy_sm_efficiency(events)
        events = pd.DataFrame(events)
        events = events.astype({"type": "category", "category": "category", "name": "string"}, copy=False)
        kernels = events[events["category"] == "Kernel"]
        self.kernel_stat = kernels.groupby("name")["duration"].agg(["count", "sum", "mean", "max", "min"]) \
            .sort_values("sum", ascending=False)


    def dump_occupancy_sm_efficiency(self, events):
        file = open("occupancy_sm_efficiency.json", "w")
        last_kernel_ts = 0
        for event in events:
            if (event["category"] == "Kernel") and ("ts" in event) and ("duration" in event):
                begin_ts = event["ts"]
                end_ts = event["ts"] + event["duration"]
                if begin_ts >= last_kernel_ts:
                    last_kernel_ts = end_ts
                    occupancy_begin = '{{"ph": "C", "name": "Occupancy", "pid": 0, "tid": "stream 7", "ts": {}, "args": {{"Occupancy": {}}}}},'.format(
                        begin_ts, event["args"]["occupancy"])
                    occupancy_end = '{{"ph": "C", "name": "Occupancy", "pid": 0, "tid": "stream 7", "ts": {}, "args": {{"Occupancy": {}}}}},'.format(
                        end_ts, 0)
                    sm_utilization_begin = '{{"ph": "C", "name": "SM_utilization", "pid": 0, "tid": "stream 7", "ts": {}, "args": {{"SM_utilization": {}}}}},'.format(
                        begin_ts, event["args"]["sm_efficiency"])
                    sm_utilization_end = '{{"ph": "C", "name": "SM_utilization", "pid": 0, "tid": "stream 7", "ts": {}, "args": {{"SM_utilization": {}}}}},'.format(
                        end_ts, 0)
                    file.write(occupancy_begin + "\n")
                    file.write(occupancy_end + "\n")
                    file.write(sm_utilization_begin + "\n")
                    file.write(sm_utilization_end + "\n")
        file.close()