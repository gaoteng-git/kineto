# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------
import numpy as np
import pandas as pd

from .trace import EventTypes


class KernelParser:
    def __init__(self):
        self.kernel_stat = None

    def parse_events(self, events):
        events_dict = []
        for event in events:
            if event.type == EventTypes.KERNEL:
                events_dict.append(vars(event))
                events_dict[-1]["grid"] = str(event.args.get("grid", "None"))
                events_dict[-1]["block"] = str(event.args.get("block", "None"))
                events_dict[-1]["regs_per_thread"] = str(event.args.get("registers per thread", 0))
                events_dict[-1]["shared_memory"] = str(event.args.get("shared memory", 0))
                events_dict[-1]["blocks_per_sm"] = event.args.get("blocks per SM", 0)
                events_dict[-1]["occupancy"] = event.args.get("est. achieved occupancy %", 0)
        events = events_dict
        events = pd.DataFrame(events)
        events = events.astype({"type": "category", "category": "category", "name": "string",
                                "grid": "string", "block": "string",
                                "regs_per_thread": "string", "shared_memory": "string"}, copy=False)

        def weighted_avg(x):
            try:
                return np.average(x, weights=events.loc[x.index, "duration"])
            except ZeroDivisionError:
                return 0

        self.kernel_stat = events.groupby(["name", "grid", "block", "regs_per_thread", "shared_memory"]).agg(
            count=('duration', "count"),
            sum=('duration', "sum"),
            mean=('duration', "mean"),
            max=('duration', "max"),
            min=('duration', "min"),
            blocks_per_sm=('blocks_per_sm', weighted_avg),
            occupancy=('occupancy', weighted_avg))\
            .sort_values("sum", ascending=False)