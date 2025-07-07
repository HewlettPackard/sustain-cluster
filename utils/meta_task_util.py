# utils/meta_task_util.py
from typing import List
import pandas as pd
from rl_components.task import Task

def create_meta_task_from_group(task_group: List[Task], dc_id: int, current_time: pd.Timestamp) -> Task:
    """
    Aggregates a list of tasks into a single representative "meta-task".
    """
    if not task_group:
        # Return an empty task if the group is empty
        return Task(
            job_name=f"meta_empty_{dc_id}_{int(current_time.timestamp())}",
            arrival_time=current_time,
            duration=0,
            cores_req=0,
            gpu_req=0,
            mem_req=0,
            bandwidth_gb=0,
            sla_deadline=current_time,  # SLA deadline is now
            origin_dc_id=dc_id
        )

    # Use the logic we designed above
    total_cores = sum(t.cores_req for t in task_group)
    total_gpus = sum(t.gpu_req for t in task_group)
    total_mem = sum(t.mem_req for t in task_group)
    total_bw = sum(t.bandwidth_gb for t in task_group)
    
    max_duration = max(t.duration for t in task_group)
    most_urgent_sla = min(t.sla_deadline for t in task_group)

    # The arrival time of the meta-task is now.
    arrival = current_time
    
    # Create the new Task object. Note that its SLA deadline is NOT calculated
    # from its own duration, but inherited from the most urgent task.
    meta_task = Task(
        job_name=f"meta_{dc_id}_{int(current_time.timestamp())}",
        arrival_time=arrival,
        duration=max_duration,
        cores_req=total_cores,
        gpu_req=total_gpus,
        mem_req=total_mem,
        bandwidth_gb=total_bw
    )
    
    # Manually override the calculated SLA deadline with the correct one.
    meta_task.sla_deadline = most_urgent_sla
    meta_task.origin_dc_id = dc_id
    
    return meta_task