# utils/marl_utils.py
from typing import List, Deque, Dict, Any
import numpy as np
import pandas as pd
from rl_components.task import Task # Assuming this is the correct path from root

# --- Define the fixed order for feature vectors ---
# This ensures consistency when converting dicts to numpy arrays for the NN.
MANAGER_META_TASK_FEATURE_ORDER = [
    "num_tasks",
    "total_cores_req",
    "total_gpu_req",
    "total_mem_req", # Worker might also care about this for local packing
    "total_bandwidth_gb", # Worker might also care about this for local packing
    "max_duration_mins", # This is the max_duration
    "max_sla_urgency"
]

WORKER_META_TASK_FEATURE_ORDER = [
    "num_tasks",
    "total_cores_req",
    "total_gpu_req",
    "total_mem_req", # Worker might also care about this for local packing
    "total_bandwidth_gb", # Worker might also care about this for local packing
    "max_duration_mins", # This is the max_duration
    "max_sla_urgency"
]

# --- Derive descriptor dimensions from the feature order lists ---
D_META_MANAGER = len(MANAGER_META_TASK_FEATURE_ORDER)
D_META_WORKER = len(WORKER_META_TASK_FEATURE_ORDER)

def aggregate_tasks_for_manager(task_queue: Deque[Task], current_time_utc: pd.Timestamp) -> np.ndarray:
    """
    Aggregates tasks from the DTA_Manager's Originating_Tasks_Queue
    into a fixed-size descriptor vector.
    """
    '''
    No, it is not expected that _get_observations (and by extension _prepare_all_manager_observations) is called multiple times per global_step. This is inefficient and, more importantly, it's the direct cause of the bug you identified: the meta-task is always zeros.
    The Problem: The Flawed manager_step/worker_step/env_step Sequence
    '''
    # print(f'Fix the logic calling 3 times to aggregate_tasks_for_manager from:')
    # print(f'1. manager_step: {task_queue}')
    # print(f'2. worker_step: {task_queue}')
    # print(f'3. env_step: {task_queue}')
    # print(f'Current time UTC: {current_time_utc}')
    

    num_tasks = len(task_queue)
    if num_tasks == 0:
        return np.zeros(D_META_MANAGER, dtype=np.float32)

    # Calculate SLA Urgency: 1 / (time_to_deadline_hours + epsilon)
    # Higher value means more urgent.
    urgencies = []
    for task in task_queue:
        time_to_deadline_mins = (task.sla_deadline - current_time_utc).total_seconds() / 60.0
        if time_to_deadline_mins <= 1: # If due now or past due
            urgencies.append(1000.0) # Assign a large, fixed urgency
        else:
            urgencies.append(1.0 / time_to_deadline_mins)

    descriptor_dict = {
        "num_tasks": float(num_tasks),
        "total_cores_req": sum(t.cores_req for t in task_queue),
        "total_gpu_req": sum(t.gpu_req for t in task_queue),
        "total_mem_req": sum(t.mem_req for t in task_queue),
        "total_bandwidth_gb": sum(t.bandwidth_gb for t in task_queue),
        "avg_duration_mins": np.mean([t.duration for t in task_queue]),
        "max_duration_mins": max(t.duration for t in task_queue),
        "max_sla_urgency": max(urgencies) if urgencies else 0.0
    }

    return np.array([descriptor_dict[key] for key in MANAGER_META_TASK_FEATURE_ORDER], dtype=np.float32)

def aggregate_tasks_for_worker(task_queue: Deque[Task], current_time_utc: pd.Timestamp) -> np.ndarray:
    """
    Aggregates tasks from the DTA_Worker's Worker_Commitment_Queue
    into a fixed-size descriptor vector.
    """
    num_tasks = len(task_queue)
    if num_tasks == 0:
        return np.zeros(D_META_WORKER, dtype=np.float32)

    urgencies, wait_intervals = [], []
    for task in task_queue:
        time_to_deadline_mins = (task.sla_deadline - current_time_utc).total_seconds() / 60.0
        if time_to_deadline_mins <= 1:
            urgencies.append(1000.0)
        else:
            urgencies.append(1.0 / time_to_deadline_mins)
        wait_intervals.append(task.wait_intervals)
    
    descriptor_dict = {
        "num_tasks": float(num_tasks),
        "total_cores_req": sum(t.cores_req for t in task_queue),
        "total_gpu_req": sum(t.gpu_req for t in task_queue),
        "total_mem_req": sum(t.mem_req for t in task_queue),
        "avg_duration_mins": np.mean([t.duration for t in task_queue]),
        "avg_wait_intervals": np.mean(wait_intervals) if wait_intervals else 0.0,
        "max_wait_intervals": float(max(wait_intervals)) if wait_intervals else 0.0,
        "max_sla_urgency": max(urgencies) if urgencies else 0.0
    }

    return np.array([descriptor_dict[key] for key in WORKER_META_TASK_FEATURE_ORDER], dtype=np.float32)