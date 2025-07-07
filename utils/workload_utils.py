import math
import logging
import numpy as np
import pandas as pd
from rl_components.task import Task

def assign_task_origins(tasks, datacenter_configs, current_time_utc, logger=None):
    """
    Assigns each task an origin datacenter (DC), based on:
    - Population weight of the DC
    - Local time activity (higher weight during 8am–8pm local time)

    Args:
        tasks (List[Task]): List of Task objects to assign.
        datacenter_configs (List[dict]): DC configuration including weights and timezone offsets.
        current_time_utc (datetime): Current simulation time in UTC.
        logger (logging.Logger or None): Optional logger for debug output.
    """
    def compute_activity_score(local_hour):
        # Simulate typical activity pattern: peak between 8am and 8pm local time
        return 1.0 if 8 <= local_hour < 20 else 0.3

    # Step 1: Calculate scores based on population × activity
    scores = {}
    for config in datacenter_configs:
        dc_id = config['dc_id']
        pop_weight = config.get('population_weight', 0.1)
        tz_shift = config.get('timezone_shift', 0)
        local_hour = (current_time_utc + pd.Timedelta(hours=tz_shift)).hour
        scores[dc_id] = pop_weight * compute_activity_score(local_hour)

    # Step 2: Normalize scores into probabilities
    total_score = sum(scores.values())
    probabilities = {dc_id: score / total_score for dc_id, score in scores.items()}

    # Step 3: Assign origin DC to each task
    dc_ids = list(probabilities.keys())
    probs = list(probabilities.values())

    for task in tasks:
        origin_dc_id = int(np.random.choice(dc_ids, p=probs))
        task.origin_dc_id = origin_dc_id

        if logger:
            logger.debug(f"Task {task.job_name} assigned origin DC{origin_dc_id}.")


def extract_tasks_from_row(row, scale=5, datacenter_configs=None,
                           current_time_utc=None, logger=None,
                           task_scale: int = 5, # <<<--- NEW PARAMETER
                           group_size: int = 1): # <<<--- NEW PARAMETER
    """
    Convert a row from task_df into a list of Task objects, scaling the number of tasks if needed.

    Args:
        row (pd.Series): A row from task_df containing 'tasks_matrix'.
        scale (int): Scaling factor for task duplication.
        datacenter_configs (List[dict]): DC configurations for assigning task origins.
        current_time_utc (datetime): Current simulation time in UTC.
        logger (logging.Logger or None): Optional logger for debug statements.
        group_size (int): Number of consecutive tasks to group into one meta-task.
                          Defaults to 1 (no grouping).

    Returns:
        List[Task]: A list of Task objects extracted and scaled from the row.
    """
    if group_size < 1:
        group_size = 1 # Ensure group size is at least 1
        
    task_scale = task_scale  # To simulate tasks that are 5 times larger than the original
    individual_tasks = []

    # --- Step 1: Extract and scale individual tasks FIRST ---
    for task_data in row['tasks_matrix']:
        job_name = task_data[0]
        arrival_time = current_time_utc  # Task arrival time
        duration = float(task_data[4])
        # Apply task_scale during initial extraction
        cores_req = task_scale * float(task_data[5]) / 100.0
        gpu_req = task_scale * float(task_data[6]) / 100.0
        mem_req = task_scale * float(task_data[7])
        bandwidth_gb = float(task_data[8]) # Bandwidth isn't scaled by task_scale here

        # Create the original task object (will get origin assigned later)
        task = Task(job_name, arrival_time, duration, cores_req, gpu_req, mem_req, bandwidth_gb)
        individual_tasks.append(task)

        # Create scaled/augmented versions (if scale > 1)
        # Note: These scaled tasks will also be part of the grouping later
        for i in range(np.random.randint(0, scale)):
            varied_duration = max(0.1, duration * np.random.uniform(0.8, 1.2))  # Ensure duration is not zero
            varied_cpu = max(0.5, cores_req * np.random.uniform(0.8, 1.2))
            varied_gpu = max(0.0, gpu_req * np.random.uniform(0.8, 1.2))
            varied_mem = max(0.5, mem_req * np.random.uniform(0.8, 1.2))
            varied_bw = max(0.1, bandwidth_gb * np.random.uniform(0.8, 1.2))
            new_task = Task(
                f"{job_name}_scaled_{i}", arrival_time, varied_duration,
                varied_cpu, varied_gpu, varied_mem, varied_bw
            )
            individual_tasks.append(new_task)

    # --- Step 2: Assign Origins to ALL individual tasks ---
    # This is crucial BEFORE grouping, so we know the 'first' origin
    if datacenter_configs and current_time_utc and individual_tasks:
        assign_task_origins(individual_tasks, datacenter_configs, current_time_utc, logger=logger)

    # --- Step 3: Group tasks if group_size > 1 ---
    final_tasks_list = []
    if group_size == 1:
        final_tasks_list = individual_tasks # No grouping needed
    elif individual_tasks: # Only group if there are tasks
        if logger:
            logger.info(f"Grouping {len(individual_tasks)} tasks into groups of {group_size}")
        num_groups = math.ceil(len(individual_tasks) / group_size)

        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            current_group = individual_tasks[start_idx:end_idx]

            if not current_group:
                continue # Should not happen, but safety check

            # Aggregate properties
            agg_cores_req = sum(t.cores_req for t in current_group)
            agg_gpu_req = sum(t.gpu_req for t in current_group)
            agg_mem_req = sum(t.mem_req for t in current_group)
            agg_bandwidth_gb = sum(t.bandwidth_gb for t in current_group) # Sum total data
            # Duration: Use the maximum duration within the group
            agg_duration = max(t.duration for t in current_group)
            # SLA Deadline: Use the *earliest* deadline within the group
            agg_sla_deadline = max(t.sla_deadline for t in current_group)
            # Origin: Use the origin of the *first* task in the group
            group_origin_dc_id = current_group[0].origin_dc_id
            # Job Name: Create a composite name
            group_job_name = f"Group_{i+1}_({current_group[0].job_name})"
            # Arrival time is the same for all in this implementation
            group_arrival_time = current_group[0].arrival_time

            # Create the aggregated meta-task
            meta_task = Task(
                job_name=group_job_name,
                arrival_time=group_arrival_time,
                duration=agg_duration,
                cores_req=agg_cores_req,
                gpu_req=agg_gpu_req,
                mem_req=agg_mem_req,
                bandwidth_gb=agg_bandwidth_gb
            )
            
            # Assign the aggregated/chosen properties
            meta_task.origin_dc_id = group_origin_dc_id
            meta_task.sla_deadline = agg_sla_deadline # Set the earliest deadline

            final_tasks_list.append(meta_task)

            if logger:
                logger.debug(f"  Group[{i}]: {meta_task.job_name} | origin=DC{meta_task.origin_dc_id} | "
                             f"CPU={meta_task.cores_req:.2f}, GPU={meta_task.gpu_req:.2f}, MEM={meta_task.mem_req:.2f}, "
                             f"BW={meta_task.bandwidth_gb:.2f}, duration={meta_task.duration:.2f}, "
                             f"SLA={meta_task.sla_deadline}")


    # --- Logging ---
    if logger:
        log_level = logging.INFO if group_size == 1 else logging.DEBUG # Log details only if grouping
        logger.log(log_level, f"extract_tasks_from_row: Returning {len(final_tasks_list)} tasks/groups (group_size={group_size}) at {current_time_utc}.")
        # Log details of the final list if debugging grouping
        if group_size > 1:
             for idx, t in enumerate(final_tasks_list):
                   logger.debug(
                       f"  FinalTask[{idx}]: {t.job_name} | origin=DC{t.origin_dc_id} | "
                       f"CPU={t.cores_req:.2f}, GPU={t.gpu_req:.2f}, MEM={t.mem_req:.2f}, "
                       f"BW={t.bandwidth_gb:.2f}, duration={t.duration:.2f}, SLA={t.sla_deadline}"
                   )


    return final_tasks_list