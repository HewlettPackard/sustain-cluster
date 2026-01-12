# utils/llm_serialization.py

import pandas as pd
import numpy as np
import traceback

class EnvSerializer:
    def __init__(self, cluster_manager):
        self.cluster_manager = cluster_manager
        self.dc_names = [dc.dc_id for dc in cluster_manager.datacenters.values()]

    def _get_relative_rank(self, value, all_values, lower_is_better=True):
        if len(all_values) <= 1: return ""
        min_val, max_val, mean_val = min(all_values), max(all_values), np.mean(all_values)
        
        if max_val == min_val: return "(Avg)"
        if value == min_val: return "(Best)" if lower_is_better else "(Worst)"
        if value == max_val: return "(Worst)" if lower_is_better else "(Best)"
        
        if lower_is_better:
            return "(<Avg)" if value < mean_val else "(>Avg)"
        else:
            return "(>Avg)" if value > mean_val else "(<Avg)"

    def _get_trend(self, manager, attr_name, threshold):
        """
        Generic lookahead that handles both CI_Manager and ElectricityPrice_Manager.
        """
        try:
            lookahead_steps = 4 # 1 hour
            
            # 1. Determine Current Index and Data Source
            if hasattr(manager, 'index'): 
                # Case: ElectricityPrice_Manager
                current_step = manager.index
            elif hasattr(manager, 'time_step'): 
                # Case: CI_Manager
                current_step = manager.time_step
            else:
                print("Unknown manager type for trend analysis.")
                return "Stable"

            # 2. Get Data Array
            data_array = getattr(manager, attr_name, None)
            
            # 3. Validation
            if data_array is None or current_step + lookahead_steps >= len(data_array):
                print("Insufficient data for trend analysis.")
                return "Stable"

            # 4. Calculate Trend
            curr = data_array[current_step]
            fut = data_array[current_step + lookahead_steps]
            diff = fut - curr
            
            if diff > threshold: return f"Rising (+{diff:.1f})"
            elif diff < -threshold: return f"Falling ({diff:.1f})"
            return "Stable"
        except Exception:
            print("Error during trend analysis.")
            traceback.print_exc()
            return "Stable"

    def _get_cooling_status(self, dc):
        try:
            # Use norm=False to get degrees Celsius
            temp_c = dc.weather_manager.get_current_temperature(norm=False)
            if temp_c < 10: status = "Free Cooling"
            elif temp_c < 25: status = "Moderate"
            else: status = "High Load"
            return f"{temp_c:.1f}C [{status}]"
        except:
            return "Unknown"

    def serialize_state(self, env, current_tasks):
        # 1. TIME
        t = env.current_time
        time_desc = f"{t.month_name()} {t.day} ({t.day_name()}) at {t.hour:02d}:00"
        
        # 2. AGGREGATED WORKLOAD REQUIREMENTS
        num_tasks = len(current_tasks)
        if num_tasks == 0: return None
            
        req_cpu = sum(t.cores_req for t in current_tasks)
        req_gpu = sum(t.gpu_req for t in current_tasks)
        req_mem = sum(t.mem_req for t in current_tasks)
        
        # Workload Type
        if req_gpu > 0: wkld_type = "AI Training (GPU)"
        elif req_cpu / num_tasks > 30: wkld_type = "HPC (CPU)"
        else: wkld_type = "Web (Light)"

        avg_duration_hr = np.mean([t.duration for t in current_tasks]) / 60.0

        # 3. GATHER DATA & RELATIVE RANKS
        dcs = list(self.cluster_manager.datacenters.values())
        ci_vals = [dc.ci_manager.get_current_ci(norm=False) for dc in dcs]
        price_vals = [dc.price_manager.get_current_price() for dc in dcs]
        
        # 4. GENERATE DC DESCRIPTIONS
        dc_descriptions = []
        for i, dc in enumerate(dcs):
            # Metrics
            c_val, p_val = ci_vals[i], price_vals[i]
            c_rank = self._get_relative_rank(c_val, ci_vals, True)
            p_rank = self._get_relative_rank(p_val, price_vals, True)
            
            # Trends
            # CI Manager uses 'carbon_smooth'
            c_trend = self._get_trend(dc.ci_manager, 'carbon_smooth', 5.0)
            # Price Manager uses 'prices' (Updated from your code)
            p_trend = self._get_trend(dc.price_manager, 'prices', 0.01)
            
            # Cooling
            cooling = self._get_cooling_status(dc)
            
            # ABSOLUTE CAPACITY CHECK
            avail_cpu = max(0, dc.available_cores)
            avail_gpu = max(0, dc.available_gpus)
            avail_mem = max(0, dc.available_mem)
            
            # TOTAL RESOURCES INSTALLED
            total_cpu = dc.total_cores
            total_gpu = dc.total_gpus
            total_mem = dc.total_mem_GB
            
            # % of RESOURCES USED
            used_cpu_pct = 100 - 100*avail_cpu / total_cpu if total_cpu > 0 else 0
            used_gpu_pct = 100 - 100*avail_gpu / total_gpu if total_gpu > 0 else 0
            used_mem_pct = 100 - 100*avail_mem / total_mem if total_mem > 0 else 0
            
            # Constraint Tag
            fits_cpu = avail_cpu >= req_cpu
            fits_gpu = avail_gpu >= req_gpu
            fits_mem = avail_mem >= req_mem
            
            if fits_cpu and fits_gpu and fits_mem:
                cap_status = "Sufficient"
            else:
                missing = []
                if not fits_cpu: missing.append("CPU")
                if not fits_gpu: missing.append("GPU")
                if not fits_mem: missing.append("Mem")
                cap_status = f"INSUFFICIENT ({','.join(missing)})"

            desc = (
                f"- DC_{i} ({dc.location}):\n"
                f"    Carbon:   {c_val:.0f} g/kWh {c_rank} (Trend: {c_trend})\n"
                f"    Price:    ${p_val:.2f} {p_rank} (Trend: {p_trend})\n"
                f"    Cooling:  {cooling}\n"
                f"    Total Resources: {total_cpu:.0f} CPU / {total_gpu:.0f} GPU / {total_mem:.0f}GB Mem\n"
                f"    Capacity: {avail_cpu:.0f} CPU / {avail_gpu:.0f} GPU / {avail_mem:.0f}GB Mem [{cap_status}]\n"
                f"    Used:    {used_cpu_pct:.0f}% CPU | {used_gpu_pct:.0f}% GPU | {used_mem_pct:.0f}% Mem"
            )
            dc_descriptions.append(desc)

        # 5. FINAL PROMPT
        prompt = (
            f"Current Time: {time_desc}\n"
            f"Batch Requirements: {num_tasks} tasks ({wkld_type}).\n"
            f"   Needed: {req_cpu:.1f} vCPU | {req_gpu:.1f} GPU | {req_mem:.1f} GB Mem\n"
            f"   Duration: ~{avg_duration_hr:.1f} hours\n"
            f"Cluster Status:\n" + "\n".join(dc_descriptions) + "\n"
            f"Instruction: Analyze Carbon, Price, and Capacity Constraints. "
            f"Select the optimal Data Center ID (0-{len(dcs)-1})."
        )
        return prompt

    def parse_action(self, llm_response):
        import re
        matches = re.findall(r'\d+', llm_response)
        if matches: return int(matches[-1])
        return 0