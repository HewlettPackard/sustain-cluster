# data_processing/augment_dataset.py
import json
import re
import os

def parse_dc_stats(prompt):
    stats = {}
    chunks = prompt.split("- DC_")
    
    for chunk in chunks[1:]:
        try:
            # 1. ID
            header_match = re.match(r"^(\d+)", chunk)
            if not header_match: continue
            dc_id = int(header_match.group(1))
            
            # 2. Carbon & Trend
            c_match = re.search(r"Carbon:\s+(\d+)", chunk)
            carbon = int(c_match.group(1)) if c_match else 9999
            
            c_trend_match = re.search(r"Carbon:.*\(Trend:\s+(.*?)\)", chunk)
            c_trend = c_trend_match.group(1) if c_trend_match else "Stable"
            
            # 3. Price & Trend
            p_match = re.search(r"Price:\s+\$?([\d\.]+)", chunk)
            price = float(p_match.group(1)) if p_match else 9999.0

            # NEW: Capture Price Trend
            p_trend_match = re.search(r"Price:.*\(Trend:\s+(.*?)\)", chunk)
            p_trend = p_trend_match.group(1) if p_trend_match else "Stable"
            
            # 4. Cooling / Temp
            t_match = re.search(r"Cooling:\s+([\d\.]+)C", chunk)
            temp = float(t_match.group(1)) if t_match else 25.0
            
            # 5. Capacity Constraints
            is_insufficient = "INSUFFICIENT" in chunk
            
            stats[dc_id] = {
                'carbon': carbon,
                'carbon_trend': c_trend,
                'price': price,
                'price_trend': p_trend, # <--- NEW FIELD
                'temp': temp,
                'insufficient': is_insufficient
            }
        except Exception as e:
            continue
            
    return stats

def extract_requirements(prompt):
    reqs = {}
    # ... (same as before) ...
    cpu_match = re.search(r"Needed:\s+([\d\.]+)\s+vCPU", prompt)
    gpu_match = re.search(r"Needed:\s+([\d\.]+)\s+GPU", prompt)
    reqs['cpu'] = float(cpu_match.group(1)) if cpu_match else 0
    reqs['gpu'] = float(gpu_match.group(1)) if gpu_match else 0
    
    dur_match = re.search(r"Duration:\s+~([\d\.]+)\s+hours", prompt)
    reqs['duration'] = float(dur_match.group(1)) if dur_match else 1.0
    return reqs

def generate_synthetic_reasoning(entry, stats):
    strategy = entry['strategy']
    action = entry['chosen_action']
    prompt = entry['prompt']
    
    if action not in stats:
        return f"**Conclusion:** Route to DC_{action}."

    chosen = stats[action]
    reqs = extract_requirements(prompt)
    reasoning = ""
    
    # 1. ANALYZE CONSTRAINTS
    insufficient_dcs = [k for k,v in stats.items() if v['insufficient']]
    valid_dcs = [k for k,v in stats.items() if not v['insufficient']]
    
    if action in insufficient_dcs:
        reasoning += f"**Constraint Alert:** DC_{action} is flagged insufficient, but heuristic forced selection. Overriding constraint.\n"
    elif insufficient_dcs:
        reasoning += f"**Capacity Check:** DCs {insufficient_dcs} are disqualified due to insufficient resources.\n"

    # 2. STRATEGY REASONING
    if "Carbon" in strategy:
        goal = "minimize carbon footprint"
        val = chosen['carbon']
        trend = chosen['carbon_trend']
        
        valid_stats = {k: stats[k] for k in valid_dcs} if valid_dcs else stats
        worst_dc = max(valid_stats, key=lambda x: valid_stats[x]['carbon'])
        diff = valid_stats[worst_dc]['carbon'] - val
        
        reasoning += f"**Objective:** {goal}.\n"
        
        # Carbon Trend Analysis
        if "Rising" in trend:
            reasoning += f"**Foresight:** Carbon is rising ({trend}), but DC_{action} remains the greenest current option.\n"
        elif "Falling" in trend:
            reasoning += f"**Foresight:** Excellent choice; carbon is falling ({trend}) for this {reqs['duration']}h job.\n"
            
        reasoning += f"**Observation:** DC_{action} offers the lowest intensity ({val} g/kWh), saving {diff} g/kWh vs DC_{worst_dc}.\n"

        if chosen['temp'] < 12:
            reasoning += f"**Efficiency:** Low external temp ({chosen['temp']}C) allows Free Cooling.\n"

    elif "Price" in strategy:
        goal = "minimize OpEx"
        val = chosen['price']
        p_trend = chosen['price_trend'] # <--- NEW
        
        valid_stats = {k: stats[k] for k in valid_dcs} if valid_dcs else stats
        worst_dc = max(valid_stats, key=lambda x: valid_stats[x]['price'])
        savings = valid_stats[worst_dc]['price'] - val
        
        reasoning += f"**Objective:** {goal}.\n"
        
        # Price Trend Analysis
        if "Rising" in p_trend:
             reasoning += f"**Foresight:** Although the price trend is {p_trend}, DC_{action} provides a significant buffer compared to competitors.\n"
        
        reasoning += (
            f"**Observation:** DC_{action} is the most cost-effective valid node (${val:.2f}). "
            f"Estimated savings: ${savings:.2f}.\n"
        )
        
        # Co-Benefit / Trade-off Analysis
        avg_carbon = sum(s['carbon'] for s in valid_stats.values()) / len(valid_stats)
        if chosen['carbon'] > avg_carbon * 1.2: # If 20% dirtier than average
            reasoning += f"**Trade-off:** This choice strictly prioritizes cost. Note that Carbon ({chosen['carbon']} g/kWh) is high in this region.\n"
        elif chosen['carbon'] < avg_carbon:
            reasoning += f"**Co-Benefit:** This choice also aligns with sustainability (Carbon is below average).\n"

    reasoning += f"**Conclusion:** Route to DC_{action}."
    return reasoning

def process_dataset(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    processed_data = []
    print(f"Augmenting {len(data)} samples with Constraint & Physics reasoning...")
    
    for entry in data:
        stats = parse_dc_stats(entry['prompt'])
        cot = generate_synthetic_reasoning(entry, stats)
        
        final_response = f"{cot}\n\nAction: {entry['chosen_action']}"
        
        processed_entry = {
            "instruction": entry['prompt'],
            "input": "", 
            "output": final_response,
            "strategy_label": entry['strategy'] 
        }
        processed_data.append(processed_entry)
        
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=4)
        
    print(f"Saved processed dataset to {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/expert_trajectories/heuristic_dataset_v2.json"
    OUTPUT_FILE = "data/expert_trajectories/train_dataset_cot_v2.json"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    process_dataset(INPUT_FILE, OUTPUT_FILE)