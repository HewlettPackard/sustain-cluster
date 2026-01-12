import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

# Set publication-quality style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'

RESULT_DIR = "results/evaluation"
OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_agent_variant(filename, agent_col):
    """
    Parses the filename or Agent column to determine the specific variant.
    Returns (Group, Label) for plotting.
    """
    name = str(agent_col).lower()
    fname = str(filename).lower()
    
    # 1. HEURISTICS
    if "random" in name or "random" in fname: return "Baselines", "Random"
    if "carbon" in name or "carbon" in fname: return "Baselines", "Carbon-Greedy"
    if "price" in name or "price" in fname:  return "Baselines", "Cost-Greedy"

    # 2. LLM VARIANTS
    # Determine Model Base (Zero-Shot vs Fine-Tuned)
    is_ft = "ecodistill" in name or "ecodistill" in fname or "ft" in fname
    base_label = "EcoDistill (FT)" if is_ft else "Llama-3 (ZS)"
    
    # Determine Capabilities (RAG / History)
    # We check the Agent string logic from evaluate_agents.py usually constructed like LLM-RAG-Hist
    has_rag = "rag" in name or "rag" in fname
    has_hist = "hist" in name or "hist" in fname

    suffix = []
    if has_rag: suffix.append("RAG")
    if has_hist: suffix.append("Hist")
    
    variant_desc = " + ".join(suffix) if suffix else "Base"
    full_label = f"{base_label} + {variant_desc}" if variant_desc != "Base" else base_label
    
    return base_label, full_label

def load_results():
    data = []
    files = glob.glob(os.path.join(RESULT_DIR, "*.csv"))
    
    if not files:
        print("❌ No result CSVs found in results/evaluation/")
        return pd.DataFrame()

    print(f"Found {len(files)} result files.")

    for f in files:
        try:
            df = pd.read_csv(f)
            # Add metadata columns
            filename = os.path.basename(f)
            
            # Apply detection logic to the first row's Agent name
            raw_agent_name = df["Agent"].iloc[0] if "Agent" in df.columns else filename
            group, label = detect_agent_variant(filename, raw_agent_name)
            
            df["Agent_Group"] = group
            df["Agent_Label"] = label
            data.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    return pd.concat(data, ignore_index=True)

def plot_pareto_frontier(df):
    """
    Carbon vs Cost Trade-off with improved markers and colors.
    """
    plt.figure(figsize=(12, 8))
    
    # Aggregate data
    summary = df.groupby(["Agent_Group", "Agent_Label"]).agg({
        "Carbon_kg": ["mean"], 
        "Cost_$": ["mean"]
    }).reset_index()
    summary.columns = ['Group', 'Label', 'Carbon', 'Cost']

    # Define markers for groups
    markers = {"Baselines": "X", "Llama-3 (ZS)": "o", "EcoDistill (FT)": "D"}
    
    sns.scatterplot(
        data=summary,
        x="Cost", 
        y="Carbon", 
        hue="Label",
        style="Group",
        markers=markers,
        s=200, # Marker size
        alpha=0.9,
        palette="viridis"
    )

    # Add annotations (avoid overlap logic simplified)
    for i, row in summary.iterrows():
        plt.text(
            row['Cost'], 
            row['Carbon'] + (summary['Carbon'].max() * 0.015), 
            row['Label'], 
            fontsize=9, 
            ha='center', 
            weight='bold' if "EcoDistill" in row['Label'] else 'normal'
        )

    plt.title("Pareto Frontier: Operational Cost vs. Carbon Emissions", fontsize=16)
    plt.xlabel("Total Operational Cost ($)", fontsize=12)
    plt.ylabel("Total Carbon Emissions (kgCO₂)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_tradeoff_pareto.pdf")
    plt.savefig(f"{OUTPUT_DIR}/1_tradeoff_pareto.png")
    print("✅ Saved Pareto Plot")

def plot_ablation_matrix(df):
    """
    Grouped Bar Chart specifically for the Ablation Study.
    Filters out Random/Heuristics to focus on LLM internals.
    """
    # Filter for LLMs only
    llm_df = df[df["Agent_Group"].isin(["Llama-3 (ZS)", "EcoDistill (FT)"])].copy()
    
    if llm_df.empty: return

    # Normalize metrics relative to Zero-Shot Base (if exists, else random)
    # Finding a baseline
    baseline_val = llm_df[llm_df["Agent_Label"] == "Llama-3 (ZS)"].agg({"Carbon_kg": "mean", "SLA_Violations": "mean"})
    if baseline_val.isna().any():
        baseline_val = llm_df.agg({"Carbon_kg": "mean", "SLA_Violations": "mean"})

    # Normalize
    llm_df["Norm_Carbon"] = llm_df["Carbon_kg"] / baseline_val["Carbon_kg"]
    llm_df["Norm_SLA"] = llm_df["SLA_Violations"] / (baseline_val["SLA_Violations"] + 1e-6) # Avoid div0

    # Reshape for plotting
    melted = llm_df.melt(
        id_vars=["Agent_Group", "Agent_Label"], 
        value_vars=["Norm_Carbon", "Norm_SLA"],
        var_name="Metric",
        value_name="Normalized Score"
    )

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=melted,
        x="Agent_Label",
        y="Normalized Score",
        hue="Metric",
        palette="magma"
    )
    
    plt.axhline(1.0, color='black', linestyle='--', alpha=0.5, label="Baseline")
    plt.title("Ablation Study: Contribution of Components (Lower is Better)", fontsize=16)
    plt.ylabel("Normalized Ratio (vs Zero-Shot)", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=15, ha="right")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_ablation_study.pdf")
    plt.savefig(f"{OUTPUT_DIR}/2_ablation_study.png")
    print("✅ Saved Ablation Plot")

def plot_sustainability_impact(df):
    """
    Simple Bar chart comparing Proposed vs Baselines on Carbon only.
    """
    # Select key agents for the main story
    target_agents = ["Random", "Carbon-Greedy", "Cost-Greedy", "Llama-3 (ZS)", "EcoDistill (FT) + RAG + Hist"]
    
    # Fuzzy match to find them in the dataset
    mask = df["Agent_Label"].apply(lambda x: any(t in x for t in target_agents))
    subset = df[mask].copy()
    
    if subset.empty: return

    plt.figure(figsize=(10, 6))
    
    # Aggregate mean + std
    agg = subset.groupby("Agent_Label")["Carbon_kg"].agg(["mean", "sem"]).reset_index()
    agg = agg.sort_values("mean", ascending=False) # Worst to best

    sns.barplot(
        data=agg,
        x="Agent_Label",
        y="mean",
        palette="Greens_r", # Dark green = low carbon
        edgecolor=".2"
    )
    
    plt.errorbar(
        x=range(len(agg)), 
        y=agg["mean"], 
        yerr=agg["sem"], 
        fmt='none', 
        c='black', 
        capsize=5
    )

    plt.title("Total Carbon Footprint Comparison", fontsize=16)
    plt.ylabel("Total Emissions (kgCO₂)", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_sustainability_impact.pdf")
    plt.savefig(f"{OUTPUT_DIR}/3_sustainability_impact.png")
    print("✅ Saved Sustainability Plot")

if __name__ == "__main__":
    df = load_results()
    if not df.empty:
        plot_pareto_frontier(df)
        plot_ablation_matrix(df)
        plot_sustainability_impact(df)
        print(f"\n🎉 All plots saved to {OUTPUT_DIR}")