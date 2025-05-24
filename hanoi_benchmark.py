import time
import csv
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Tuple, Dict
from call_claude import call_claude

# Output directory for our charts and graphs
VISUALISATION_DIR = "visualisations"
os.makedirs(VISUALISATION_DIR, exist_ok=True)

# Simple token counter - good enough for benchmark purposes
def estimate_tokens(text: str) -> int:
    """Rough token count based on character length."""
    # Quick approximation - 4 chars per token is reasonable for English text
    return len(text) // 4

# Cost calculator for API usage
def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str = "claude-3-sonnet") -> float:
    """Calculate approximate cost based on token counts."""
    if model == "claude-3-sonnet":
        # Claude 3 Sonnet pricing (May 2023)
        input_price_per_1k = 0.003  # £0.003 per 1K input tokens
        output_price_per_1k = 0.015  # £0.015 per 1K output tokens
    elif model == "claude-3-opus":
        # Claude 3 Opus pricing (May 2023)
        input_price_per_1k = 0.015  # £0.015 per 1K input tokens
        output_price_per_1k = 0.075  # £0.075 per 1K output tokens
    else:
        # Default to Sonnet pricing
        input_price_per_1k = 0.003
        output_price_per_1k = 0.015
        
    input_cost = (prompt_tokens / 1000) * input_price_per_1k
    output_cost = (completion_tokens / 1000) * output_price_per_1k
    
    return input_cost + output_cost

# Test with 2-6 disks - beyond 6 gets unwieldy
DISK_RANGE = range(2, 7)
CSV_FILE = "hanoi_benchmark_results.csv"

### ----------------------------------------
### Prompt Templates
### ----------------------------------------

def generate_prompt(system: str, num_disks: int) -> str:
    """Create the appropriate prompt for each system type."""
    if system == "system1":
        # System 1: Fast & intuitive
        return f"Solve the Tower of Hanoi with {num_disks} disks. List only the moves step by step."
    elif system == "system1.5":
        # System 1.5: Step-by-step reasoning
        return f"Reason step by step and solve the Tower of Hanoi with {num_disks} disks. Explain each move and then provide it."
    elif system == "system2":
        # System 2: Strategic planning
        return (
            f"You are an expert problem solver. Plan carefully before acting.\n"
            f"First, explain your strategy in detail for solving Tower of Hanoi with {num_disks} disks.\n"
            f"Then, write all valid moves one per line, in correct order, from peg A to C using peg B.\n"
            f"Only list valid moves — no errors."
        )
    else:
        raise ValueError(f"Unknown system type: {system}")

### ----------------------------------------
### Solution Validator
### ----------------------------------------

def validate_moves(moves: List[str], num_disks: int) -> Tuple[bool, int]:
    """Check if a sequence of moves correctly solves the puzzle."""
    # Initial tower state
    pegs = {"A": list(range(num_disks, 0, -1)), "B": [], "C": []}
    illegal_moves = 0
    move_pattern = re.compile(r"Move disk (\d+) from (\w) to (\w)")

    # Process each move
    for move in moves:
        match = move_pattern.search(move)
        if not match:
            continue  # Skip lines that don't contain moves

        # Extract move details
        disk, src, tgt = int(match.group(1)), match.group(2), match.group(3)
        
        # Check if move is valid
        if not pegs[src] or pegs[src][-1] != disk:
            illegal_moves += 1
            continue
        if pegs[tgt] and pegs[tgt][-1] < disk:
            illegal_moves += 1
            continue
            
        # Apply the move
        pegs[src].pop()
        pegs[tgt].append(disk)

    # Check if puzzle is solved correctly
    is_complete = pegs["C"] == list(range(num_disks, 0, -1))
    return is_complete and illegal_moves == 0, illegal_moves

### ----------------------------------------
### Benchmark Runner
### ----------------------------------------

def run_benchmarks():
    """Run the full benchmark suite across all systems and disk counts."""
    # Define our metrics
    fieldnames = [
        "system", "num_disks", "runtime_sec", "num_moves", "illegal_moves", 
        "valid", "optimal_moves", "prompt_tokens", "completion_tokens", 
        "total_tokens", "estimated_cost", "tokens_per_correct_move"
    ]
    
    # Create results file
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Test each system
        for system in ["system1", "system1.5", "system2"]:
            print(f"Running benchmarks for {system}...")
            for num_disks in DISK_RANGE:
                print(f"  Testing with {num_disks} disks...")
                
                # Generate the appropriate prompt
                prompt = generate_prompt(system, num_disks)
                prompt_tokens = estimate_tokens(prompt)

                # Time the response
                start_time = time.time()
                response = call_claude(prompt)
                end_time = time.time()
                
                # Calculate token usage
                completion_tokens = estimate_tokens(response)
                total_tokens = prompt_tokens + completion_tokens
                
                # Estimate API cost
                cost = estimate_cost(prompt_tokens, completion_tokens)

                # Extract and validate moves
                moves = [line.strip() for line in response.splitlines() if "Move disk" in line]
                valid, illegal_moves = validate_moves(moves, num_disks)
                optimal = 2**num_disks - 1  # Optimal number of moves for Tower of Hanoi
                
                # Calculate efficiency metric
                correct_moves = len(moves) - illegal_moves
                tokens_per_correct_move = total_tokens / max(1, correct_moves)  # Avoid division by zero

                # Record results
                writer.writerow({
                    "system": system,
                    "num_disks": num_disks,
                    "runtime_sec": round(end_time - start_time, 2),
                    "num_moves": len(moves),
                    "illegal_moves": illegal_moves,
                    "valid": valid,
                    "optimal_moves": optimal,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost": round(cost, 6),
                    "tokens_per_correct_move": round(tokens_per_correct_move, 2)
                })

    print(f"Benchmark completed and saved to {CSV_FILE}")

### ----------------------------------------
### Data Visualisation
### ----------------------------------------

def plot_results():
    """Generate all charts and visualisations from benchmark data."""
    # Load the results
    df = pd.read_csv(CSV_FILE)
    sns.set(style="whitegrid")

    # Calculate move efficiency (actual vs optimal)
    df["move_efficiency"] = df["optimal_moves"] / df["num_moves"]
    df.loc[~df["valid"], "move_efficiency"] = 0  # Invalid solutions get 0 efficiency
    
    # Define the metrics we want to chart
    metrics = {
        "runtime_sec": "Runtime (seconds)",
        "num_moves": "Number of Moves",
        "illegal_moves": "Illegal Moves",
        "move_efficiency": "Move Efficiency (optimal/actual)",
        "prompt_tokens": "Prompt Tokens",
        "completion_tokens": "Completion Tokens",
        "total_tokens": "Total Tokens Used",
        "estimated_cost": "Estimated Cost (£)",
        "tokens_per_correct_move": "Tokens per Correct Move"
    }
    
    # Create individual charts for each metric
    for metric, title in metrics.items():
        plt.figure(figsize=(10, 6))
        for system in df["system"].unique():
            subset = df[df["system"] == system]
            plt.plot(subset["num_disks"], subset[metric], 
                     marker='o', linewidth=2, label=system)
        
        plt.title(f"{title} vs Number of Disks", fontsize=14)
        plt.xlabel("Number of Disks", fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALISATION_DIR, f"{metric}_plot.png"))
        plt.close()
    
    # Create a comprehensive dashboard
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("Tower of Hanoi LLM Benchmark: Comprehensive Analysis", fontsize=20, y=0.98)
    
    # Performance metrics (top row)
    ax1 = plt.subplot(3, 3, 1)
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        ax1.plot(subset["num_disks"], subset["runtime_sec"], marker='o', linewidth=2, label=system)
    ax1.set_title("Runtime (seconds)", fontsize=12)
    ax1.set_xlabel("Number of Disks", fontsize=10)
    ax1.grid(True)
    
    ax2 = plt.subplot(3, 3, 2)
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        ax2.plot(subset["num_disks"], subset["illegal_moves"], marker='o', linewidth=2, label=system)
    ax2.set_title("Illegal Moves", fontsize=12)
    ax2.set_xlabel("Number of Disks", fontsize=10)
    ax2.grid(True)
    
    ax3 = plt.subplot(3, 3, 3)
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        ax3.plot(subset["num_disks"], subset["move_efficiency"], marker='o', linewidth=2, label=system)
    ax3.set_title("Move Efficiency", fontsize=12)
    ax3.set_xlabel("Number of Disks", fontsize=10)
    ax3.grid(True)
    
    # Token usage metrics (middle row)
    ax4 = plt.subplot(3, 3, 4)
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        ax4.plot(subset["num_disks"], subset["prompt_tokens"], marker='o', linewidth=2, label=system)
    ax4.set_title("Prompt Tokens", fontsize=12)
    ax4.set_xlabel("Number of Disks", fontsize=10)
    ax4.grid(True)
    
    ax5 = plt.subplot(3, 3, 5)
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        ax5.plot(subset["num_disks"], subset["completion_tokens"], marker='o', linewidth=2, label=system)
    ax5.set_title("Completion Tokens", fontsize=12)
    ax5.set_xlabel("Number of Disks", fontsize=10)
    ax5.grid(True)
    
    ax6 = plt.subplot(3, 3, 6)
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        ax6.plot(subset["num_disks"], subset["total_tokens"], marker='o', linewidth=2, label=system)
    ax6.set_title("Total Tokens", fontsize=12)
    ax6.set_xlabel("Number of Disks", fontsize=10)
    ax6.grid(True)
    
    # Cost and efficiency metrics (bottom row)
    ax7 = plt.subplot(3, 3, 7)
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        ax7.plot(subset["num_disks"], subset["estimated_cost"], marker='o', linewidth=2, label=system)
    ax7.set_title("Estimated Cost (£)", fontsize=12)
    ax7.set_xlabel("Number of Disks", fontsize=10)
    ax7.grid(True)
    
    ax8 = plt.subplot(3, 3, 8)
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        ax8.plot(subset["num_disks"], subset["tokens_per_correct_move"], marker='o', linewidth=2, label=system)
    ax8.set_title("Tokens per Correct Move", fontsize=12)
    ax8.set_xlabel("Number of Disks", fontsize=10)
    ax8.grid(True)
    
    ax9 = plt.subplot(3, 3, 9)
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        valid_rate = subset["valid"].astype(int).values
        ax9.plot(subset["num_disks"], valid_rate, marker='o', linewidth=2, label=system)
    ax9.set_title("Solution Validity (1=Valid, 0=Invalid)", fontsize=12)
    ax9.set_xlabel("Number of Disks", fontsize=10)
    ax9.grid(True)
    
    # Add a single legend for the entire figure
    lines, labels = ax1.get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02), fontsize=12)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(VISUALISATION_DIR, "comprehensive_benchmark.png"), dpi=300)
    plt.close()
    
    # Create a cost-benefit analysis chart
    plt.figure(figsize=(12, 8))
    
    # Bubble sizes for each system
    size_map = {"system1": 100, "system1.5": 250, "system2": 400}
    
    # Plot each system
    for system in df["system"].unique():
        subset = df[df["system"] == system]
        plt.scatter(
            subset["total_tokens"],
            subset["move_efficiency"],
            s=[size_map[system]] * len(subset),
            alpha=0.7,
            label=system
        )
        
        # Add disk count labels
        for i, row in subset.iterrows():
            plt.annotate(
                f"{int(row['num_disks'])} disks",
                (row["total_tokens"], row["move_efficiency"]),
                xytext=(5, 5),
                textcoords="offset points"
            )
    
    plt.title("Cost-Benefit Analysis: Token Usage vs. Solution Quality", fontsize=16)
    plt.xlabel("Total Tokens (Cost Proxy)", fontsize=14)
    plt.ylabel("Move Efficiency (Quality Proxy)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Prompting Strategy", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALISATION_DIR, "cost_benefit_analysis.png"), dpi=300)
    plt.close()

    print(f"Charts and visualisations generated in '{VISUALISATION_DIR}' directory")

### ----------------------------------------
### Main Entry Point
### ----------------------------------------

if __name__ == "__main__":
    run_benchmarks()
    plot_results() 