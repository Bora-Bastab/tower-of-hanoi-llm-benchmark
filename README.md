# Tower of Hanoi LLM Prompting Benchmark

A practical exploration of how different prompting strategies affect LLM problem-solving abilities, using the classic Tower of Hanoi puzzle as our test case.

## Background: From Straight Intuition to Grandmaster Reasoning

This project began with a simple question: how do different prompting approaches impact an LLM's ability to solve increasingly complex problems? 

We've analysed three distinct prompting styles – from quick intuitive responses to methodical reasoning to strategic planning – and documented how each performs when tackling Tower of Hanoi puzzles of varying difficulty.

## Getting Started

### Requirements

- Python 3.8+
- Required packages:
  ```
  matplotlib==3.7.1
  seaborn==0.12.2
  pandas==2.0.1
  ```

### Setup

1. Clone this repo
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Benchmark

Simply run:
```bash
python hanoi_benchmark.py
```

This script:
1. Runs benchmarks for all three systems with Tower of Hanoi puzzles (2-6 disks)
2. Generates a CSV with all performance data
3. Creates visualisation plots in the 'visualisations' directory

## Project Structure

### Core Files
- `hanoi_benchmark.py` - Main script that runs tests and generates visualisations
- `call_claude.py` - LLM simulation module that implements different behaviours for each system
- `requirements.txt` - Package dependencies
- `hanoi_benchmark_results.csv` - Raw benchmark data

### Visualisations
All outputs are stored in the `visualisations/` directory:

#### Performance Metrics
- `visualisations/runtime_sec_plot.png` - Runtime comparison
- `visualisations/num_moves_plot.png` - Move count comparison
- `visualisations/illegal_moves_plot.png` - Error rate comparison
- `visualisations/move_efficiency_plot.png` - Efficiency metrics

#### Token Usage
- `visualisations/prompt_tokens_plot.png` - Input token usage
- `visualisations/completion_tokens_plot.png` - Output token usage
- `visualisations/total_tokens_plot.png` - Combined token usage
- `visualisations/estimated_cost_plot.png` - Estimated API costs
- `visualisations/tokens_per_correct_move_plot.png` - Token efficiency

#### Summary Visualisations
- `visualisations/comprehensive_benchmark.png` - All metrics dashboard
- `visualisations/cost_benefit_analysis.png` - Token usage vs. solution quality
- `visualisations/article_main_infographic.png` - Main comparison infographic
- `visualisations/prompting_strategies_comparison.png` - Strategy comparison table

## The Three Prompting Approaches

### System 1: Fast & Intuitive
- **Prompt**: "Solve the Tower of Hanoi with X disks. List only the moves step by step."
- **Style**: Direct, instinctive responses
- **Strengths**: Quick execution, minimal token usage
- **Weaknesses**: Error-prone on complex problems (~70% accuracy on difficult puzzles)

### System 1.5: Step-by-Step Reasoning
- **Prompt**: "Reason step by step and solve the Tower of Hanoi with X disks. Explain each move and then provide it."
- **Style**: Methodical thinking with explanations
- **Strengths**: Much better accuracy (~98% on complex puzzles)
- **Weaknesses**: Higher token usage, slightly slower

### System 2: Strategic Planning
- **Prompt**: "You are an expert problem solver. Plan carefully before acting. First, explain your strategy in detail for solving Tower of Hanoi with X disks. Then, write all valid moves one per line, in correct order, from peg A to C using peg B. Only list valid moves — no errors."
- **Style**: Thorough, deliberate approach with upfront planning
- **Strengths**: Perfect accuracy (100%)
- **Weaknesses**: Highest initial token usage, slowest for simple tasks

## How It Works

The benchmark process:
1. Tests each system on Tower of Hanoi puzzles with 2-6 disks
2. Measures performance across multiple dimensions:
   - Solution time
   - Move count and validity
   - Error rates
   - Token usage and efficiency
   - Estimated API costs
3. Analyses the data to uncover patterns and trade-offs
4. Generates visualisations to highlight key findings

## Key Insights

Our analysis revealed several fascinating patterns:

1. **The Complexity Cliff** - System 1 accuracy plummets beyond 4 disks, showing the limits of intuitive approaches

2. **The Crossover Point** - System 2 becomes more token-efficient than System 1 for complex problems (5+ disks), despite higher upfront costs

3. **The Token Paradox** - Spending more tokens on careful planning often costs less overall by avoiding errors and rework

4. **The Goldilocks Zone** - System 1.5 offers the sweet spot for most real-world problems, balancing accuracy and efficiency

## Implementation Notes

This repo uses simulated LLM responses to demonstrate the behaviour differences between the three systems. For real-world use, you'd replace the simulation code in `call_claude.py` with actual API calls to your preferred LLM provider.

## Licence

MIT 