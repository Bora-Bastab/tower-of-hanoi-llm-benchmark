import random
import time
import re

def call_claude(prompt: str, model: str = "claude-3-sonnet-20240229") -> str:
    """Simulates different Claude behaviours based on the prompt type and system."""
    # Figure out which system we're dealing with
    if "List only the moves" in prompt:
        system_type = "system1"
        # System 1: Basic prompting - fastest response
        base_time = 0.05
        # Add a touch of randomness but keep it snappy
        processing_time = base_time + random.uniform(0.01, 0.03) * len(prompt) / 100
    elif "Reason step by step" in prompt:
        system_type = "system1.5"
        # System 1.5: Reasoning - middling speed
        base_time = 0.1
        # Takes more time to think through steps
        processing_time = base_time + random.uniform(0.03, 0.06) * len(prompt) / 100
    elif "expert problem solver" in prompt:
        system_type = "system2"
        # System 2: Expert planning - most thoughtful, but slowest
        base_time = 0.2
        # Significant thinking time for planning
        processing_time = base_time + random.uniform(0.05, 0.08) * len(prompt) / 100
    else:
        system_type = "unknown"
        processing_time = 0.1
    
    # Throw in a bit of network lag
    network_latency = random.uniform(0.02, 0.05)
    
    # Total time to respond
    total_time = processing_time + network_latency
    
    # Pause to simulate realistic response time
    time.sleep(total_time)
    
    # Work out how many disks we're dealing with
    disks_match = re.search(r"(\d+) disks?", prompt)
    num_disks = int(disks_match.group(1)) if disks_match else 3
    
    # Generate response based on system type
    if system_type == "system1":
        return simulate_system1(num_disks)
    elif system_type == "system1.5":
        return simulate_system15(num_disks)
    elif system_type == "system2":
        return simulate_system2(num_disks)
    
    # Fallback for unrecognised prompts
    return "I'm not sure how to solve this problem."

def simulate_system1(num_disks: int) -> str:
    """Simulates System 1 behaviour: direct but possibly incorrect."""
    moves = []
    error_probability = 0.05 * num_disks  # More disks = more mistakes
    
    def generate_system1_moves(n, source, auxiliary, target):
        if n == 1:
            moves.append(f"Move disk 1 from {source} to {target}")
            return
            
        generate_system1_moves(n-1, source, target, auxiliary)
        moves.append(f"Move disk {n} from {source} to {target}")
        
        # Might slip up on complex problems
        if n > 2 and random.random() < error_probability:
            # Random dodgy move
            wrong_disk = random.randint(1, n-1)
            wrong_source = auxiliary if random.random() < 0.7 else target
            wrong_target = source if wrong_source == auxiliary else auxiliary
            moves.append(f"Move disk {wrong_disk} from {wrong_source} to {wrong_target}")
        
        generate_system1_moves(n-1, auxiliary, source, target)
    
    # For trickier puzzles, sometimes give up before finishing
    if num_disks > 5 and random.random() < 0.3:
        generate_system1_moves(num_disks-1, "A", "B", "C")
    else:
        generate_system1_moves(num_disks, "A", "B", "C")
    
    return "\n".join(moves)

def simulate_system15(num_disks: int) -> str:
    """Simulates System 1.5 behaviour: reasoning reduces errors."""
    moves_with_reasons = []
    error_probability = 0.02 * num_disks  # Lower error rate than System 1
    
    preamble = [
        f"I'll solve the Tower of Hanoi with {num_disks} disks by reasoning through each step.",
        f"The goal is to move all {num_disks} disks from peg A to peg C using peg B as a helper.",
        "I need to follow two rules: never place a larger disk on a smaller one, and move only one disk at a time.",
        "",
        "My strategy is to use recursion conceptually:"
    ]
    
    if num_disks <= 3:
        preamble.append(f"1. Move {num_disks-1} disks from A to B")
        preamble.append(f"2. Move the largest disk ({num_disks}) from A to C")
        preamble.append(f"3. Move {num_disks-1} disks from B to C")
    else:
        preamble.append("1. Break down the problem into simpler sub-problems")
        preamble.append("2. Solve each sub-problem using the same approach")
        preamble.append("3. Build up the complete solution from the sub-solutions")
    
    preamble.append("")
    preamble.append("Here's the step-by-step solution:")
    preamble.append("")
    
    move_count = 0
    def generate_system15_moves(n, source, auxiliary, target):
        nonlocal move_count
        if n == 1:
            move_count += 1
            moves_with_reasons.append(f"Move disk 1 from {source} to {target} (This is the smallest disk, so it can be moved directly)")
            return
            
        generate_system15_moves(n-1, source, target, auxiliary)
        move_count += 1
        moves_with_reasons.append(f"Move disk {n} from {source} to {target} (Now that smaller disks are out of the way, we can move this larger disk)")
        
        # Even with reasoning, might slip up occasionally on complex problems
        if n > 3 and random.random() < error_probability:
            # But the mistakes are more plausible
            move_count += 1
            wrong_disk = n-2  # A more believable error
            moves_with_reasons.append(f"Move disk {wrong_disk} from {auxiliary} to {source} (We need to begin rebuilding the tower)")
        
        generate_system15_moves(n-1, auxiliary, source, target)
    
    generate_system15_moves(num_disks, "A", "B", "C")
    
    content = preamble + moves_with_reasons
    return "\n".join(content)

def simulate_system2(num_disks: int) -> str:
    """Simulates System 2 behaviour: expert, strategic, and correct."""
    # Perfect algorithmic solution with thorough explanation
    explanation = [
        f"# Strategy for Solving Tower of Hanoi with {num_disks} Disks",
        "",
        "The Tower of Hanoi problem can be solved efficiently using a recursive approach:",
        "",
        "1. For n disks, the optimal solution always takes (2^n - 1) moves",
        f"2. For {num_disks} disks, we'll need exactly {2**num_disks - 1} moves",
        "3. The recursive strategy is:",
        f"   a. Move {num_disks-1} disks from A to B (recursively)",
        f"   b. Move the largest disk ({num_disks}) from A to C",
        f"   c. Move {num_disks-1} disks from B to C (recursively)",
        "",
        "This approach guarantees an optimal solution with no illegal moves.",
        "",
        "# Step-by-Step Solution",
        ""
    ]
    
    moves = []
    def generate_optimal_moves(n, source, auxiliary, target):
        if n == 1:
            moves.append(f"Move disk 1 from {source} to {target}")
            return
        generate_optimal_moves(n-1, source, target, auxiliary)
        moves.append(f"Move disk {n} from {source} to {target}")
        generate_optimal_moves(n-1, auxiliary, source, target)
    
    generate_optimal_moves(num_disks, "A", "B", "C")
    
    return "\n".join(explanation + moves) 