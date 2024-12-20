# config.py
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

@dataclass
class RevisionConfig:
    demand: int = 1 # items per week
    planning_horizon: int = 120  # weeks
    total_target: int = 80  # vehicles
    setup_cost: float = 1000  # per batch
    unit_cost: float = 100  # per unit
    holding_cost: float = 50  # per unit per week
    backorder_cost: float = 5000  # per unit per week
    lead_times: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1])
    
    def __post_init__(self):
        # Derived attributes
        self.num_processes = len(self.lead_times)
        self.num_states = 2 * self.num_processes - 1
        self.carbon_costs = [self.unit_cost * (1.5 ** i) for i in range(self.num_processes)]
        
        # Validation
        if self.planning_horizon <= 0:
            raise ValueError("Planning horizon must be positive")
        if self.total_target <= 0:
            raise ValueError("Production target must be positive")
        if any(lt <= 0 for lt in self.lead_times):
            raise ValueError("Lead times must be positive")
        if self.holding_cost < 0:
            raise ValueError("Holding cost cannot be negative")
        if self.backorder_cost < 0:
            raise ValueError("Backorder cost cannot be negative")

@dataclass 
class SimulationConfig:
    num_replications: int = 15
    random_seed: int = 42
    log_dir: str = "simulation_logs"
    
    def __post_init__(self):
        np.random.seed(self.random_seed)
        if self.num_replications <= 0:
            raise ValueError("Number of replications must be positive")