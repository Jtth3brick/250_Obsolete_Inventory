from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
import logging
import numpy as np

from .environment import EnvironmentState
from .config import RevisionConfig
from .policies import Policy

@dataclass(frozen=True)
class SimulationResults:
    """Results from a single simulation run"""
    states: List[EnvironmentState]
    production_schedule: Dict[int, Dict[int, float]] 
    info: Dict[str, Any]  # Additional metrics/info
    config: RevisionConfig  # Reference to configuration

    @property
    def total_cost(self) -> float:
        """Total cost across all states"""
        return sum(state.costs.get("total", 0) for state in self.states)

    @property 
    def carbon_emissions(self) -> float:
        """Total carbon emissions across all states"""
        return sum(state.costs.get("carbon", 0) for state in self.states)

    @property
    def service_level(self) -> float:
        """Calculate service level as ratio of fulfilled demand"""
        if not self.states:
            return 0.0
            
        total_demand = self.config.total_target
        final_produced = self.states[-1].total_produced
        
        # Avoid division by zero
        if total_demand == 0:
            return 1.0
            
        return final_produced / total_demand

    @property
    def summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics"""
        if not self.states:
            return {
                "total_cost": 0.0,
                "carbon_emissions": 0.0,
                "service_level": 0.0,
                "holding_costs": 0.0,
                "backorder_costs": 0.0,
                "setup_costs": 0.0,
                "unit_costs": 0.0,
                "final_inventory": {},
                "cpu_time": self.info.get("cpu_time", 0)
            }
            
        return {
            "total_cost": self.total_cost,
            "carbon_emissions": self.carbon_emissions,
            "service_level": self.service_level,
            "holding_costs": sum(s.costs.get("holding", 0) for s in self.states),
            "backorder_costs": sum(s.costs.get("backorder", 0) for s in self.states),
            "setup_costs": sum(s.costs.get("setup", 0) for s in self.states),
            "unit_costs": sum(s.costs.get("unit", 0) for s in self.states),
            "final_inventory": self.states[-1].inventory,
            "cpu_time": self.info.get("cpu_time", 0)
        }
class RevisionSimulator:
    """Handles simulation experiments for revision policies"""
    
    def __init__(self, config: RevisionConfig, log_dir: str = "simulation_logs"):
        """
        Initialize simulator.
        
        Args:
            config: Environment configuration
            log_dir: Directory for simulation logs
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self._setup_logging()
        
    def run_episode(self, policy: Policy, seed: Optional[int] = None) -> SimulationResults:
        """
        Run a single simulation episode.
        
        Args:
            policy: Policy to evaluate
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize tracking
        state = self._initial_state()
        states = [state]
        production_schedule = {}  # {time: {process: quantity}}
        
        # Track CPU time
        start_time = time.process_time()
        
        # Run episode
        # Changed condition to check for total_produced
        while (state.time < self.config.planning_horizon and 
            state.total_produced < self.config.total_target):
            # Get action from policy
            action = policy.get_action(state)
            
            # Track production
            production_schedule[state.time] = action.copy()
            
            # Step environment
            next_state, costs = state.next_state(action, self.config, demand_active=state.time >= min(self.config.lead_times))
            
            # Update policy (if it implements learning)
            policy.update(state, action, next_state, costs)
            
            # Track results
            states.append(next_state)
            state = next_state
            
            # Add early stopping log
            if state.total_produced >= self.config.total_target:
                self.logger.info(f"Target reached at time {state.time}")
        
        # Calculate total CPU time
        cpu_time = time.process_time() - start_time
        
        # Compile results
        results = SimulationResults(
            states=states,
            production_schedule=production_schedule,
            info={
                "seed": seed,
                "cpu_time": cpu_time,
                "stopped_early": state.time < self.config.planning_horizon
            },
            config=self.config
        )
        
        return results
        
    def evaluate_policy(
        self,
        policy: Policy,
        num_episodes: int = 100,
        seeds: Optional[List[int]] = None
    ) -> List[SimulationResults]:
        """
        Evaluate a policy over multiple episodes.
        
        Args:
            policy: Policy to evaluate
            num_episodes: Number of episodes to run
            seeds: Random seeds for reproducibility
            
        Returns:
            Results from all episodes
        """
        if seeds is not None and len(seeds) != num_episodes:
            raise ValueError("Must provide seed for each episode")
            
        results = []
        for episode in range(num_episodes):
            self.logger.info(f"Starting episode {episode + 1}/{num_episodes}")
            
            seed = seeds[episode] if seeds else None
            episode_results = self.run_episode(policy, seed)
            results.append(episode_results)
            
        self._log_results(results, policy.__class__.__name__)
        return results

    def _initial_state(self) -> EnvironmentState:
        """Create initial environment state"""
        return EnvironmentState(
            time=0,
            revision_state=0,
            inventory={p: 0.0 for p in range(self.config.num_processes)},
            incoming_orders={p: {} for p in range(self.config.num_processes)},
            total_produced=0
        )

    def _log_results(self, results: List[SimulationResults], policy_name: str):
        """Log statistical analysis of results"""
        stats = {
            "cost": [r.total_cost for r in results],
            "emissions": [r.carbon_emissions for r in results],
            "service": [r.service_level for r in results]
        }
        
        self.logger.info(f"\nResults for {policy_name}:")
        self.logger.info(f"Episodes: {len(results)}")
        self.logger.info(f"Cost: ${np.mean(stats['cost']):,.2f} (±${np.std(stats['cost']):,.2f})")
        self.logger.info(f"Emissions: {np.mean(stats['emissions']):,.2f} (±{np.std(stats['emissions']):,.2f})")
        self.logger.info(f"Service Level: {np.mean(stats['service'])*100:.1f}% (±{np.std(stats['service'])*100:.1f}%)")

    def _setup_logging(self):
        """Configure logging"""
        self.logger = logging.getLogger("RevisionSimulator")
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / "simulation.log")
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.handlers = []
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)