from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np
from .config import RevisionConfig

@dataclass
class EnvironmentState:
    """Represents the state of the manufacturing environment at a point in time"""
    time: int  # Current time period
    revision_state: int  # Current revision state (0-8)
    inventory: Dict[int, float]  # Inventory levels by process
    incoming_orders: Dict[int, Dict[int, float]]  # Orders by process and arrival time
    total_produced: int  # Total units produced so far
    costs: Dict[str, float] = field(default_factory=lambda: {})
    
    def get_current_process(self) -> int:
        """Get current active manufacturing process"""
        return self.revision_state // 2
    
    def get_total_incoming(self) -> float:
        """Calculate total incoming inventory across all processes"""
        return sum(sum(orders.values()) for orders in self.incoming_orders.values())
    
    def get_transition_proba(self, config: RevisionConfig):
            if self.revision_state == config.num_states - 1:
                return 0
            current = self.revision_state // 2
            prob = 0.2/(current + 1) if self.revision_state % 2 == 0 else 0.4
            return prob

    def validate(self, config: RevisionConfig) -> bool:
        """Validate state consistency against configuration"""
        return (
            0 <= self.time < config.planning_horizon and
            0 <= self.revision_state < config.num_states and
            all(p in self.inventory for p in range(config.num_processes)) and
            all(p in self.incoming_orders for p in range(config.num_processes))
        )
    
    def next_state(self, orders: Dict[int, float], config: RevisionConfig, 
            advance: bool = None, demand_active: bool = True) -> Tuple['EnvironmentState', Dict[str, float]]:
        """
        Compute the next state given orders and optional demand/advance parameters.
        """
        remaining_units = config.total_target - self.total_produced

        # Use Poisson distribution for integer demand
        demand = np.random.poisson(lam=config.demand) if (remaining_units > 0 and demand_active) else 0

        # Force all remaining demand if in the final week
        remaining_time = config.planning_horizon - self.time - 1  # Subtract 1 to account for current time
        if remaining_time == 0:
            demand = remaining_units
                    
        # Determine state transition
        if advance is None:
            prob = self.get_transition_proba(config=config)
            advance = np.random.random() < prob
        next_revision = self.revision_state + 1 if advance else self.revision_state
        
        # Process incoming orders
        new_incoming = {p: orders_dict.copy() for p, orders_dict in self.incoming_orders.items()}
        for p, qty in orders.items():
            if qty > 0:
                arrival = self.time + config.lead_times[p]
                if arrival < config.planning_horizon:
                    new_incoming[p][arrival] = new_incoming[p].get(arrival, 0) + qty

        # Set outdated order quantities to zero
        for p in new_incoming.keys():
            for arrival_time in new_incoming[p].keys():
                if new_incoming[p][arrival_time] > 0 and p < self.get_current_process():
                    new_incoming[p][arrival_time] = 0
                    
        # Update inventory with arrivals
        new_inventory = self.inventory.copy()
        for p in range(config.num_processes):
            new_inventory[p] += new_incoming[p].pop(self.time, 0)
        
        # Handle demand fulfillment
        current_process = next_revision // 2
        fulfilled = 0
        remaining = demand
        
        # Clear obsolete inventory
        for p in range(config.num_processes):
            if p < current_process:
                new_inventory[p] = 0
            else:
                break
        
        # Fulfill demand from oldest valid process first
        for p in range(current_process, config.num_processes):
            use = min(new_inventory[p], remaining)
            new_inventory[p] -= use
            fulfilled += use
            remaining -= use
        
        # All demand is fulfilled after backorder costs are paid
        fulfilled = demand
        
        # Calculate all costs
        costs = {
            "setup": sum(1 for qty in orders.values() if qty > 0) * config.setup_cost,
            "unit": sum(config.unit_cost * qty for qty in orders.values()),
            "carbon": sum(config.carbon_costs[p] * qty for p, qty in orders.items()),
            "holding": sum(max(0, qty) * config.holding_cost for qty in new_inventory.values()),
            "backorder": remaining * config.backorder_cost
        }
        costs["total"] = sum(costs.values())
        
        return (EnvironmentState(
            time=self.time + 1,
            revision_state=next_revision,
            inventory=new_inventory,
            incoming_orders=new_incoming,
            total_produced=self.total_produced + fulfilled,
            costs=costs
        ), costs)