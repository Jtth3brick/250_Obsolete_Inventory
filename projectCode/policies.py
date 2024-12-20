import copy
from itertools import product
import gurobipy as gp
from gurobipy import GRB
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
from .environment import EnvironmentState
from .config import RevisionConfig

class Policy(ABC):
    def __init__(self, config: RevisionConfig):
        self.config = config

    @abstractmethod
    def get_action(self, state: EnvironmentState) -> Dict[int, float]:
        """Return order quantities by process given current state"""
        pass
        
    def update(self, state: EnvironmentState, action: Dict[int, float], 
               next_state: EnvironmentState, costs: Dict[str, float]):
        """Optional learning update after each step. Default no-op."""
        pass

class SimpleFeasiblePolicy(Policy):
    """Simply orders most expensive item from start"""
    def get_action(self, state: EnvironmentState) -> Dict[int, float]:
        orders = {p: 0.0 for p in range(self.config.num_processes)}
        
        if state.time == 0:
            final_process = self.config.num_processes - 1
            orders[final_process] = self.config.total_target
            
        return orders

class SimpleSSPolicy(Policy):
    "Purely reactive, maintains (s, S) on all revisions"
    def __init__(self, config: RevisionConfig, reorder_point: float = 2, order_up_to: float = 5):
        super().__init__(config)
        self.reorder_point = reorder_point
        self.order_up_to = order_up_to
        
    def get_action(self, state: EnvironmentState) -> Dict[int, float]:
        orders = {p: 0.0 for p in range(self.config.num_processes)}
        current_process = state.get_current_process()
        
        # Calculate remaining units needed
        remaining_units = (
            self.config.total_target - 
            state.total_produced - 
            sum(state.inventory.get(i, 0) for i in range(current_process, self.config.num_processes)) -
            state.get_total_incoming()
        )        

        # Order for current and all future processes if below reorder point
        for p in range(current_process, self.config.num_processes):
            if state.inventory.get(p, 0) <= self.reorder_point:
                target_order = min(self.order_up_to - state.inventory[p], remaining_units)
                orders[p] = min(max(0, target_order), self.order_up_to)
                remaining_units -= target_order
            
        return orders

class MyopicPolicy(Policy):
    def __init__(self, config: RevisionConfig):
        super().__init__(config)

        
    def calculate_revision_probas(self, state: EnvironmentState, periods: int) -> List[float]:
        """
        Calculate probability weights for each future period based on bernoulli trials
        """
        current_process = state.get_current_process()
        forced_next_state = state.next_state({}, self.config, advance=True)[0]
        in_warning = state.get_current_process() != forced_next_state.get_current_process()
        
        # Get transition probability
        p = state.get_transition_proba(config=self.config)
            
        # Calculate probability of still being in same revision for each period
        cumulative_probas = [(1 - p) ** t for t in range(periods)]
        
        # If not in warning state, also consider probability of next transition
        if not in_warning: # will never enter in last state case
            p_next = forced_next_state.get_transition_proba(self.config)
            for t in range(1, periods):
                # Probability of transitioning at time t and staying there
                transition_proba = p * (1 - p) ** (t-1) * (1 - p_next) ** (periods - t)
                cumulative_probas[t] += transition_proba

        return cumulative_probas
        
    def get_action(self, state: EnvironmentState) -> Dict[int, float]:
        """
        Solve optimization problem considering probabilistic future costs
        """
        orders = {p: 0.0 for p in range(self.config.num_processes)}
        current_process = state.get_current_process()
        
        # Skip if we've met the target
        if state.total_produced >= self.config.total_target:
            return orders
            
        remaining_horizon = self.config.planning_horizon - state.time
        remaining_target = self.config.total_target - state.total_produced
        
        # Current inventory position
        current_inventory = sum(state.inventory.values())
        current_incoming = state.get_total_incoming()
        total_position = current_inventory + current_incoming
        
        # If we get here, we should consider ordering
        model = gp.Model("MyopicModel")
        model.setParam('OutputFlag', 0)
        
        # Decision variables - only for current process
        O = model.addVar(vtype=GRB.INTEGER, name="O")
        Y = model.addVar(vtype=GRB.BINARY, name="setup")
        B = model.addVars(remaining_horizon, name="backorder")
        
        # Get probability weights
        weights = self.calculate_revision_probas(state, remaining_horizon)
        
        # Expected demand during this revision
        expected_revision_demand = min(
            self.config.demand * sum(weights),  # Expected duration * demand rate
            remaining_target - total_position    # Can't exceed remaining need
        )
        
        # Objective function
        model.setObjective(
            # Immediate costs
            self.config.setup_cost * Y +      # Setup costs
            self.config.unit_cost * O +       # Unit costs
            self.config.carbon_costs[current_process] * O +   # Carbon costs
            # Future costs (holding vs backorder tradeoff)
            gp.quicksum(
                weights[t] * (
                    self.config.holding_cost * (
                        total_position + O -
                        min(t * self.config.demand, remaining_target)
                    ) +
                    self.config.backorder_cost * B[t]  # Use auxiliary variable
                )
                for t in range(remaining_horizon)
            ),
            GRB.MINIMIZE
        )

        # Add constraints for the auxiliary backorder variables
        for t in range(remaining_horizon):
            model.addConstr(
                B[t] >= weights[t] * (
                    min(t * self.config.demand, remaining_target) -
                    (total_position + O)
                )
            )
            model.addConstr(B[t] >= 0)

        # Modify the big-M constraint to be less restrictive
        M = remaining_target - total_position

        model.addConstr(O <= M * Y)
        
        # Don't order more than remaining target
        model.addConstr(O <= max(0, remaining_target - total_position))
        
        try:
            model.optimize()
            
            if model.status == GRB.OPTIMAL:
                orders[current_process] = float(O.x) 
            else:
                print(f"Model status: {model.status}")
                
        except Exception as e:
            print(f"Optimization error: {e}")
            
        return orders

class OptimalPolicy(Policy):
    """
    Policy that considers integer quantiles of expected demand across all processes.
    """
    def __init__(self, config: RevisionConfig):
        super().__init__(config)
        self.cache = {}
    
    def _get_state_key(self, state: EnvironmentState) -> tuple:
        """Create hashable key for state caching"""
        return (
            state.time,
            state.total_produced,
            tuple((k, state.inventory[k]) for k in sorted(state.inventory)),
            tuple((k, tuple((t, state.incoming_orders[k][t]) for t in sorted(state.incoming_orders[k]) if state.incoming_orders[k][t])) for k in sorted(state.incoming_orders)),
            state.revision_state
        )
    
    def get_action(self, state: EnvironmentState) -> Dict[int, float]:
        action, _ = self.get_action_and_cost(state)
        return action
        
    def get_action_and_cost(self, state: EnvironmentState, quit_cost: float = None) -> Dict[int, float]:
        """Determine action using demand-based quantities"""
        if self._get_state_key(state) in self.cache:
            print("cache hit")
            return self.cache[self._get_state_key(state)]

        orders = {p: 0.0 for p in range(self.config.num_processes)}
         
        # Calculate remaining need
        remaining_target = self.config.total_target - state.total_produced
        current_position = sum(state.inventory.values()) + state.get_total_incoming()
        remaining_need = max(0, remaining_target - current_position)

        # Base cases
        if self._is_terminal(state) or remaining_need == 0:
            # print("hit base case")
            orders[max(orders.keys())] = remaining_need
            # Simulate to end of horizon to get cost estimate
            sim_state = state
            total_cost = 0
            while sim_state.total_produced < self.config.total_target and sim_state.time < self.config.planning_horizon - 1:
                sim_state, costs = sim_state.next_state(orders, self.config)
                total_cost += costs["total"]
            return orders, total_cost
        
        # Generate and evaluate candidates
        candidates = self._generate_candidates(state, remaining_need)
        best_orders, best_cost = self._evaluate_candidates(state, candidates, quit_cost=quit_cost)

        self.cache[self._get_state_key] = best_orders, best_cost
        
        return best_orders, best_cost
        
    def _generate_candidates(self, state: EnvironmentState, remaining_need: float) -> List[Dict[int, float]]:
        """Generate candidates using probability-weighted demand quantities"""
        candidates = []
        
        # Calculate remaining horizon
        remaining_horizon = self.config.planning_horizon - state.time
        
        # Check if we can actually receive orders in time
        current_process = state.get_current_process()
        
        # If no orders can arrive in time, return empty order
        if remaining_horizon <= min(self.config.lead_times[current_process:]):
            return [{p: 0.0 for p in range(self.config.num_processes)}]
        
        # Generate order quantities including the full remaining need
        base_quantities = [0, remaining_need]  # Always consider ordering nothing or everything
        
        # Add some intermediate quantities for more granular exploration
        if remaining_need > 0:
            intermediate = [remaining_need * q for q in [0.5]]
            base_quantities.extend(intermediate)
        
        # Remove duplicates and ensure non-negative
        quantities = sorted(list(set([max(0, int(q)) for q in base_quantities])))
        
        # Generate candidates for current, next and last state combinations
        process_range = range(state.get_current_process(), self.config.num_processes)
        current_process = process_range[0]
        next_process = process_range[1] if len(process_range) > 1 else process_range[0]
        last_process = process_range[-1]
        
        for curr_qty in quantities:
            for next_qty in quantities:
                for last_qty in quantities:
                    orders = {proc: 0.0 for proc in range(self.config.num_processes)}
                    orders[current_process] = curr_qty
                    if next_process != current_process:
                        orders[next_process] = next_qty
                    if last_process != next_process and last_process != current_process:
                        orders[last_process] = last_qty
                    candidates.append(orders)
                
        return candidates
                       
    def _calculate_revision_probas(self, state: EnvironmentState, periods: int) -> List[float]:
        """
        Calculate probability weights for each future period based on bernoulli trials
        """
        current_process = state.get_current_process()
        forced_next_state = state.next_state({}, self.config, advance=True)[0]
        in_warning = state.get_current_process() != forced_next_state.get_current_process()
        
        # Get transition probability
        p = state.get_transition_proba(config=self.config)
            
        # Calculate probability of still being in same revision for each period
        cumulative_probas = [(1 - p) ** t for t in range(periods)]
        
        # If not in warning state, also consider probability of next transition
        if not in_warning:
            p_next = forced_next_state.get_transition_proba(self.config)
            for t in range(1, periods):
                # Probability of transitioning at time t and staying there
                transition_proba = p * (1 - p) ** (t-1) * (1 - p_next) ** (periods - t)
                cumulative_probas[t] += transition_proba

        return cumulative_probas

    def _evaluate_candidates(self, state: EnvironmentState, 
                           candidates: List[Dict[int, float]],
                           quit_cost: float = None) -> Dict[int, float]:
        """Evaluate candidates with full cost consideration"""
        best_cost = float('inf') if not quit_cost else quit_cost
        best_orders = None
        
        remaining_target = self.config.total_target - state.total_produced
        current_position = sum(state.inventory.values()) + state.get_total_incoming()

        # Calculate expected periods until transition
        weights = self._calculate_revision_probas(state, self.config.planning_horizon - state.time)
        expected_till_transition = max(1, int(sum(weights)))  # At least 1 period
        
        for k, orders in enumerate(candidates):
            if state.time == 0:
                print(f"{k} of {len(candidates)}")
            total_cost = 0
            
            # Simulate state transitions and accumulate costs
            trivial_order = {}
            sim_state = copy.deepcopy(state)
            for i in range(expected_till_transition - 1):
                if i == 0:
                    sim_order = orders
                else:
                    sim_order = trivial_order
                
                next_state, costs = sim_state.next_state(sim_order, self.config, advance=False, demand_active=True)
                total_cost += costs["total"]
                sim_state = next_state
                
            # Handle final transition with revision advance
            next_state, costs = sim_state.next_state(trivial_order, self.config, advance=True, demand_active=True)
            total_cost += costs["total"]

            # early exit
            if total_cost >= best_cost:
                continue
            
            remaining_quit_cost = best_cost - total_cost
            total_cost += self.get_action_and_cost(next_state, quit_cost = remaining_quit_cost)[1]
            
            # Update best solution if this is lowest cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_orders = orders
            
        default_orders = {p: 0.0 for p in range(self.config.num_processes)}
        return (best_orders if best_orders else default_orders, best_cost)
        
    def _is_terminal(self, state: EnvironmentState) -> bool:
        """Check if we're in a terminal state"""
        if state.total_produced >= self.config.total_target:
            return True
            
        current_process = state.get_current_process()
        min_lead_time = min(self.config.lead_times[current_process:])
        time_remaining = self.config.planning_horizon - state.time
        
        return time_remaining <= min_lead_time