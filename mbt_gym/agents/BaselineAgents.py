from copy import deepcopy

import gym

import numpy as np
import warnings
from scipy.linalg import expm
from scipy.interpolate import interp1d
from scipy.sparse.linalg import expm_multiply

from mbt_gym.agents.Agent import Agent
from mbt_gym.gym.TradingEnvironment import TradingEnvironment, INVENTORY_INDEX, TIME_INDEX, BID_INDEX, ASK_INDEX
from mbt_gym.rewards.RewardFunctions import CjMmCriterion, PnL, RunningTargetInventoryPenalty
from mbt_gym.stochastic_processes.price_impact_models import PriceImpactModel, TemporaryAndPermanentPriceImpact
from mbt_gym.stochastic_processes.midprice_models import AmmSelfContainedMidpriceModel, AmmSelfContainedMidGeopriceModel

class RandomAgent(Agent):
    def __init__(self, env: gym.Env, seed: int = None):
        self.action_space = deepcopy(env.action_space)
        self.action_space.seed(seed)
        self.num_trajectories = env.num_trajectories

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.action_space.sample().reshape(1, -1), self.num_trajectories, axis=0)


class FixedActionAgent(Agent):
    def __init__(self, fixed_action: np.ndarray, env: gym.Env):
        self.fixed_action = fixed_action
        self.env = env

    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.repeat(self.fixed_action.reshape(1, -1), self.env.num_trajectories, axis=0)


class FixedSpreadAgent(Agent):
    def __init__(self, env: gym.Env, half_spread: float = 1.0, offset: float = 0.0):
        self.half_spread = half_spread
        self.offset = offset
        self.env = env

    def get_action(self, state: np.ndarray) -> np.ndarray:
        action = np.array([[self.half_spread - self.offset, self.half_spread + self.offset]])
        return np.repeat(action, self.env.num_trajectories, axis=0)


class HumanAgent(Agent):
    def get_action(self, state: np.ndarray):
        bid = float(input(f"Current state is {state}. How large do you want to set midprice-bid half spread? "))
        ask = float(input(f"Current state is {state}. How large do you want to set ask-midprice half spread? "))
        return np.array([bid, ask])


class AvellanedaStoikovAgent(Agent):
    def __init__(self, risk_aversion: float = 0.1, env: TradingEnvironment = None):
        self.risk_aversion = risk_aversion
        self.env = env or TradingEnvironment()
        assert isinstance(self.env, TradingEnvironment)
        self.terminal_time = self.env.terminal_time
        self.volatility = self.env.midprice_model.volatility
        self.rate_of_arrival = self.env.arrival_model.intensity
        self.fill_exponent = self.env.fill_probability_model.fill_exponent

    def get_action(self, state: np.ndarray):
        inventory = state[:, 1]
        time = state[:, 2]
        action = self._get_action(inventory, time)
        if action.min() < 0:
            warnings.warn("Avellaneda-Stoikov agent is quoting a negative spread")
        return action

    def _get_price_adjustment(self, inventory: int, time: float) -> float:
        return inventory * self.risk_aversion * self.volatility**2 * (self.terminal_time - time)

    def _get_spread(self, time: float) -> float:
        if self.risk_aversion == 0:
            return 2 / self.fill_exponent  # Limit as risk aversion -> 0
        volatility_aversion_component = self.risk_aversion * self.volatility**2 * (self.terminal_time - time)
        fill_exponent_component = 2 / self.risk_aversion * np.log(1 + self.risk_aversion / self.fill_exponent)
        return volatility_aversion_component + fill_exponent_component

    def _get_action(self, inventory: int, time: float):
        bid_half_spread = (self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2).reshape(-1, 1)
        ask_half_spread = (-self._get_price_adjustment(inventory, time) + self._get_spread(time) / 2).reshape(-1, 1)
        return np.append(bid_half_spread, ask_half_spread, axis=1)


class CarteaJaimungalMmAgent(Agent):
    def __init__(
        self,
        env: TradingEnvironment = None,
        max_inventory: int = 100,
    ):
        self.env = env or TradingEnvironment()
        assert self.env.action_type == "limit"
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL)), "Reward function for CjMmAgent is incorrect."
        self.kappa = self.env.fill_probability_model.fill_exponent
        self.num_trajectories = self.env.num_trajectories
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            self.lambdas = self.env.arrival_model.intensity
            self.max_inventory = max_inventory
            self.a_matrix, self.z_vector = self._calculate_a_and_z()
            self.large_depth = 10_000

    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            assert (
                state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray):
        deltas = np.zeros(shape=(self.num_trajectories, 2))
        h_t = self._calculate_ht(current_time)
        # If the inventory goes above the max level, we quote a large depth to bring it back and quote on the opposite
        # side as if we had an inventory equal to sign(inventory) * self.max_inventory.
        indices = np.clip(self.max_inventory + inventories, 0, 2 * self.max_inventory)
        indices = indices.astype(int)
        indices_minus_one = np.clip(indices - 1, 0, 2 * self.max_inventory)
        indices_plus_one = np.clip(indices + 1, 0, 2 * self.max_inventory)
        h_0 = h_t[indices]
        h_plus_one = h_t[indices_plus_one]
        h_minus_one = h_t[indices_minus_one]
        max_inventory_bid = h_plus_one == h_0
        max_inventory_ask = h_minus_one == h_0
        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one + h_0 + self.large_depth * max_inventory_bid).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask).reshape(-1)
        return deltas

    def _calculate_ht(self, current_time: float) -> float:
        omega_function = self._calculate_omega(current_time)
        return 1 / self.kappa * np.log(omega_function)

    def _calculate_omega(self, current_time: float):
        """This is Equation (10.11) from [CJP15]."""
        return np.matmul(expm(self.a_matrix * (self.terminal_time - current_time)), self.z_vector)

    def _calculate_a_and_z(self):
        matrix_size = 2 * self.max_inventory + 1
        Amatrix = np.zeros(shape=(matrix_size, matrix_size))
        z_vector = np.zeros(shape=(matrix_size, 1))
        for i in range(matrix_size):
            inventory = self.max_inventory - i
            Amatrix[i, i] = -self.phi * self.kappa * inventory**2
            z_vector[i, 0] = np.exp(-self.alpha * self.kappa * inventory**2)
            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[0] * np.exp(-1)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[1] * np.exp(-1)
        return Amatrix, z_vector


class CarteaJaimungalOeAgent(Agent):
    def __init__(
        self,
        phi: float = 2 * 10 ** (-4),
        alpha: float = 0.0001,
        env: TradingEnvironment = None,
    ):
        self.phi = phi
        self.alpha = alpha
        self.env = env or TradingEnvironment()
        self.price_impact_model = env.price_impact_model
        assert self.env.action_type == "speed"
        self.terminal_time = self.env.terminal_time
        self.temporary_price_impact = self.price_impact_model.temporary_impact_coefficient
        self.permanent_price_impact = self.price_impact_model.permanent_impact_coefficient
        self.num_trajectories = self.env.num_trajectories

    def get_action(self, state: np.ndarray):
        action = np.zeros(shape=(self.num_trajectories, 1))
        # The formulae below is in page 147 of Cartea, Jaimungal, Penalva (2015)
        # Algorithmic and High-Frequency Trading
        # Cambridge University Press
        gamma = np.sqrt(self.phi / self.temporary_price_impact)
        zeta = (self.alpha - 0.5 * self.permanent_price_impact + np.sqrt(self.temporary_price_impact * self.phi)) / (
            self.alpha - 0.5 * self.permanent_price_impact - np.sqrt(self.temporary_price_impact * self.phi)
        )
        initial_inventory = self.env.initial_inventory

        time_left = self.terminal_time - state[0, TIME_INDEX]
        action[:, :] = (
            gamma
            * initial_inventory
            * (
                (zeta * np.exp(gamma * time_left) + np.exp(-gamma * time_left))
                / (zeta * np.exp(gamma * self.terminal_time) - np.exp(-gamma * self.terminal_time))
            )
        )
        return -np.sign(initial_inventory) * action






##########################################################################################################################################
# AMM pools agents
##########################################################################################################################################
class PoolMmAgent(Agent):
    def __init__(
        self,
        env: TradingEnvironment = None,
        max_inventory: int = 200,
        min_inventory: int = 0,
        target_inventory: int = 100,
        verbose: bool = False,
    ):
        self.env = env or TradingEnvironment()
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL, RunningTargetInventoryPenalty)), "Reward function for AmmAgent is incorrect."
        assert isinstance(self.env.midprice_model, AmmSelfContainedMidpriceModel), "Midprice model for AMM trader is incorrect."
        self.kappa = self.env.fill_probability_model.fill_exponent 
        self.num_trajectories = self.env.num_trajectories
        self.target_inventory = target_inventory
        self.verbose = verbose
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            self.lambdas = self.env.arrival_model.intensity
            
            self.max_inventory = max_inventory
            self.min_inventory = min_inventory
            self.max_index  =  int((self.max_inventory - self.min_inventory)/self.env.trader.unit_size)
            self.a_matrix, self.z_vector = self._calculate_a_and_z()
            self.large_depth = 10_000

    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            assert (
                state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray):
        deltas = np.zeros(shape=(self.num_trajectories, 2))
        h_t = self._calculate_ht(current_time)
        # If the inventory goes above the max level, we quote a large depth to bring it back and quote on the opposite
        # side as if we had an inventory equal to sign(inventory) * self.max_inventory.
        
        if self.verbose:
            print('\n')
            print('pool inventory = ', inventories, 'min inventory= ', self.min_inventory, ', max inventory:', self.max_inventory, ". target inventory:", self.target_inventory)
  
        indices = np.clip( int((-inventories+self.max_inventory)/self.env.trader.unit_size), 0, self.max_index-1)
        indices = indices.astype(int)
        
        #print('pool index = ', indices)
        #print('max index = ', self.max_index)
        indices_minus_one = np.clip(indices + 1, 0, self.max_index-1)
        indices_plus_one  = np.clip(indices - 1, 0, self.max_index-1)
        h_0 = h_t[indices]
        h_plus_one = h_t[indices_plus_one]
        h_minus_one = h_t[indices_minus_one]
        max_inventory_bid = indices_plus_one == indices
        max_inventory_ask = indices_minus_one == indices
        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one + h_0 + self.large_depth * max_inventory_bid - 1/self.env.midprice_model.jump_size_L).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask + 1/self.env.midprice_model.jump_size_L).reshape(-1)
        #print('delta minus= ', deltas[:,BID_INDEX], ', delta plus:', deltas[:,ASK_INDEX])
        #deltas = np.clip(deltas, 0, 1e5)
        return deltas

    def _calculate_ht(self, current_time: float) -> float:
        omega_function = self._calculate_omega(current_time)
        return (1 / self.kappa) * np.log(omega_function)

    def _calculate_omega(self, current_time: float):
        """This is Equation (10.11) from [CJP15]."""
        return np.matmul(expm(self.a_matrix * (self.terminal_time - current_time)), self.z_vector)

    def _calculate_a_and_z(self):
        matrix_size = self.max_index
        Amatrix     = np.zeros(shape=(matrix_size, matrix_size))
        z_vector    = np.zeros(shape=(matrix_size, 1))
        for i in range(matrix_size):
            inventory = self.max_inventory - i*self.env.trader.unit_size
            Amatrix[i, i] = -self.phi * self.kappa * (inventory-self.target_inventory)**2 / self.env.trader.unit_size
            z_vector[i, 0] = np.exp(-self.alpha * self.kappa  * (inventory-self.target_inventory)**2 / self.env.trader.unit_size ) 
            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[BID_INDEX] * np.exp(-1) * np.exp(self.kappa / self.env.midprice_model.jump_size_L)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[ASK_INDEX] * np.exp(-1) * np.exp(-self.kappa / self.env.midprice_model.jump_size_L)
        return Amatrix, z_vector



class PooGeoMmAgent(Agent):
    def __init__(
        self,
        env: TradingEnvironment = None,
        verbose: bool = False,
        target_inventory: int = 100,
        min_inventory: int = 100,
        size_inventory_space: int = 1000, # inventory space go from max_inventory to min_inventory / (1+Delta)^(-size_inventory_space)
    ):
        self.env = env or TradingEnvironment()
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL, RunningTargetInventoryPenalty)), "Reward function for AmmAgent is incorrect."
        assert isinstance(self.env.midprice_model, AmmSelfContainedMidGeopriceModel), "Midprice model for AMM trader is incorrect."
        self.kappa = self.env.fill_probability_model.fill_exponent 
        self.num_trajectories = self.env.num_trajectories
        self.target_inventory = target_inventory
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            self.lambdas = self.env.arrival_model.intensity
            self.verbose = verbose

            # inventory management
            self.size_inventory_space = size_inventory_space
            self.min_inventory = min_inventory
            self.unit_size = self.env.trader.unit_size
            self.max_inventory = self.min_inventory * (1 + self.unit_size)**(size_inventory_space-1)
            self.inventory_space = np.array([self.min_inventory*(1+self.unit_size)**i for i in range(self.size_inventory_space)])

            #self.max_index  =  int((self.max_inventory - self.min_inventory)/self.env.trader.unit_size)
            self.a_matrix, self.z_vector = self._calculate_a_and_z()
            self.large_depth = 10_000

    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            assert (
                state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray):
        deltas  = np.zeros(shape=(self.num_trajectories, 2))
        z       = self.env.midprice_model.current_state
        h_t   = self._calculate_ht(current_time, z)
        indices = np.clip( int( np.log(inventories/self.min_inventory)/np.log(1+self.unit_size)) , 0, self.size_inventory_space-1)

        #print('pool index = ', indices)
        #print('max index = ', self.max_index)
        indices_minus_one = np.clip(indices + 1, 0, self.size_inventory_space-1)
        indices_plus_one  = np.clip(indices - 1, 0, self.size_inventory_space-1)
        h_0 = h_t[indices, 0]
        h_plus_one = h_t[indices_plus_one, 0]
        h_minus_one = h_t[indices_minus_one, 0]
        max_inventory_bid = indices_plus_one == indices
        max_inventory_ask = indices_minus_one == indices
        print('indices:', indices)
        print('bid:', - h_plus_one + h_0)
        print('ask:', - h_minus_one + h_0)
        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one + h_0 + self.large_depth * max_inventory_bid + 1).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask - 1).reshape(-1)
        #deltas = np.clip(deltas, 0, 1e5)
        return deltas

    def _calculate_ht(self, current_time: float, z: float) -> float:
        omega_function = self._calculate_omega(current_time)
        #return (self.unit_size * z / self.kappa) * self.inventory_space[::-1] * np.log(omega_function)
        return np.log(omega_function) / self.kappa # not the real theta

    def _calculate_omega(self, current_time: float):
        """This is Equation (10.11) from [CJP15]."""
        return np.matmul(expm(self.a_matrix * (self.terminal_time - current_time)), self.z_vector)

    def _calculate_a_and_z(self):
        matrix_size = self.size_inventory_space

        Amatrix     = np.zeros(shape=(matrix_size, matrix_size))
        z_vector    = np.zeros(shape=(matrix_size, 1))
        for (i, inventory) in enumerate(self.inventory_space[::-1]):
            Amatrix[i, i] = - self.phi * self.kappa * (inventory-self.target_inventory)**2  / self.unit_size / inventory

            #(inventory-self.target_inventory)**2 / self.env.trader.unit_size / inventory
            z_vector[i, 0] = np.exp(-self.alpha * self.kappa  * (inventory-self.target_inventory)**2 / self.env.trader.unit_size / inventory ) 

            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[BID_INDEX] * np.exp(-1 + self.kappa)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[ASK_INDEX] * np.exp(-1 - self.kappa)

        
        return Amatrix, z_vector
    


class PooGeoInterpMmAgent(Agent):
    def __init__(
        self,
        env: TradingEnvironment = None,
        verbose: bool = False,
        target_inventory: int = 100,
        min_inventory: int = 100,
        size_inventory_space: int = 1000, # inventory space go from max_inventory to min_inventory / (1+Delta)^(-size_inventory_space)
    ):
        self.env = env or TradingEnvironment()
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL, RunningTargetInventoryPenalty)), "Reward function for AmmAgent is incorrect."
        assert isinstance(self.env.midprice_model, AmmSelfContainedMidGeopriceModel), "Midprice model for AMM trader is incorrect."
        self.kappa = self.env.fill_probability_model.fill_exponent 
        self.num_trajectories = self.env.num_trajectories
        self.target_inventory = target_inventory
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            self.lambdas = self.env.arrival_model.intensity
            self.verbose = verbose

            # inventory management
            self.size_inventory_space = size_inventory_space
            self.min_inventory = min_inventory
            self.unit_size = self.env.trader.unit_size
            self.max_inventory = self.min_inventory * (1 + self.unit_size)**(size_inventory_space-1)
            self.inventory_space = np.array([self.min_inventory*(1+self.unit_size)**i for i in range(self.size_inventory_space)])

            #self.max_index  =  int((self.max_inventory - self.min_inventory)/self.env.trader.unit_size)
            self.a_matrix, self.z_vector, self.omegas = self._calculate_a_and_z()
            self.large_depth = 10_000

    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            assert (
                state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray):
        deltas  = np.zeros(shape=(self.num_trajectories, 2))
        z       = self.env.midprice_model.current_state
        h_t   = self._calculate_ht(current_time, z)
        indices = np.clip( int( np.log(inventories/self.min_inventory)/np.log(1+self.unit_size)) , 0, self.size_inventory_space-1)

        #print('pool index = ', indices)
        #print('max index = ', self.max_index)
        indices_minus_one = np.clip(indices + 1, 0, self.size_inventory_space-1)
        indices_plus_one  = np.clip(indices - 1, 0, self.size_inventory_space-1)
        h_0 = h_t[indices, 0]
        h_plus_one = h_t[indices_plus_one, 0]
        h_minus_one = h_t[indices_minus_one, 0]
        max_inventory_bid = indices_plus_one == indices
        max_inventory_ask = indices_minus_one == indices
        print('indices:', indices)
        print('bid:', - h_plus_one + h_0)
        print('ask:', - h_minus_one + h_0)
        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one + h_0 + self.large_depth * max_inventory_bid + 1).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask - 1).reshape(-1)
        #deltas = np.clip(deltas, 0, 1e5)
        return deltas

    def _calculate_ht(self, current_time: float, z: float) -> float:
        return (1/ self.kappa) * np.log(self.omegas[int((self.terminal_time - current_time)/self.env.step_size)-1, :, :]) 

        #omega_function = self._calculate_omega(current_time)
        #return (self.unit_size * z / self.kappa) * self.inventory_space[::-1] * np.log(omega_function)
        #return np.log(omega_function) / self.kappa # not the real theta

    def _calculate_omega(self, current_time: float):
        """This is Equation (10.11) from [CJP15]."""
        return np.matmul(expm(self.a_matrix * (self.terminal_time - current_time)), self.z_vector)

    def _calculate_a_and_z(self):
        matrix_size = self.size_inventory_space

        Amatrix     = np.zeros(shape=(matrix_size, matrix_size))
        z_vector    = np.zeros(shape=(matrix_size, 1))
        for (i, inventory) in enumerate(self.inventory_space[::-1]):
            Amatrix[i, i] = - self.phi * self.kappa * (inventory-self.target_inventory)**2  / self.unit_size / inventory

            #(inventory-self.target_inventory)**2 / self.env.trader.unit_size / inventory
            z_vector[i, 0] = np.exp(-self.alpha * self.kappa  * (inventory-self.target_inventory)**2 / self.env.trader.unit_size / inventory ) 

            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[ASK_INDEX] * np.exp(-1 + self.kappa)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[BID_INDEX] * np.exp(-1 - self.kappa)

        # Compute exp(A (T-t)) @ z for all values of t
        omegas = expm_multiply(Amatrix, 
                                z_vector, 
                                start = 0, 
                                stop  = self.env.terminal_time, 
                                num   = int(self.env.terminal_time/self.env.step_size), endpoint=True)

        return Amatrix, z_vector, omegas
        
        #return Amatrix, z_vector
    


class PoolInterpMmAgent(Agent):
    def __init__(
        self,
        env: TradingEnvironment = None,
        max_inventory: int = 200,
        min_inventory: int = 0,
        target_inventory: int = 100,
        verbose: bool = False,
    ):
        self.env = env or TradingEnvironment()
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL, RunningTargetInventoryPenalty)), "Reward function for AmmAgent is incorrect."
        assert isinstance(self.env.midprice_model, AmmSelfContainedMidpriceModel), "Midprice model for AMM trader is incorrect."
        self.kappa = self.env.fill_probability_model.fill_exponent 
        self.num_trajectories = self.env.num_trajectories
        self.target_inventory = target_inventory
        self.verbose = verbose
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            self.lambdas = self.env.arrival_model.intensity
            
            self.max_inventory = max_inventory
            self.min_inventory = min_inventory
            self.max_index  =  int((self.max_inventory - self.min_inventory)/self.env.trader.unit_size)
            self.inventory_space = np.linspace(self.min_inventory, self.max_inventory, self.max_index)
            self.a_matrix, self.z_vector, self.omegas = self._calculate_a_and_z()
            self.large_depth = 10_000

    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            assert (
                state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray):
        deltas = np.zeros(shape=(self.num_trajectories, 2))
        h_t = self._calculate_ht(current_time)
        # If the inventory goes above the max level, we quote a large depth to bring it back and quote on the opposite
        # side as if we had an inventory equal to sign(inventory) * self.max_inventory.
        
        if self.verbose:
            print('\n')
            print('pool inventory = ', inventories, 'min inventory= ', self.min_inventory, ', max inventory:', self.max_inventory, ". target inventory:", self.target_inventory)
  
        indices = np.clip( int((-inventories+self.max_inventory)/self.env.trader.unit_size), 0, self.max_index-1)
        indices = indices.astype(int)
        
        #print('pool index = ', indices)
        #print('max index = ', self.max_index)
        indices_minus_one = np.clip(indices + 1, 0, self.max_index-1)
        indices_plus_one  = np.clip(indices - 1, 0, self.max_index-1)
        h_0 = h_t[indices]
        h_plus_one = h_t[indices_plus_one]
        h_minus_one = h_t[indices_minus_one]
        max_inventory_bid = indices_plus_one == indices
        max_inventory_ask = indices_minus_one == indices

        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one + h_0 + self.large_depth * max_inventory_bid - 1/self.env.midprice_model.jump_size_L).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask + 1/self.env.midprice_model.jump_size_L).reshape(-1)
        #print('delta minus= ', deltas[:,BID_INDEX], ', delta plus:', deltas[:,ASK_INDEX])
        #deltas = np.clip(deltas, 0, 1e5)
        return deltas

    def _calculate_ht(self, current_time: float) -> float:
        return (1/ self.kappa) * np.log(self.omegas[int((self.terminal_time - current_time)/self.env.step_size)-1, :, :]) 

    #def _calculate_omega(self, current_time: float):
    #    """This is Equation (10.11) from [CJP15]."""
    #    return np.matmul(expm(self.a_matrix * (self.terminal_time - current_time)), self.z_vector)

    def _calculate_a_and_z(self):
        matrix_size = self.max_index
        Amatrix     = np.zeros(shape=(matrix_size, matrix_size))
        z_vector    = np.zeros(shape=(matrix_size, 1))
        for i in range(matrix_size):
            inventory = self.max_inventory - i*self.env.trader.unit_size
            Amatrix[i, i] = - self.phi * self.kappa * (inventory-self.target_inventory)**2 / self.env.trader.unit_size
            z_vector[i, 0] = np.exp(-self.alpha * self.kappa  * (inventory-self.target_inventory)**2 / self.env.trader.unit_size )
            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[BID_INDEX] * np.exp(-1) * np.exp(self.kappa / self.env.midprice_model.jump_size_L)
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[ASK_INDEX] * np.exp(-1) * np.exp(-self.kappa / self.env.midprice_model.jump_size_L)

        # Compute exp(A (T-t)) @ z for all values of t
        omegas = expm_multiply(Amatrix, 
                                z_vector, 
                                start = 0, 
                                stop  = self.env.terminal_time, 
                                num   = int(self.env.terminal_time/self.env.step_size), endpoint=True)

        return Amatrix, z_vector, omegas






class PoolInterpGEtaMmAgent(Agent): #pool agent with interpolation and general etas
    def __init__(
        self,
        env: TradingEnvironment = None,
        max_inventory: int = 200,
        min_inventory: int = 0,
        target_inventory: int = 100,
        verbose: bool = False,
    ):
        self.env = env or TradingEnvironment()
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL, RunningTargetInventoryPenalty)), "Reward function for AmmAgent is incorrect."
        assert isinstance(self.env.midprice_model, AmmSelfContainedMidpriceModel), "Midprice model for AMM trader is incorrect."
        self.kappa = self.env.fill_probability_model.fill_exponent 
        self.num_trajectories = self.env.num_trajectories
        self.target_inventory = target_inventory
        self.verbose = verbose
        if isinstance(self.env.reward_function, PnL):
            self.inventory_neutral = True
            self.risk_neutral_action = 1 / self.kappa * np.ones((env.num_trajectories, env.action_space.shape[0]))
        else:
            self.inventory_neutral = False
            self.phi = env.reward_function.per_step_inventory_aversion
            self.alpha = env.reward_function.terminal_inventory_aversion
            assert self.env.reward_function.inventory_exponent == 2.0, "Inventory exponent must be = 2."
            self.terminal_time = self.env.terminal_time
            self.lambdas = self.env.arrival_model.intensity
            
            self.max_inventory = max_inventory
            self.min_inventory = min_inventory
            self.max_index  =  round((self.max_inventory - self.min_inventory)/self.env.trader.unit_size)
            self.inventory_space = np.linspace(self.min_inventory, self.max_inventory, self.max_index)
            self.a_matrix, self.z_vector, self.omegas = self._calculate_a_and_z()
            self.large_depth = 10_000

    def get_action(self, state: np.ndarray):
        if self.inventory_neutral:
            return self.risk_neutral_action
        else:
            assert (
                state[0, TIME_INDEX] == state[-1, TIME_INDEX]
            ), "CarteaJaimungalMmAgent needs to be called on a tensor with a uniform time stamp."
            current_time = state[0, TIME_INDEX]
            inventories = state[:, INVENTORY_INDEX]
            return self._calculate_deltas(inventories=inventories, current_time=current_time)

    def _calculate_deltas(self, current_time: float, inventories: np.ndarray):
        deltas = np.zeros(shape=(self.num_trajectories, 2))
        h_t = self._calculate_ht(current_time)
        # If the inventory goes above the max level, we quote a large depth to bring it back and quote on the opposite
        # side as if we had an inventory equal to sign(inventory) * self.max_inventory.
        
        if self.verbose:
            print('\n')
            print('pool inventory = ', inventories, 'min inventory= ', self.min_inventory, ', max inventory:', self.max_inventory, ". target inventory:", self.target_inventory)
  
        indices = np.clip( int((-inventories+self.max_inventory)/self.env.trader.unit_size), 0, self.max_index-1)
        indices = indices.astype(int)
        
        #print('pool index = ', indices)
        #print('max index = ', self.max_index)
        indices_minus_one = np.clip(indices + 1, 0, self.max_index-1)
        indices_plus_one  = np.clip(indices - 1, 0, self.max_index-1)
        h_0 = h_t[indices]
        h_plus_one = h_t[indices_plus_one]
        h_minus_one = h_t[indices_minus_one]
        max_inventory_bid = indices_plus_one == indices
        max_inventory_ask = indices_minus_one == indices


        Z =  self.env.midprice_model.current_state
        tmpeta_bid = self.env.midprice_model.eta_bid(inventories, 
                                                                    self.env.trader.unit_size, 
                                                                    self.env.midprice_model.jump_size_L)
        tmpeta_ask = self.env.midprice_model.eta_ask(inventories, 
                                                                    self.env.trader.unit_size, 
                                                                    self.env.midprice_model.jump_size_L)
        
        
        adjust_bid =  -(inventories+self.env.trader.unit_size)*self.env.midprice_model.eta_bid(inventories, 
                                                                    self.env.trader.unit_size, 
                                                                    self.env.midprice_model.jump_size_L)/self.env.trader.unit_size
        adjust_ask =  (inventories-self.env.trader.unit_size)*self.env.midprice_model.eta_ask(inventories, 
                                                                    self.env.trader.unit_size, 
                                                                    self.env.midprice_model.jump_size_L)/self.env.trader.unit_size

        deltas[:, BID_INDEX] = (1 / self.kappa - h_plus_one  + h_0 + self.large_depth * max_inventory_bid + adjust_bid).reshape(-1)
        deltas[:, ASK_INDEX] = (1 / self.kappa - h_minus_one + h_0 + self.large_depth * max_inventory_ask + adjust_ask).reshape(-1)

        if False:
            print('impact on price from bid:', tmpeta_bid)
            print('impact on price from ask:',tmpeta_ask )
            print('\n')
            print('impact on inventory from bid:', self.env.trader.unit_size)
            print('impact on inventory from ask:', - self.env.trader.unit_size)
            print('\n')
            print('impact on y*Z from bid:', (inventories + self.env.trader.unit_size)*(Z-tmpeta_bid) - inventories * Z )
            print('impact on y*Z from ask:', (inventories - self.env.trader.unit_size)*(Z+tmpeta_ask) - inventories * Z)
            print('\n')
            print('impact on cash from bid:', -self.env.trader.unit_size * (Z  ) + (inventories + self.env.trader.unit_size)*(Z-tmpeta_bid) - inventories * Z )
            print('impact on cash from ask:', self.env.trader.unit_size * (Z  ) + (inventories - self.env.trader.unit_size)*(Z+tmpeta_ask) - inventories * Z)
            print('\n')
            print('impact on cash with deltas from bid:', -self.env.trader.unit_size * (Z - deltas[:, BID_INDEX] ) + (inventories + self.env.trader.unit_size)*(Z-tmpeta_bid) - inventories * Z )
            print('impact on cash with deltas from ask:', self.env.trader.unit_size * (Z + deltas[:, ASK_INDEX] ) + (inventories - self.env.trader.unit_size)*(Z+tmpeta_ask) - inventories * Z)
    

        deltas = np.clip(deltas, 0, 1e5)
        return deltas

    def _calculate_ht(self, current_time: float) -> float:
        return (1/ self.kappa) * np.log(self.omegas[int((self.terminal_time - current_time)/self.env.step_size)-1, :, :]) 

    def _calculate_a_and_z(self):
        matrix_size = self.max_index
        Amatrix     = np.zeros(shape=(matrix_size, matrix_size))
        z_vector    = np.zeros(shape=(matrix_size, 1))
        for i in range(matrix_size):
            inventory = self.max_inventory - i*self.env.trader.unit_size
            Amatrix[i, i] = - self.phi * self.kappa * (inventory-self.target_inventory)**2 / self.env.trader.unit_size
            z_vector[i, 0] = np.exp(-self.alpha * self.kappa  * (inventory-self.target_inventory)**2 / self.env.trader.unit_size )
            if i + 1 < matrix_size:
                Amatrix[i, i + 1] = self.lambdas[ASK_INDEX] * np.exp(-1) * np.exp(self.kappa * (matrix_size - i - 1) *\
                                                     self.env.midprice_model.eta_ask(inventory, 
                                                                                     self.env.midprice_model.unit_size, 
                                                                                     self.env.midprice_model.jump_size_L) )
            if i > 0:
                Amatrix[i, i - 1] = self.lambdas[BID_INDEX] * np.exp(-1) * np.exp(- self.kappa * (matrix_size - i + 1) *\
                                                     self.env.midprice_model.eta_bid(inventory, 
                                                                                     self.env.trader.unit_size, 
                                                                                     self.env.midprice_model.jump_size_L) )
        # Compute exp(A (T-t)) @ z for all values of t        
        omegas = expm_multiply(Amatrix, 
                               z_vector, 
                               start = 0, 
                               stop  = self.env.terminal_time, 
                               num   = int(self.env.terminal_time/self.env.step_size), endpoint=True)

        return Amatrix, z_vector, omegas




##########################################################################################################################################
# Arbitrageur agents
##########################################################################################################################################

class ArbitrageurAmmAgent_old(Agent):
    def __init__(
        self,
        env: TradingEnvironment = None,
        agent: PoolMmAgent = None,
        max_inventory: int = 150,
        min_inventory: int = -150,
        trade_size: int = 1,
    ):
        self.env = env or TradingEnvironment()
        self.agent = agent
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL)), "Reward function for AmmAgent is incorrect."
        self.num_trajectories = self.env.num_trajectories
        self.max_inventory = max_inventory
        self.min_inventory = min_inventory
        self.trade_size = trade_size
        self.historical_pool_prices = []
        self.historical_ba = []
        
        
    def get_action(self, state: np.ndarray):
        price_S = self.env.midprice_model.current_state
        price_Z = self.agent.env.midprice_model.current_state 
        state_agent = self.agent.env.state
        deltas    = self.agent.get_action(state_agent)
        delta_ask = deltas[:, ASK_INDEX]
        delta_bid = deltas[:, BID_INDEX]
        indicator_buy = (state[:,INVENTORY_INDEX] < self.max_inventory)
        indicator_sell = (state[:,INVENTORY_INDEX] > self.min_inventory)

        action = int((price_S>price_Z + delta_ask)*indicator_buy) - int((price_S<(price_Z - delta_bid))*indicator_sell)

        reshaped_action = np.reshape(action*self.trade_size, (self.num_trajectories,1))
        
        # update the pool's agent's trader
        print('S = ', price_S, ', Z=', price_Z, ' deltas = ', deltas, '   *** action = ', action)
        self.agent.env._update_market_state(arrivals = np.ones((self.num_trajectories, 2)), 
                                            fills    = np.concatenate( (reshaped_action<0, reshaped_action>0), axis=1), 
                                            action   = deltas)
        self.agent.env._update_agent_state(arrivals = np.ones((self.num_trajectories, 2)), 
                                            fills    = np.concatenate( (reshaped_action<0, reshaped_action>0), axis=1) , 
                                            action   = deltas)
        self.historical_pool_prices += [price_Z]
        self.historical_ba          += [deltas]
        return reshaped_action
 



class ArbitrageurAmmAgent(Agent):
    def __init__(
        self,
        env: TradingEnvironment = None,
        agent: PoolMmAgent = None,
        max_inventory: int = 150,
        min_inventory: int = -150,
        trade_size: int = 1,
        verbose: bool=False,
    ):
        self.env = env or TradingEnvironment()
        self.agent = agent
        assert isinstance(self.env.reward_function, (CjMmCriterion, PnL)), "Reward function for AmmAgent is incorrect."
        self.num_trajectories = self.env.num_trajectories
        self.max_inventory = max_inventory
        self.min_inventory = min_inventory
        self.trade_size = trade_size
        self.verbose = verbose

        # historical values (TODO: change this)
        self.historical_pool_prices = []
        self.historical_ba = []
        self.historical_pool_inventory = []
        self.pool_earnings_history = []
        self.arb_earnings_history = []
        
    def get_action(self, state: np.ndarray):
        opportunity_arbitrage = True
        cum_action = None

        # close the arbitrage
        while opportunity_arbitrage:
            price_S = self.env.midprice_model.current_state
            price_Z = self.agent.env.midprice_model.current_state 
            state_agent = self.agent.env.state
            deltas    = self.agent.get_action(state_agent)
            delta_ask = deltas[:, ASK_INDEX]
            delta_bid = deltas[:, BID_INDEX]
            indicator_buy = (state[:,INVENTORY_INDEX] < self.max_inventory)
            indicator_sell = (state[:,INVENTORY_INDEX] > self.min_inventory)

            action = int((price_S>price_Z + delta_ask)*indicator_buy) - int((price_S<(price_Z - delta_bid))*indicator_sell)
            reshaped_action = np.reshape(action*self.trade_size, (self.num_trajectories,1))

            # is there an arbitrage opportunity ? 
            opportunity_arbitrage = (action != 0)

            # cumulative action of the arbitrageur
            cum_action   = action if cum_action is None else cum_action+action # cumulative action for the arbitrage

            # Compute the spread between S and Z+-bid/ask spread
            spreadPrices = float((price_S - price_Z - delta_ask)*(price_S>price_Z + delta_ask)) + float((- price_S + price_Z - delta_bid)*(price_S<(price_Z - delta_bid)))
            
            # update the pool's agent's trader
            self.historical_pool_inventory += [(self.agent.env.state[0, TIME_INDEX], self.agent.env.state[:, INVENTORY_INDEX].copy())]

            if action != 0: 
                if self.verbose:
                    print('\n*****************')
                    print('Current time:', self.agent.env.state[0, TIME_INDEX] )
                    print('S = ', price_S[0][0], ', Z=', price_Z[0][0], ' deltas = ', deltas[0], '   *** action = ', action)
                    print('Spread to target', spreadPrices)
                    print('Inventory of pool:', self.agent.env.state[:, INVENTORY_INDEX], 'min inventory= ', self.agent.min_inventory, ', max inventory:', self.agent.max_inventory, ". target inventory:", self.agent.target_inventory)
                    print('Sending an order:', action*self.agent.env.midprice_model.unit_size)

                if action == 1:
                    self.pool_earnings_history += [ (self.agent.env.state[0, TIME_INDEX], np.abs(delta_ask*self.agent.env.midprice_model.unit_size))]
                else:
                    self.pool_earnings_history += [ (self.agent.env.state[0, TIME_INDEX], np.abs(delta_bid*self.agent.env.midprice_model.unit_size))]

                self.arb_earnings_history  += [ (self.agent.env.state[0, TIME_INDEX], np.abs(spreadPrices*self.agent.env.midprice_model.unit_size))]

            else:
                self.pool_earnings_history += [(self.agent.env.state[0, TIME_INDEX], 0)]
                self.arb_earnings_history  += [(self.agent.env.state[0, TIME_INDEX], 0)]
                

            self.agent.env._update_market_state(arrivals = np.ones((self.num_trajectories, 2)), 
                                                fills    = np.concatenate( (reshaped_action<0, reshaped_action>0), axis=1), 
                                                action   = deltas)
            self.agent.env._update_agent_state_no_time(arrivals  = np.ones((self.num_trajectories, 2)), 
                                                fills    = np.concatenate( (reshaped_action<0, reshaped_action>0), axis=1), 
                                                action   = deltas)
            
            self.historical_pool_prices += [(self.agent.env.state[0, TIME_INDEX], price_Z)]
            
            self.historical_ba          += [(self.agent.env.state[0, TIME_INDEX], deltas) ]
        
        # once all trades 
        self.agent.env._update_clock()
        reshaped_action = np.reshape(cum_action*self.trade_size, (self.num_trajectories,1))
        
        
        return reshaped_action







