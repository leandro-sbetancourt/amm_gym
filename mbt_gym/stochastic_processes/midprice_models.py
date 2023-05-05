from math import sqrt
from typing import Optional
import numpy as np

from mbt_gym.stochastic_processes.StochasticProcessModel import StochasticProcessModel

MidpriceModel = StochasticProcessModel

BID_INDEX = 0
ASK_INDEX = 1


CASH_INDEX = 0
INVENTORY_INDEX = 1
TIME_INDEX = 2
ASSET_PRICE_INDEX = 3



class ConstantMidpriceModel(MidpriceModel):
    def __init__(
        self,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price]]),
            max_value=np.array([[initial_price]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        pass


class BrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 2.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * np.sqrt(terminal_time)


class GeometricBrownianMotionMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 0.1,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.drift * self.current_state * self.step_size
            + self.volatility
            * self.current_state
            * sqrt(self.step_size)
            * self.rng.normal(size=(self.num_trajectories, 1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        stdev = sqrt(
            initial_price**2
            * np.exp(2 * self.drift * terminal_time)
            * (np.exp(self.volatility**2 * terminal_time) - 1)
        )
        return initial_price * np.exp(self.drift * terminal_time) + 4 * stdev


class OuMidpriceModel(MidpriceModel):
    def __init__(
        self,
        mean_reversion_level: float = 0.0,
        mean_reversion_speed: float = 1.0,
        volatility: float = 2.0,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.mean_reversion_level = mean_reversion_level
        self.mean_reversion_speed = mean_reversion_speed
        self.volatility = volatility
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state += -self.mean_reversion_speed * (
            self.current_state - self.mean_reversion_level * np.ones((self.num_trajectories, 1))
        ) + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: What should this be?


class ShortTermOuAlphaMidpriceModel(MidpriceModel):
    def __init__(
        self,
        volatility: float = 2.0,
        ou_process: OuMidpriceModel = None,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.volatility = volatility
        self.ou_process = ou_process or OuMidpriceModel(initial_price=0.0)
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array(
                [
                    [
                        initial_price - (self._get_max_asset_price(initial_price, terminal_time) - initial_price),
                        self.ou_process.min_value,
                    ]
                ]
            ),
            max_value=np.array([[self._get_max_asset_price(initial_price, terminal_time), self.ou_process.max_value]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, self.ou_process.initial_state[0][0]]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.ou_process.current_state * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )
        self.ou_process.update(arrivals, fills, actions)
        self.current_state[:, 1] = self.ou_process.current_state

    def _get_max_asset_price(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: what should this be?


class BrownianMotionJumpMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 2.0,
        jump_size: float = 1.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.jump_size = jump_size
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        fills_bid = fills[:, BID_INDEX] * arrivals[:, BID_INDEX]
        fills_ask = fills[:, ASK_INDEX] * arrivals[:, ASK_INDEX]
        self.current_state = (
            self.current_state
            + self.drift * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
            + (self.jump_size * fills_ask - self.jump_size * fills_bid).reshape(-1,1)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time


class OuJumpMidpriceModel(MidpriceModel):
    def __init__(
        self,
        mean_reversion_level: float = 0.0,
        mean_reversion_speed: float = 1.0,
        volatility: float = 2.0,
        jump_size: float = 1.0,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.mean_reversion_level = mean_reversion_level
        self.mean_reversion_speed = mean_reversion_speed
        self.volatility = volatility
        self.jump_size = jump_size
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        fills_bid = fills[:, BID_INDEX] * arrivals[:, BID_INDEX]
        fills_ask = fills[:, ASK_INDEX] * arrivals[:, ASK_INDEX]
        self.current_state = (
            self.current_state
            - self.mean_reversion_speed
            * (self.current_state - self.mean_reversion_level * np.ones((self.num_trajectories, 1)))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))            
            + (self.jump_size * fills_ask - self.jump_size * fills_bid).reshape(-1,1)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time


class ShortTermJumpAlphaMidpriceModel(MidpriceModel):
    def __init__(
        self,
        volatility: float = 2.0,
        ou_jump_process: OuJumpMidpriceModel = None,
        initial_price: float = 100.0,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.volatility = volatility
        self.ou_jump_process = ou_jump_process or OuJumpMidpriceModel(initial_price=0.0)
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array(
                [
                    [
                        initial_price - (self._get_max_asset_price(initial_price, terminal_time) - initial_price),
                        self.ou_jump_process.min_value,
                    ]
                ]
            ),
            max_value=np.array(
                [[self._get_max_asset_price(initial_price, terminal_time), self.ou_jump_process.max_value]]
            ),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, self.ou_jump_process.initial_state[0][0]]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.ou_jump_process.current_state * self.step_size * np.ones((self.num_trajectories, 1))
            + self.volatility * sqrt(self.step_size) * self.rng.normal(size=(self.num_trajectories, 1))
        )
        self.ou_jump_process.update(arrivals, fills, actions)
        self.current_state[:, 1] = self.ou_jump_process.current_state

    def _get_max_asset_price(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time  # TODO: what should this be?


class HestonMidpriceModel(MidpriceModel):
    # Current/Initial State with the Heston model will consist of price AND current variance, not just price
    def __init__(
        self,
        drift: float = 0.05,
        volatility_mean_reversion_rate: float = 3,
        volatility_mean_reversion_level: float = 0.04,
        weiner_correlation: float = -0.8,
        volatility_of_volatility: float = 0.6,
        initial_price: float = 100,
        initial_variance: float = 0.2**2,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility_mean_reversion_rate = volatility_mean_reversion_rate
        self.terminal_time = terminal_time
        self.weiner_correlation = weiner_correlation
        self.volatility_mean_reversion_level = volatility_mean_reversion_level
        self.volatility_of_volatility = volatility_of_volatility
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price, initial_variance]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        weiner_means = np.array([0, 0])
        weiner_corr = np.array([[1, self.weiner_correlation], [self.weiner_correlation, 1]])
        weiners = np.random.multivariate_normal(weiner_means, cov=weiner_corr, size=self.num_trajectories)
        self.current_state[:, 0] = (
            self.current_state[:, 0]
            + self.drift * self.current_state[:, 0] * self.step_size
            + np.sqrt(self.current_state[:, 1] * self.step_size) * self.current_state[:, 0] * weiners[:, 0]
        )
        self.current_state[:, 1] = np.abs(
            self.current_state[:, 1]
            + self.volatility_mean_reversion_rate
            * (self.volatility_mean_reversion_level - self.current_state[:, 1])
            * self.step_size
            + self.volatility_of_volatility * np.sqrt(self.current_state[:, 1] * self.step_size) * weiners[:, 1]
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility_mean_reversion_level * terminal_time


class ConstantElasticityOfVarianceMidpriceModel(MidpriceModel):
    def __init__(
        self,
        drift: float = 0.0,
        volatility: float = 0.1,
        gamma: float = 1,  # gamma = 1 is just gbm
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        seed: Optional[int] = None,
    ):
        self.drift = drift
        self.volatility = volatility
        self.gamma = gamma
        self.terminal_time = terminal_time
        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.current_state
            + self.current_state * self.drift * self.step_size  # *np.ones((self.num_trajectories, 1))
            + self.volatility
            * (self.current_state**self.gamma)
            * np.sqrt(self.step_size)
            * np.random.normal(size=self.num_trajectories)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price + 4 * self.volatility * terminal_time



class AmmSelfContainedMidpriceModel(MidpriceModel):
    def __init__(
        self,
        jump_size_L: float = 1.0,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        unit_size: float = 1,
        eta_bid: callable = lambda y, Delta, L: Delta / (L  * (y - Delta)),
        eta_ask: callable = lambda y, Delta, L: Delta / (L  * (y + Delta)),
        seed: Optional[int] = None,
    ):
        self.jump_size_L = jump_size_L
        self.terminal_time = terminal_time
        self.unit_size = unit_size
        self.eta_ask = eta_ask
        self.eta_bid = eta_bid

        if eta_bid is None: self.eta_bid = lambda y, Delta, L: Delta / (L  * (y - Delta))
        if eta_ask is None: self.eta_ask = lambda y, Delta, L: Delta / (L  * (y + Delta))

        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        fills_bid = fills[:, BID_INDEX] * arrivals[:, BID_INDEX]
        fills_ask = fills[:, ASK_INDEX] * arrivals[:, ASK_INDEX]
        previous_inventory = state[:, INVENTORY_INDEX] - fills_bid + fills_ask
        asketa = self.eta_ask(previous_inventory, self.unit_size, self.jump_size_L)
        bideta = self.eta_bid(previous_inventory, self.unit_size, self.jump_size_L)
        
        self.current_state = (
            self.current_state
            +  (fills_ask * asketa -  fills_bid * bideta).reshape(-1,1)
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price * 2 





class AmmSelfContainedMidGeopriceModel(MidpriceModel):
    def __init__(
        self,
        initial_price: float = 100,
        terminal_time: float = 1.0,
        step_size: float = 0.01,
        num_trajectories: int = 1,
        unit_size: float = 0.01/100, # in % 
        eta_plus: callable = None,
        eta_minus: callable = None,
        seed: Optional[int] = None,
    ):
        self.terminal_time = terminal_time
        self.unit_size = unit_size

        super().__init__(
            min_value=np.array([[initial_price - (self._get_max_value(initial_price, terminal_time) - initial_price)]]),
            max_value=np.array([[self._get_max_value(initial_price, terminal_time)]]),
            step_size=step_size,
            terminal_time=terminal_time,
            initial_state=np.array([[initial_price]]),
            num_trajectories=num_trajectories,
            seed=seed,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        fills_bid = fills[:, BID_INDEX] * arrivals[:, BID_INDEX]
        fills_ask = fills[:, ASK_INDEX] * arrivals[:, ASK_INDEX]

        eta_ask = 1 - 1/(self.unit_size+1)
        eta_bid = 1/(1-self.unit_size) - 1

        self.current_state = (
            self.current_state* (1+(fills_ask * eta_ask -  fills_bid * eta_bid).reshape(-1,1))
        )

    def _get_max_value(self, initial_price, terminal_time):
        return initial_price * 2 




class MarketDataReplayModel(MidpriceModel):
    def __init__(
        self,
        historical_data: np.array = None,
        num_trajectories: int = 1,
    ):
        self.historical_data = historical_data # this should be of shape (nb_data, num_trajectories)
        self.current_index  = 0
        
        super().__init__(
            min_value=np.array([[-np.inf]]),
            max_value=np.array([[np.inf]]),
            step_size=0.01,
            terminal_time=1.0,
            initial_state=np.array([[self.historical_data[0]]]),
            num_trajectories=num_trajectories,
            seed=None,
        )

    def update(self, arrivals: np.ndarray, fills: np.ndarray, actions: np.ndarray, state: np.ndarray = None) -> np.ndarray:
        self.current_state = (
            self.historical_data[self.current_index]
        )
        self.current_index += 1 # should not go over np.shape(self.historical_data)[0]

    def _get_max_value(self, initial_price, terminal_time):
        return np.max(self.historical_data)
