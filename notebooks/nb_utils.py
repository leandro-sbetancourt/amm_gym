import sys
sys.path.append("../") # This version of the notebook is in the subfolder "notebooks" of the repo

#import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime

from copy import deepcopy

from mbt_gym.gym.TradingEnvironment import TradingEnvironment, ASSET_PRICE_INDEX, INVENTORY_INDEX
from mbt_gym.gym.Traders import AmmTrader, MarketOrderTrader
from mbt_gym.agents.BaselineAgents import *
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
from mbt_gym.gym.helpers.plotting import *
from mbt_gym.agents.BaselineAgents import CarteaJaimungalMmAgent
from mbt_gym.gym.helpers.generate_trajectory import generate_trajectory
from mbt_gym.gym.StableBaselinesTradingEnvironment import StableBaselinesTradingEnvironment
from mbt_gym.gym.wrappers import *
from mbt_gym.rewards.RewardFunctions import CjCriterion, CjMmCriterion, RunningTargetInventoryPenalty
from mbt_gym.stochastic_processes.midprice_models import *
from mbt_gym.stochastic_processes.midprice_models import MarketDataReplayModel
from mbt_gym.stochastic_processes.fill_probability_models import *
from mbt_gym.stochastic_processes.arrival_models import *

import matplotlib.ticker as mtick
import matplotlib.dates as md

def rescale_plot(W=5, l=6, w=3, fontsize=10):
    plt.rcParams.update({
            'figure.figsize': (W, W/(l/w)),     # 6:3 aspect ratio
            'font.size' : fontsize,                   # Set font size to 11pt
            'axes.labelsize': fontsize,               # -> axis labels
            'legend.fontsize': fontsize,              # -> legends
            'font.family': 'lmodern',
            'text.usetex': True,
            'text.latex.preamble':             # LaTeX preamble
                r"\usepackage{amssymb}\usepackage{lmodern}\usepackage{amsfonts}\usepackage{amsmath}"
                # ... more packages if needed
            
        })


def run_simulation(arb_env, arb_agent, pool_agent, num_trajectories, n_steps):
    obs_space_dim    = arb_env.observation_space.shape[0]
    action_space_dim = arb_env.action_space.shape[0]
    observations     = np.zeros((arb_env.num_trajectories, obs_space_dim, arb_env.n_steps + 1))
    actions          = np.zeros((arb_env.num_trajectories, action_space_dim, arb_env.n_steps))
    rewards          = np.zeros((arb_env.num_trajectories, 1, arb_env.n_steps))
    
    obs                   = arb_env.reset()
    observations[:, :, 0] = obs
    count                 = 0

    for _i in range(n_steps):
        action                        = arb_agent.get_action(obs)
        obs,  reward, done, _         = arb_env.step(action)

        actions[:, :, count]          = action
        observations[:, :, count + 1] = obs
        rewards[:, :, count]          = reward.reshape(-1, 1)

        if (pool_agent.env.num_trajectories > 1 and done[0]) or (pool_agent.env.num_trajectories == 1 and done):
            break

        count += 1
    
    return observations

def getSimulationData(arb_agent, observations, initial_pool_value, initial_inventory_pool):
    Z          = [s[1][0][0] for s in arb_agent.historical_pool_prices]
    timesIndex = [s[0] for s in arb_agent.historical_pool_prices]
    bids       = [s[1][0][0] for s in arb_agent.historical_ba]
    asks       = [s[1][0][1] for s in arb_agent.historical_ba]
    historical_pool_inventory      = [s[1][0] for s in arb_agent.historical_pool_inventory]
    pool_earnings_history          = [s[1][0] if type(s[1])==np.ndarray else s[1]  for s in arb_agent.pool_earnings_history]
    historical_pool_inventory_cash = [0] + [-qtty * Zi for (qtty, Zi) in zip(np.diff(historical_pool_inventory), Z[1:])]
    
    allData         = pd.DataFrame(index = timesIndex)
    allData['Z']    = Z
    allData['bids'] = bids
    allData['asks'] = asks
    allData['pool_inv_y']    = historical_pool_inventory
    allData['pool_inv_x']    = np.cumsum(historical_pool_inventory_cash)
    allData['pool_earnings'] = pool_earnings_history
    
    max_risk  =  allData.pool_inv_y.abs().max() # This is used for loss / pnl calculations
    
    pool_earnings_history = allData[["pool_earnings"]].groupby(allData.index).sum()
    bids_history          = allData[["bids"]].groupby(allData.index).sum()
    asks_history          = allData[["asks"]].groupby(allData.index).sum()

    allData = allData.groupby(allData.index).first()
    allData['S'] = observations[0, ASSET_PRICE_INDEX, :-1]
    allData['pool_earnings'] = pool_earnings_history
    allData['pool_earnings'] = allData['pool_earnings'].cumsum()
    
    ImpermanentLoss = allData.pool_inv_x+allData.pool_inv_y*allData.Z - initial_inventory_pool*allData.Z
    allData['ImpermanentLoss'] = ImpermanentLoss
    
    allData.loc[allData[allData.bids>9000].index, "bids"] = np.nan
    allData.loc[allData[allData.asks>9000].index, "asks"] = np.nan
    
    bid_history = (allData['Z'] - allData['bids'])
    ask_history = (allData['Z'] + allData['asks'])
    
    allData['poolValue'] = allData.pool_inv_y*allData.S + allData.pool_inv_x
    allData['poolValue'] -= allData['poolValue'].iloc[0]
    allData['Buy and Hold']      = max_risk*allData.Z - max_risk*allData.Z.iloc[0]
    allData['Buy and Hold']      = allData.pool_inv_y*allData.S - (allData.pool_inv_y*allData.S).iloc[0]
    allData['Buy and Hold']      = (allData.pool_inv_y * allData.Z.diff(1).shift(-1)).cumsum().fillna(method='ffill')

    allData['LPWealth']  = allData.pool_earnings + (allData.pool_inv_y*allData.Z+ allData.pool_inv_x)
    
    return allData, bid_history, ask_history

def getSimulationData2(initial_pool_value, initial_inventory_pool,
                      historical_pool_prices,  historical_ba, 
                      historical_pool_inventory, pool_earnings_history, historical_oracle_prices):
    
    Z          = [s[1] for s in historical_pool_prices]
    timesIndex = [s[0] for s in historical_pool_prices]
    bids       = [s[1][0] for s in historical_ba]
    asks       = [s[1][1] for s in historical_ba]
    
    historical_pool_inventory      = [s[1] for s in historical_pool_inventory]
    pool_earnings_history          = [s[1] if type(s[1])==np.ndarray else s[1]  for s in pool_earnings_history]
    historical_pool_inventory_cash = [0] + [-qtty * Zi for (qtty, Zi) in zip(np.diff(historical_pool_inventory), Z[1:])]
    
    allData         = pd.DataFrame(index = timesIndex)
    allData['Z']    = Z
    allData['bids'] = bids
    allData['asks'] = asks
    allData['pool_inv_y']    = historical_pool_inventory
    allData['pool_inv_x']    = np.cumsum(historical_pool_inventory_cash)
    allData['pool_earnings'] = pool_earnings_history
    allData['S'] = [s[1] for s in historical_oracle_prices]
    
    max_risk  =  allData.pool_inv_y.abs().max() # This is used for loss / pnl calculations
    
    pool_earnings_history = allData[["pool_earnings"]].groupby(allData.index).sum()
    bids_history          = allData[["bids"]].groupby(allData.index).sum()
    asks_history          = allData[["asks"]].groupby(allData.index).sum()
    
    allData = allData.groupby(allData.index).first()
    allData['pool_earnings'] = pool_earnings_history
    allData['pool_earnings'] = allData['pool_earnings'].cumsum()
    
    ImpermanentLoss = allData.pool_inv_x+allData.pool_inv_y*allData.Z - initial_inventory_pool*allData.Z
    allData['ImpermanentLoss'] = ImpermanentLoss
    
    allData.loc[allData[allData.bids>9000].index, "bids"] = np.nan
    allData.loc[allData[allData.asks>9000].index, "asks"] = np.nan
    
    bid_history = (allData['Z'] - allData['bids'])
    ask_history = (allData['Z'] + allData['asks'])
    
    allData['poolValue'] = allData.pool_inv_y*allData.S + allData.pool_inv_x
    allData['poolValue'] -= allData['poolValue'].iloc[0]
    allData['Buy and Hold']      = max_risk*allData.Z - max_risk*allData.Z.iloc[0]
    allData['Buy and Hold']      = allData.pool_inv_y*allData.S - (allData.pool_inv_y*allData.S).iloc[0]
    allData['Buy and Hold']      = (allData.pool_inv_y * allData.Z.diff(1).shift(-1)).cumsum().fillna(method='ffill')

    allData['LPWealth']  = allData.pool_earnings + (allData.pool_inv_y*allData.Z+ allData.pool_inv_x)
    
    return allData, bid_history, ask_history

def get_pool_agent(_arrival_rate, _phi, _alpha, _fill_exponent, 
                   _initial_inventory_pool, _target_inventory, 
                   _jump_size_L, _unit_size, _min_inventory_pool, _max_inventory_pool,
                   _initial_price,  _max_depth,
                   _terminal_time,  _step_size,
                   _seed, _n_steps, _num_trajectories,
                   _eta_func_bid, _eta_func_ask, _verbose):
    
    
    _midprice_model_internal = AmmSelfContainedMidpriceModel(jump_size_L      = _jump_size_L, 
                                                             unit_size        = _unit_size,
                                                             terminal_time    = _terminal_time, 
                                                             step_size        = _step_size, 
                                                             initial_price    = _initial_price, 
                                                             num_trajectories = _num_trajectories, 
                                                             seed             = _seed,
                                                             eta_ask          = _eta_func_ask,
                                                             eta_bid          = _eta_func_bid)
    
    _arrival_model = PoissonArrivalModel(intensity = np.array([_arrival_rate, _arrival_rate]), 
                                         step_size = _step_size, 
                                         seed      = _seed)
    
    _fill_probability_model = ExponentialFillFunction(fill_exponent    = _fill_exponent, 
                                                     step_size        = _step_size, 
                                                     num_trajectories = _num_trajectories, 
                                                     seed             = _seed)

    _AMMtrader = AmmTrader(midprice_model        = _midprice_model_internal, 
                          arrival_model          = _arrival_model, 
                          fill_probability_model = _fill_probability_model,
                          num_trajectories       = _num_trajectories, 
                          max_depth              = _max_depth, 
                          unit_size              = _unit_size,
                          seed                   = _seed)

    _env_params_pool = dict(  terminal_time     = _terminal_time, 
                              n_steps           = _n_steps,
                              initial_inventory = _initial_inventory_pool,
                              midprice_model    = _midprice_model_internal,
                              arrival_model     = _arrival_model,
                              fill_probability_model = _fill_probability_model,
                              trader            = _AMMtrader,
                              reward_function   = RunningTargetInventoryPenalty(_phi, _alpha, 2, _initial_inventory_pool),
                              normalise_action_space      = False,
                              normalise_observation_space = False,
                              num_trajectories  = _num_trajectories, 
                              min_inventory     = _min_inventory_pool,
                              max_inventory     = _max_inventory_pool,
                              max_cash          = 1e15,
                              seed              = _seed)

    return PoolInterpGEtaMmAgent(env               = TradingEnvironment(**_env_params_pool), 
                                   target_inventory  = _target_inventory, 
                                   min_inventory     = _min_inventory_pool,
                                   max_inventory     = _max_inventory_pool,
                                   verbose           = _verbose)


def get_arb_env(pool_agent, historical_data, num_trajectories, seed, terminal_time, n_steps, initial_inventory_arb,
                   phi, alpha):
    initial_inventory_arb   = 0
    midprice_model_external = MarketDataReplayModel(historical_data    = historical_data,
                                                    num_trajectories = num_trajectories)

    ArbTrader = MarketOrderTrader(num_trajectories  = 1,
                                  min_size          = -1e15,
                                  max_size          = 1e15,
                                  seed              = seed)

    env_params_arb = dict(terminal_time      = terminal_time, 
                          n_steps            = n_steps,
                          initial_inventory  = initial_inventory_arb,
                          midprice_model     = midprice_model_external,
                          trader             = ArbTrader,
                          reward_function    = CjMmCriterion(phi, alpha),
                          normalise_action_space      = False,
                          normalise_observation_space = False,
                          num_trajectories   = num_trajectories,
                          min_inventory      = -1e15,
                          max_inventory      = 1e15,
                          max_cash           = 1e15,
                          seed               = seed)
    
    arb_env   = TradingEnvironment(**env_params_arb)
    arb_agent = ArbitrageurAmmAgent(env           = TradingEnvironment(**env_params_arb),
                                     agent         = pool_agent,
                                     min_inventory = -1e14,
                                     max_inventory = 1e14,
                                     verbose       = False)
    return arb_agent, arb_env

def get_LT_LP_Binance_data(LTdata, LPdata, BinanceData, trade_date, from_datetime, to_datetime):
    oneDayLPdata = LPdata[ ((LPdata.timestamp<=to_datetime) & (LPdata.timestamp>from_datetime))].set_index('timestamp')
    oneDayLTdata = LTdata[ ((LTdata.timestamp<=to_datetime) & (LTdata.timestamp>from_datetime))].set_index('timestamp')
    trade_date   = from_datetime.split(' ')[0]
    oneDaybinanceLTdata = BinanceData[ ((BinanceData.index<=to_datetime) &
                                            (BinanceData.index>from_datetime))]
        
    if ((len(oneDayLTdata)>10) and (len(oneDaybinanceLTdata)>10)):

        oneDayLTdata['executionPrice'] = -oneDayLTdata['amount0'].astype(float)/oneDayLTdata['amount1'].astype(float)

        oneDayLTdata['convexity'] = 2 * oneDayLTdata.executionPrice**1.5  / oneDayLTdata.kappa

        # preapre other auxiliary data
        fill_exponents  = (2/oneDayLTdata['convexity']).values
        pool_sizes      = (2 *  oneDayLTdata.kappa * oneDayLTdata.executionPrice ** 0.5).values # This is in USDC
        hist_prices     = oneDaybinanceLTdata.price.values

        initial_convexity = oneDayLTdata['convexity'].iloc[0]
        #trade_size_model  = (pool_sizes[0] * 0.2) / hist_prices[0] / 5000 # matrix of size 1000
        trade_sizes       = oneDayLTdata.amount1.abs()#/trade_size_model

        bothPrices    = pd.concat((oneDaybinanceLTdata.price.reset_index().groupby('time').last().sort_index(), 
                               oneDayLTdata.poolPricePrev.reset_index().groupby('timestamp').last().sort_index()), axis=1
                              ).fillna(method='ffill').fillna(method='bfill').loc[oneDaybinanceLTdata.index, :]


        return oneDayLTdata, oneDayLPdata, oneDaybinanceLTdata,\
                fill_exponents, pool_sizes, hist_prices,\
                initial_convexity, trade_sizes, bothPrices
    else:
        return oneDayLTdata, oneDayLPdata, oneDaybinanceLTdata,\
                None, None, None,\
                None, None, None

def get_binance_month_data(s_month):
    oneDaybinanceLTdata = pd.read_csv(f"data/ETHUSDC-trades-{s_month}.csv", index_col=0, header=None)
    oneDaybinanceLTdata.columns = ['price','qty','quoteQty','time','isBuyerMaker','isBestMatch']
    oneDaybinanceLTdata.index.name = 'trade Id'
    
    oneDaybinanceLTdata['time'] = [datetime.datetime.fromtimestamp(i/1000) for i in oneDaybinanceLTdata.time.values]
    oneDaybinanceLTdata['time'] = oneDaybinanceLTdata['time'] #+ pd.to_timedelta(1, unit='h')
    oneDaybinanceLTdata = oneDaybinanceLTdata.set_index('time')
    return oneDaybinanceLTdata






def runOneSimulation(pool_agent, bothPrices, unit_size, min_inventory_pool, max_inventory_pool, target_inventory,
                     eta_func_bid, eta_func_ask, jump_size_L, add_noise = None):
    verbose = False
    
    # containers for historical out of sample data
    pool_earnings_history     = []
    arb_earnings_history      = []
    historical_pool_prices    = []
    historical_oracle_prices  = []
    historical_ba             = []
    historical_pool_inventory = []
    historical_pool_cash      = []
    historical_pool_value_adjustments = [] # this accounts for pnls when the LP resets her position

    # set the terminal time in days
    terminal_time = (bothPrices.index[-1] - bothPrices.index[0]).total_seconds()/60/60/24

    # create the agent for the out of sample period
    current_inventory = 0
    current_cash      = 0
    price_Z           = bothPrices.iloc[0].Binance
    price_S           = bothPrices.iloc[0].Binance

    for i_time in range(len(bothPrices.index)):
        _time                 = bothPrices.index[i_time]
        opportunity_arbitrage = True
        direction_oaa         = 0
        size_arbitrage        = 0
        
        while opportunity_arbitrage:
            indicator_buy  = (current_inventory + unit_size < max_inventory_pool)
            indicator_sell = (current_inventory - unit_size > min_inventory_pool)

            # get current information
            price_S = bothPrices.iloc[i_time].Binance # binance

            # if one hits max or min inventory, stop / recalibrate parameters
            if ((not indicator_buy) or (not indicator_sell)):

                # here the LP liquidates her inventory and keeps the cash as additional pnl
                additional_pnl = current_inventory * price_S + current_cash
                if verbose:  print('I hit the min/max inventory, additional PnL =', additional_pnl)

                # reset the inventory
                current_inventory = 0
                current_cash      = 0
            else:
                additional_pnl    = 0

            # get the LP's quotes for unit_size
            deltas = np.round(pool_agent._calculate_deltas(current_time = 0., 
                                                           inventories  = current_inventory), 3)[0]
            delta_bid, delta_ask = deltas[0], deltas[1]

            # verify if there is an arbitrage opportunity
            action = int((price_S>price_Z + delta_ask)*indicator_buy) - int((price_S<(price_Z - delta_bid))*indicator_sell)
            opportunity_arbitrage = (action != 0)
            if direction_oaa == 0: 
                direction_oaa = action #store the direction of the oaa
            else:
                # if the direction is opposite the previous one, stop to avoid going back and forth because of a too large L
                if direction_oaa != action: opportunity_arbitrage = False

            # Compute the spread between S and Z+-bid/ask spread
            spreadPrices = float((price_S - price_Z - delta_ask)*(price_S>price_Z + delta_ask)) + float((- price_S + price_Z - delta_bid)*(price_S<(price_Z - delta_bid)))

            # store the LP's inventory and the oracle price
            historical_pool_inventory += [(_time, current_inventory)]
            historical_pool_cash      += [(_time, current_cash)]
            historical_oracle_prices  += [(_time, price_S)]

            if opportunity_arbitrage:
                if verbose:
                    print('\n*****************')
                    print('Current time:', _time)
                    print('S = ', price_S, ', Z=', price_Z, ' deltas = ', deltas, '   *** action = ', action)
                    print('Spread to target', spreadPrices)
                    print('Inventory of LP:', current_inventory, 'min inventory= ', min_inventory_pool, ', max inventory:', max_inventory_pool, ". target inventory:", target_inventory)
                    print('Order to send by the LT:', action * unit_size)
                
                size_arbitrage    += np.abs(action * unit_size)
                current_inventory -= action * unit_size
                current_cash      += price_Z * unit_size * action # separate the earnings from the cash

                # if arbitrage, update the inventory and impact the rate Z 
                if action == 1: # the LT buys
                    price_Z               += eta_func_bid(current_inventory, unit_size, jump_size_L)
                    pool_earnings_history += [ (_time, delta_ask*unit_size )]

                else:
                    price_Z               -= eta_func_ask(current_inventory, unit_size, jump_size_L)
                    pool_earnings_history += [ (_time, delta_bid*unit_size)]

            else:
                pool_earnings_history += [(_time, 0 )]
                arb_earnings_history  += [(_time, 0)]

            historical_pool_value_adjustments +=  [ (_time, additional_pnl )]
            historical_pool_prices += [(_time, price_Z)]
            historical_ba          += [(_time, deltas)]

            if verbose: print('*****************')
        
        # if there has been arbitrage and we want noise trading, add here
        if add_noise is not None:
            nb_noise_trades = int(add_noise*size_arbitrage/unit_size)
            earnings_from_noise = 0
            if verbose: print(f'I add {nb_noise_trades} noise trades')
            
            for i_noise_trade in range(nb_noise_trades):
                buysell = np.random.choice((1, -1))
                
                indicator_buy  = (current_inventory + unit_size < max_inventory_pool)
                indicator_sell = (current_inventory - unit_size > min_inventory_pool)

                # if one hits max or min inventory, stop / recalibrate parameters
                if ((not indicator_buy) or (not indicator_sell)):
                    # here the LP liquidates her inventory and keeps the cash as additional pnl
                    additional_pnl = current_inventory * price_S + current_cash
                    if verbose:  print('I hit the min/max inventory, additional PnL =', additional_pnl)
                    # reset the inventory
                    current_inventory = 0
                    current_cash      = 0
                else:
                    additional_pnl    = 0
                
                # get the LP's quotes for unit_size
                deltas = np.round(pool_agent._calculate_deltas(current_time = 0., 
                                                               inventories  = current_inventory), 3)[0]
                delta_bid, delta_ask = deltas[0], deltas[1]
                
                # store the LP's inventory and the oracle price
                historical_pool_inventory += [(_time, current_inventory)]
                historical_pool_cash      += [(_time, current_cash)]
                historical_oracle_prices  += [(_time, price_S)]
                
                # update inv
                current_inventory -= buysell * unit_size
                current_cash      += price_Z * unit_size * buysell # separate the earnings from the cash
                
                # uodate price
                if buysell == 1: # the noisy LT buys
                    price_Z               += eta_func_bid(current_inventory, unit_size, jump_size_L)
                    pool_earnings_history += [ (_time, delta_ask*unit_size )]
                    earnings_from_noise += delta_ask*unit_size
                else:
                    price_Z               -= eta_func_ask(current_inventory, unit_size, jump_size_L)
                    pool_earnings_history += [ (_time, delta_bid*unit_size)]
                    earnings_from_noise += delta_bid*unit_size
                
                # update historical values
                historical_pool_value_adjustments +=  [(_time, additional_pnl )]
                historical_pool_prices += [(_time, price_Z)]
                historical_ba          += [(_time, deltas)]
            #if verbose: 
            #print('earnings_from_noise=', earnings_from_noise)
    return pool_earnings_history, arb_earnings_history, historical_pool_prices, historical_oracle_prices,\
            historical_ba, historical_pool_inventory, historical_pool_cash, historical_pool_value_adjustments


def getOneSimulationData(initial_pool_value, initial_inventory_pool,
                      historical_pool_prices,  historical_ba, 
                      historical_pool_inventory, pool_earnings_history, 
                      historical_oracle_prices, historical_pool_cash,
                      historical_pool_value_adjustments):
    
    Z          = [s[1] for s in historical_pool_prices]
    timesIndex = [s[0] for s in historical_pool_prices]
    bids       = [s[1][0] for s in historical_ba]
    asks       = [s[1][1] for s in historical_ba]
    
    historical_pool_inventory      = [s[1] for s in historical_pool_inventory]
    pool_earnings_history          = [s[1] if type(s[1])==np.ndarray else s[1]  for s in pool_earnings_history]
    historical_pool_cash           = [s[1] for s in historical_pool_cash]
    #historical_pool_inventory_cash = [0] + [-qtty * Zi for (qtty, Zi) in zip(np.diff(historical_pool_inventory), Z[1:])]
    historical_pool_value_adjustments = [s[1] for s in historical_pool_value_adjustments]
    
    allData         = pd.DataFrame(index = timesIndex)
    allData['Z']    = Z
    allData['bids'] = bids
    allData['asks'] = asks
    allData['pool_inv_y']    = historical_pool_inventory
    allData['pool_inv_x']    = historical_pool_cash #np.cumsum(historical_pool_inventory_cash)
    allData['pool_earnings'] = pool_earnings_history
    allData['pool_adjustments'] = historical_pool_value_adjustments
    
    allData['S'] = [s[1] for s in historical_oracle_prices]
    
    max_risk  =  allData.pool_inv_y.abs().max() # This is used for loss / pnl calculations
    
    pool_earnings_history = allData[["pool_earnings"]].groupby(allData.index).sum()
    pool_asjustment_history = allData[["pool_adjustments"]].groupby(allData.index).sum()
    bids_history          = allData[["bids"]].groupby(allData.index).sum()
    asks_history          = allData[["asks"]].groupby(allData.index).sum()
    
    allData = allData.groupby(allData.index).first()
    allData['pool_earnings'] = pool_earnings_history
    allData['pool_adjustments'] = pool_asjustment_history
    allData['pool_earnings'] = allData['pool_earnings'].cumsum()
    allData['pool_adjustments'] = allData['pool_adjustments'].cumsum()
    
    ImpermanentLoss = allData.pool_inv_x+allData.pool_inv_y*allData.Z - initial_inventory_pool*allData.Z
    allData['ImpermanentLoss'] = ImpermanentLoss
    
    allData.loc[allData[allData.bids>9000].index, "bids"] = np.nan
    allData.loc[allData[allData.asks>9000].index, "asks"] = np.nan
    
    bid_history = (allData['Z'] - allData['bids'])
    ask_history = (allData['Z'] + allData['asks'])
    
    allData['poolValue'] = allData.pool_inv_y*allData.S + allData.pool_inv_x + allData.pool_adjustments 
    allData['poolValue'] -= allData['poolValue'].iloc[0]
    allData['Buy and Hold']      = max_risk*allData.S - max_risk*allData.S.iloc[0]
    allData['LPWealth']  = allData.pool_earnings + allData.poolValue 
    
    return allData, bid_history, ask_history


def plot_impact_curves(pool_agent, maxTradeSize, unit_size, jump_size_L, initial_convexity, initial_inventory_pool):
    fig, axes = plt.subplots(1, 2, sharex = False, constrained_layout = True)

    trade_size    = np.arange(unit_size, maxTradeSize, unit_size)
    trade_size2   = np.arange(unit_size, maxTradeSize, unit_size)

    impact_curve_buy  = np.array([- unit_size * jump_size_L / ( (initial_inventory_pool - i*unit_size) - unit_size) for i in range(len(trade_size))])
    impact_curve_sell = np.array([unit_size * jump_size_L / ( (initial_inventory_pool + i*unit_size) + unit_size) for i in range(len(trade_size))])

    impact_curve_buy  = np.cumsum(impact_curve_buy)
    impact_curve_sell = np.cumsum(impact_curve_sell)
    impact_curve_buy_uniswap  = np.array([initial_convexity*i*unit_size for i in range(len(trade_size2))])

    axes[0].plot(trade_size, impact_curve_buy, color='k', lw=3)
    axes[0].plot(trade_size, impact_curve_buy_uniswap, color='b', lw=3)

    ##################
    # exec cost curve
    ##################
    trade_size   = np.arange(-maxTradeSize, maxTradeSize, unit_size)
    trade_size2  = np.arange(0,    maxTradeSize, unit_size)

    exec_cost_sell  =  np.array([np.round(pool_agent._calculate_deltas(current_time = 0., 
                                                            inventories  = initial_inventory_pool+i*unit_size
                                                           ), 3)[0][0]
                                         for i in range(len(trade_size2))])
    
    exec_cost_buy   =  np.array([np.round(pool_agent._calculate_deltas(current_time = 0., 
                                                            inventories  = initial_inventory_pool-i*unit_size
                                                           ), 3)[0][1]
                                                 for i in range(len(trade_size2))])

    exec_cost_sell_uniswap      =  np.array([0.5*initial_convexity*i*unit_size for i in range(len(trade_size2))])
    exec_cost_buy_uniswap       =  np.array([0.5*initial_convexity*i*unit_size for i in range(len(trade_size2))])


    axes[1].plot(trade_size, np.concatenate((np.cumsum(exec_cost_sell)[::-1],
                                             np.cumsum(exec_cost_buy))), color='k', lw=3)
    axes[1].plot(trade_size, np.concatenate((np.cumsum(exec_cost_sell_uniswap)[::-1],
                                             np.cumsum(exec_cost_buy_uniswap))), color='b', lw=3)

    axes[1].set_title('Execution costs')
    axes[1].set_xlabel('Trade size in ETH')
    axes[0].legend(['CQV',  'Uniswap'], fancybox=True, framealpha=0.2, handlelength=0.4, ncol=1, loc='best')
    axes[1].legend(['CQV', 'Uniswap'], fancybox=True, framealpha=0.2, handlelength=0.4, ncol=1, loc='best')

    axes[0].grid(); axes[1].grid()

    axes[0].set_title('Marginal rate impact')
    axes[0].set_xlabel('Trade size in ETH')
    axes[0].set_ylabel('USDC')
    
    return None


def plot_one_sim_result(_allData, bothPrices, uniswapindex, 
                        bid_history, ask_history):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, constrained_layout=True)

    ax1.plot(_allData.index, _allData['Z'], lw=2., color='k')
    ax1.plot(_allData.index, _allData['S'], lw=2., color='blue')
    ax1.plot(_allData.index, bothPrices.iloc[uniswapindex].Uniswap, lw=2., color='gray')

    ax1.fill_between(_allData.index, bid_history.values, ask_history.values, alpha=0.4, 
                     facecolor='tan', hatch="ooo", edgecolor="grey")

    ax1.plot(_allData[_allData['pool_inv_y'].diff(1).shift(-1)>0].index ,
             _allData[_allData['pool_inv_y'].diff(1).shift(-1)>0].Z+0.005, 
             marker = 'v', linestyle = 'None', 
             markerfacecolor='None', markeredgecolor='r')

    ax1.plot(_allData[_allData['pool_inv_y'].diff(1).shift(-1)<0].index ,
             _allData[_allData['pool_inv_y'].diff(1).shift(-1)<0].Z-0.005, 
             marker = '^', linestyle = 'None', 
             markerfacecolor='None', markeredgecolor='green')

    ax1.legend(['CQV',  'Binance', "Uniswap",  r'$[Z_t - \delta_t^b, Z_t + \delta_t^a]$'],  #, 'Sell', 'Buy'
               fancybox=True, framealpha=0.2, handlelength=0.4, ncol=1, loc='lower right' )

    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('\${x:,.2f}'))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')

    for ax in (ax1, ): 
        ax.grid('both')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("left")
        ax.set_axisbelow(True)

    ax2.plot(_allData.index , _allData['poolValue'], color='k', lw=3)
    ax2.plot(_allData.index , _allData['Buy and Hold'], color='blue', lw=3)
    ax2.plot(_allData.index , _allData['pool_earnings'], color='gray', lw=3)
    ax2.plot(_allData.index , _allData['LPWealth'], color='lightcoral', lw=3)

    #ax2.fill_between(_allData.index, 
    #                 _allData['Buy and Hold'],
    #                 _allData['poolValue'], 
    #                 alpha=0.4, facecolor='lightcoral', hatch="ooo", edgecolor="grey")

    ax2.legend(['Pool value', 'Buy and Hold', 'Earnings', 'LP total wealth'], 
               fancybox=True, framealpha=0.2, handlelength=0.5, ncol=1, loc='center right')

    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter('\${x:,.0f}'))
    ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter('\${x:,.0f}'))
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Performance')

    for ax in (ax2, ): 
        ax.grid('both')
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("left")
        ax.set_axisbelow(True)

    ax1.set_xticks(ax1.get_xticks()[::2], rotation=-45, ha='right')
    ax1.xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
