import datetime
import numpy as np
import pandas as pd
import time
import sys

from simtools import log_message


# Lee-Ready tick strategy simulator

# Record a trade in our trade array
def record_trade(trade_df, idx, trade_px, trade_qty, current_bar, trade_type, side):
    # print( "Trade! {} {} {} {}".format( idx, trade_px, trade_qty, current_bar ) )
    trade_df.loc[idx] = [trade_px, trade_qty, current_bar, trade_type, side]

    return


# TODO: calc P&L and other statistics
def trade_statistics( trade_df ):
    # TODO: calculate total P&L
    # TODO: calculate intraday P&L (time series). P&L has two components. Roughly:
    #       1. realized "round trip" P&L  sum of (sell price - buy price) * shares traded
    #       2. unrealized P&L of open position:  quantity held * (current price - avg price)
    # TODO: calculate maximum position (both long and short)
    # TODO: calculate worst and best intraday P&L
    # TODO: calculate total P&L
    P_L = 0.0
    realized_P_L = 0.0
    unrealized_P_L = 0.0
    max_long_position = 0
    worst_P_L = 0.0
    best_P_L = 0.0
    max_short_position = 0.0
    current_position = 0.0
    avg_price = 0.0
    result = {}
    history=pd.DataFrame( columns = [ 'P_L' , 'current position' ], index=trade_df.index )
    for index, row in trade_df.iterrows():
        if row.side == 'b':
            current_position -= row.shares
            unrealized_P_L += row.shares * row.price
        else:
            current_position += row.shares
            unrealized_P_L -= row.shares * row.price
        if current_position > max_long_position :
            max_long_position = current_position
        elif current_position < max_short_position :
            max_short_position = current_position
        if current_position==0:
            P_L=unrealized_P_L
        else:
            avg_price= abs(unrealized_P_L/current_position)
            P_L = current_position*(row.price-avg_price)
        if P_L>best_P_L:
            best_P_L=P_L
        elif P_L<worst_P_L:
            worst_P_L=P_L
        history.loc[index]=[P_L,current_position]
        #history['current position']=current_position
    
    result['P_L']=P_L
    result['best P_L']=best_P_L
    result['worst P_L']=worst_P_L
    result['max long position'] = max_long_position
    result['max short position'] = max_short_position
    result['current position'] = current_position
    result['hist']=history
    return result


# MAIN ALGO LOOP
def algo_loop(trading_day, last_day):
    
    log_message('Beginning Tick Strategy run')
    
    round_lot = 100

    #NBBO
    NBBO_last_day=last_day[(last_day.qu_source == 'N') & (last_day.natbbo_ind == 4)]
    
    avg_spread = (NBBO_last_day.ask_px - NBBO_last_day.bid_px).mean()
    half_spread = avg_spread / 2
    print("Average stock spread for sample: {:.4f}".format(avg_spread))

    # init our price and volume variables
    [last_price, last_size, bid_price, bid_size, ask_price, ask_size, volume] = np.zeros(7)

    # init our counters
    [trade_count, quote_count, cumulative_volume] = [0, 0, 0]

    # init some time series objects for collection of telemetry
    fair_values = pd.Series(index=trading_day.index)
    midpoints = pd.Series(index=trading_day.index)
    tick_factors = pd.Series(index=trading_day.index)

    # let's set up a container to hold trades. preinitialize with the index
    trades = pd.DataFrame(columns=['price', 'shares', 'bar', 'trade_type', 'side'], index=trading_day.index)

    # MAIN EVENT LOOP
    current_bar = 0

    # track state and values for a current working order
    live_order = False
    live_order_price = 0.0
    live_order_quantity = 0.0
    order_side = '-'

    # other order and market variables
    total_quantity_filled = 0
    vwap_numerator = 0.0

    total_trade_count = 0
    total_agg_count = 0
    total_pass_count = 0

    # fair value pricing variables
    midpoint = 0.0
    fair_value = 0.0

    # define our accumulator for the tick EMA
    message_type = 0
    tick_coef = 1
    tick_window = 20
    tick_factor = 0
    tick_ema_alpha = 2 / (tick_window + 1)
    prev_tick = 0
    prev_price = 0

    # risk factor for part 2
    # TODO: implement and evaluate
    risk_factor = 0.0
    risk_coef = -0.5

    n = 2
    new_midpoint=0.0
    cache = [[0 for i in range(4)] for i in range(n)]
    trade_lag =  [0 for i in range(3)]

    log_message('starting main loop')
    for index, row in trading_day.iterrows():
        
        # Update risk factor
        # TODO: implement and evaluate
        if total_quantity_filled > 5000:
            risk_factor = 1

        elif total_quantity_filled <= -5000:
            risk_factor = -1

        else:
            risk_factor = 0

        # get the time of this message
        time_from_open = (index - pd.Timedelta(hours=9, minutes=30))
        minutes_from_open = (time_from_open.hour * 60) + time_from_open.minute

        # MARKET DATA HANDLING
        if pd.isna(row.trade_px):  # it's a quote
            # skip if not NBBO
            if not ((row.qu_source == 'N') and (row.natbbo_ind == 4)):
                continue
            # set our local NBBO variables
            if (row.bid_px > 0 and row.bid_size > 0):
                bid_price = row.bid_px
                bid_size = row.bid_size
            if (row.ask_px > 0 and row.ask_size > 0):
                ask_price = row.ask_px
                ask_size = row.ask_size
            quote_count += 1
            message_type = 'q'
            cache.pop(0)
            cache.append([message_type, row.bid_px, row.ask_px, row.trade_px])
        else:  # it's a trade
            # store the last trade price
            prev_price = last_price
            # now get the new data
            last_price = row.trade_px
            last_size = row.trade_size
            trade_count += 1
            cumulative_volume += row.trade_size
            vwap_numerator += last_size * last_price
            message_type = 't'
            cache.pop(0)
            cache.append([message_type, row.bid_px, row.ask_px, row.trade_px])
            trade_lag.pop(0)
            trade_lag.append(row.trade_px)
            # CHECK OPEN ORDER(S) if we have a live order,
            # has it been filled by the trade that just happened?
            if live_order:
                if (order_side == 'b') and (last_price <= live_order_price):
                    fill_size = min(live_order_quantity, last_size)
                    record_trade(trades, index, live_order_price, fill_size, current_bar, 'p', order_side)
                    total_quantity_filled += fill_size
                    total_pass_count += 1

                    # even if we only got partially filled, let's assume we're cancelling the entire quantity.
                    # If we're still behind we'll replace later in the loop
                    live_order = False
                    live_order_price = 0.0
                    live_order_quantity = 0.0

                if (order_side == 's') and (last_price >= live_order_price):
                    fill_size = min(live_order_quantity, last_size)
                    record_trade(trades, index, live_order_price, fill_size, current_bar, 'p', order_side)
                    total_quantity_filled -= fill_size
                    total_pass_count += 1

                    # even if we only got partially filled, let's assume we're cancelling the entire quantity.
                    # If we're still behind we'll replace later in the loop
                    live_order = False
                    live_order_price = 0.0
                    live_order_quantity = 0.0
        
        
        # TICK FACTOR
        # only update if it's a trade
        if message_type == 't':
            # calc the tick
            this_tick = np.sign(last_price - prev_price)
            if this_tick == 0:
                this_tick = prev_tick

            # now calc the tick
            if tick_factor == 0:
                tick_factor = this_tick
            else:
                tick_factor = (tick_ema_alpha * this_tick) + (1 - tick_ema_alpha) * tick_factor

                # store the last tick
            prev_tick = this_tick

        # PRICING LOGIC
        new_midpoint = bid_price + (ask_price - bid_price) / 2
        if new_midpoint > 0:
            midpoint = new_midpoint
            
        # Determine the trade's trend using tick test
        if message_type == 't':
            flag=0
            temp = cache.copy()
            temp.reverse()
            temp.pop(0)
            while (len(temp)!=1):
                lag=temp.pop(0)
                if lag[0]=='t':
                    if last_price > lag[3]:
                        order_side = 'b'
                    if last_price < lag[3]:
                        order_side = 's'
                    if last_price == lag[3]:
                        if trade_lag[0] < lag[3]:
                            order_side = 'b'
                        if trade_lag[0] > lag[3]:
                            order_side = 's'
                        else:
                            order_side = '-'
                    flag=1
                    break

            # Determine the trade's trend using quote (if available, flag = 0)     
            if flag==0 and temp[0][0]=='q':
                midpoint = temp[0][1] + (temp[0][2] - temp[0][1]) / 2

                if last_price > midpoint:
                    order_side = 'b'
                if last_price < midpoint:
                    order_side = 's'
                if last_price == midpoint:
                    if last_price > trade_lag[1]:
                        order_side = 'b'
                    if last_price < trade_lag[1]:
                        order_side = 's'
                    if last_price == trade_lag[1]:
                        if trade_lag[0] < trade_lag[1]:
                            order_side = 'b'
                        if trade_lag[0] > trade_lag[1]:
                            order_side = 's'
                        else:
                            order_side = '-'
            # FAIR VALUE CALCULATION（when me meet the trade message)
            fair_value = midpoint + half_spread * ((tick_coef * tick_factor) + (risk_coef * risk_factor))

        # collect our data
        fair_values[index] = fair_value
        midpoints[index] = midpoint
        tick_factors[index] = tick_factor

        # TRADE DECISION
        # TODO: determine if we want to buy or sell
        # if fair price is < bid, sell agg for all bid price and amount
        # if fair price is < ask, sell passive
        # if fair price is > ask, buy agg for all ask price and amount
        # if fair price is > bid, buy passive

        # TRADING LOGIC
        # check where our FV is versus the BBO and constrain
        if message_type == 'q':
            if order_side == 'b':
                if fair_value >= ask_price:
                    total_agg_count += 1

                    new_trade_price = ask_price

                    # now place our aggressive order: assume you can execute the full size across spread
                    new_order_quantity = ask_size

                    record_trade(trades, index, new_trade_price, new_order_quantity, current_bar, 'a', order_side)

                    # update quantity remaining
                    total_quantity_filled += new_order_quantity

                    live_order_quantity = 0.0
                    live_order_price = 0.0
                    live_order = False

                else:  # we're not yet willing to cross the spread, stay passive
                    live_order_price = bid_price
                    live_order_quantity = ask_size
                    live_order = True

            elif order_side == 's':
                if fair_value <= bid_price:
                    total_agg_count += 1

                    new_trade_price = bid_price

                    # now place our aggressive order: assume you can execute the full size across spread
                    new_order_quantity = bid_size

                    # new_order_quantity = quantity_behind
                    record_trade(trades, index, new_trade_price, new_order_quantity, current_bar, 'a', order_side)

                    # update quantity remaining
                    total_quantity_filled -= new_order_quantity

                    live_order_quantity = 0.0
                    live_order_price = 0.0
                    live_order = False

                else:  # not yet willing to cross spread
                    live_order_price = ask_price
                    live_order_quantity = bid_size
                    live_order = True
            else:
                # no order here. for now just continue
                continue

    # looping done
    log_message('end simulation loop')
    log_message('order analytics')

    # Now, let's look at some stats
    trades = trades.dropna()
    day_vwap = vwap_numerator / cumulative_volume

    # prep our text output
    avg_price = 0
    if trades['shares'].sum() != 0:
        avg_price = (trades['price'] * trades['shares']).sum() / trades['shares'].sum()

    log_message('Algo run complete.')

    # assemble results and return
    # TODO: add P&L
    return {'midpoints': midpoints,
            'fair_values': fair_values,
            'tick_factors': tick_factors,
            'trades': trades,
            'quote_count': quote_count,
            'day_vwap': day_vwap,
            'avg_price': avg_price
            }