from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from lumibot.entities import TradingFee
from datetime import datetime 
from alpaca_trade_api import REST 
from timedelta import Timedelta 
from finbert_utils import estimate_sentiment
from collections import namedtuple
import pandas as pd
import numpy as np
API_KEY = "YOUR API KEY" 
API_SECRET = "YOUR API SECRET" 
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}

class MLTrader(Strategy): 
    symb = "NVDA"# "BTC-USD" #"NVDA" #"MSFT" #"AMZN" #"NFLX" #"TSLA" #"AMD" #"HUBS" # "AMZ" #"SHOP"
    x = 1 #number of days before
    data_price = []
    TRADE_INDICATOR_NEWS_ONLY = 1
    TRADE_INDICATOR_MOVING_AVG_ONLY = 2
    TRADE_INDICATOR_MOVING_AVG_AND_NEWS = 3
    USE_TRADE_INDICATOR = TRADE_INDICATOR_MOVING_AVG_AND_NEWS
    # Create two trading fees, one that is a percentage and one that is a flat fee
    trading_fee_1 = TradingFee(flat_fee=10) # $10 flat fee
    #trading_fee_2 = TradingFee(percent_fee=0.01) # 1% trading fee
    ai_propability = .999 #.999
    trading_iterator_count = 0
    GT_NONE = 0
    GT_BUY = 1
    GT_SEL = 2
    ma_indication_value = GT_NONE
    PointF = namedtuple('PointF', ['X', 'Y'], defaults=[0, 0])
    def initialize(self, symbol:str=symb, cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H" 
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        cash = self.get_cash() 
        if not cash>=0: raise Exception("Exception: Cash is negative")
        last_price = self.get_last_price(self.symbol)
        self.data_price.append(last_price)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity

    def get_dates(self): 
        today = self.get_datetime()
        x_days_prior = today - Timedelta(days=self.x)
        return today.strftime('%Y-%m-%d'), x_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        today, x_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=x_days_prior, end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 
    
    def on_trading_iteration(self):
        self.trading_iterator_count += 1
        try:
            cash, last_price, quantity = self.position_sizing()
            print(" cash: ", cash)
        except:
            print("Exception: position_sizing error") 
        print(self.get_datetime())
        if self.USE_TRADE_INDICATOR == self.TRADE_INDICATOR_NEWS_ONLY or \
            self.USE_TRADE_INDICATOR == self.TRADE_INDICATOR_MOVING_AVG_AND_NEWS:
            probability, sentiment = self.get_sentiment()
        
        if self.USE_TRADE_INDICATOR == self.TRADE_INDICATOR_MOVING_AVG_ONLY or \
            self.USE_TRADE_INDICATOR == self.TRADE_INDICATOR_MOVING_AVG_AND_NEWS:
            #### MOVING AVERAGE
            data_price_df = pd.Series(self.data_price)
            # Calculating the short window mov avg (4 days)
            short_rolling = data_price_df.rolling(window=4).mean()
            short_rolling.head(4)
            # Calculating the long-window mov avg (24 days)
            long_rolling = data_price_df.rolling(window=24).mean()
            long_rolling.tail()
            ema_short = data_price_df.ewm(span=4, adjust=False).mean()
            # Use the diff between the prices and the EMA timeseries
            trading_positions_raw = data_price_df - ema_short
            trading_positions_raw.tail()
            # Use the sign of the difference to determine whether the price or the EMA is greater
            trading_positions = trading_positions_raw.apply(np.sign)
            trading_positions.tail()
            trading_positions_final = trading_positions.shift(1)
            # Leave out trade signals by two days
            if self.USE_TRADE_INDICATOR == self.TRADE_INDICATOR_MOVING_AVG_ONLY:
                if self.trading_iterator_count == 1 or self.trading_iterator_count == 2:
                    trading_positions_final.values[self.trading_iterator_count-1] = 0 #neutral
            else:
                if self.trading_iterator_count == 1 or self.trading_iterator_count == 2:
                    if sentiment == "neutral":
                        trading_positions_final.values[self.trading_iterator_count-1] = 0 #neutral
                    elif sentiment == "positive":
                        trading_positions_final.values[self.trading_iterator_count-1] = 1 #buy
                    elif sentiment == "negative":
                        trading_positions_final.values[self.trading_iterator_count-1] = -1 #sell
            if(trading_positions_final.values[self.trading_iterator_count-1] != 0):
                self.ma_indication_value = self.GT_BUY if trading_positions_final.values[self.trading_iterator_count-1] > 0 else self.GT_SEL
            else: #undefined
                self.ma_indication_value = self.GT_NONE

        if self.USE_TRADE_INDICATOR == self.TRADE_INDICATOR_MOVING_AVG_AND_NEWS:
            if cash > last_price: 
                if sentiment == "positive" and probability > self.ai_propability and self.ma_indication_value == self.GT_BUY: 
                    if self.last_trade == "sell": 
                        self.sell_all() 
                    order = self.create_order(
                        self.symbol, 
                        quantity, 
                        "buy", 
                        type="bracket", 
                        take_profit_price=last_price*1.5,
                        stop_loss_price=last_price*.9
                    )
                    self.submit_order(order) 
                    self.last_trade = "buy"
                    self.first_trade_was_buy = True         
                elif sentiment == "negative" and probability > self.ai_propability and self.ma_indication_value == self.GT_SEL: 
                    if self.last_trade == "buy": 
                        self.sell_all()
                    if self.last_trade == None:
                        print("INVALID SELL BEFORE BUY")
                    else:
                        order = self.create_order(
                            self.symbol, 
                            quantity, 
                            "sell", 
                            type="bracket", 
                            take_profit_price=last_price*.5,
                            stop_loss_price=last_price*1.1
                        )
                        self.submit_order(order) 
                        self.last_trade = "sell"
        elif self.USE_TRADE_INDICATOR == self.TRADE_INDICATOR_MOVING_AVG_ONLY:
            if cash > last_price: 
                if self.ma_indication_value == self.GT_BUY: 
                    if self.last_trade == "sell": 
                        self.sell_all() 
                    order = self.create_order(
                        self.symbol, 
                        quantity, 
                        "buy", 
                        type="bracket", 
                        take_profit_price=last_price*1.5,
                        stop_loss_price=last_price*.9
                    )
                    self.submit_order(order) 
                    self.last_trade = "buy"
                    self.first_trade_was_buy = True
                elif self.ma_indication_value == self.GT_SEL:
                    if self.last_trade == "buy": 
                        self.sell_all()
                    if self.last_trade == None:
                        print("INVALID SELL BEFORE BUY")
                    else:
                        order = self.create_order(
                            self.symbol, 
                            quantity, 
                            "sell", 
                            type="bracket", 
                            take_profit_price=last_price*.5,
                            stop_loss_price=last_price*1.1
                        )
                        self.submit_order(order) 
                        self.last_trade = "sell"
        else: #self.TRADE_INDICATOR_NEWS_ONLY:
            if cash > last_price: 
                if sentiment == "positive" and probability > self.ai_propability: 
                    if self.last_trade == "sell": 
                        self.sell_all() 
                    order = self.create_order(
                        self.symbol, 
                        quantity, 
                        "buy", 
                        type="bracket", 
                        take_profit_price=last_price*1.5,
                        stop_loss_price=last_price*.9
                    )
                    self.submit_order(order) 
                    self.last_trade = "buy"
                    self.first_trade_was_buy = True
                elif sentiment == "negative" and probability > self.ai_propability:
                    if self.last_trade == "buy": 
                        self.sell_all() 
                    if self.last_trade == None:
                        print("INVALID SELL BEFORE BUY")
                    else:   
                        order = self.create_order(
                            self.symbol, 
                            quantity, 
                            "sell", 
                            type="bracket", 
                            take_profit_price=last_price*.5,
                            stop_loss_price=last_price*1.1
                        )
                        self.submit_order(order) 
                        self.last_trade = "sell"

start_date = datetime(2020,12,31)
end_date = datetime(2024,3,7) 
broker = Alpaca(ALPACA_CREDS) 
strategy = MLTrader(name='mlstrat', broker=broker, benchmark_asset=MLTrader.symb,
                    parameters={"symbol":MLTrader.symb, "cash_at_risk":.7})
strategy.backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, benchmark_asset=MLTrader.symb, 
    parameters={"symbol":MLTrader.symb, "cash_at_risk":.7}, buy_trading_fees=[MLTrader.trading_fee_1], sell_trading_fees=[MLTrader.trading_fee_1],
)
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()
