from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import backtrader as bt
import pandas as pd
import datapungi_fed as dpf

# economic data (weekly)
data = dpf.data("***") # API from st louis fed required
df = data.series('STLFSI2').pct_change(fill_method ='ffill').shift(1) # St. Louis Fed Financial Stress Index
df2 = data.series('WEI').pct_change(fill_method ='ffill').shift(1) # Weekly Economic Index (Lewis-Mertens-Stock)
df3 = data.series('WAAA').pct_change(fill_method ='ffill').shift(1) # Moody's Seasoned Aaa Corporate Bond Yield
df.index = df.index + pd.DateOffset(days=2)
df2.index = df2.index + pd.DateOffset(days=1)
df3.index = df3.index + pd.DateOffset(days=2)

# create dataframe for Factor open/close/high/low/volume pricing data
Factor = pd.read_csv('VLUE.csv')
Factor['Date'] = Factor['Date'].astype(str).str[0:10]
Factor['Date'] = pd.to_datetime(Factor['Date'])
datetime_index = pd.DatetimeIndex(Factor['Date'])
Factor = Factor.set_index(datetime_index).drop(['Date'], axis=1)
Factor=Factor.resample('W').mean()

# create VLUE_2019 dataset
Factor_2019onwards = pd.read_csv('VLUE.csv')
Factor_2019onwards['Date'] = Factor_2019onwards['Date'].astype(str).str[0:10]
Factor_2019onwards['Date'] = pd.to_datetime(Factor_2019onwards['Date'])
datetime_index = pd.DatetimeIndex(Factor_2019onwards['Date'])
Factor_2019onwards = Factor_2019onwards.set_index(datetime_index).drop(['Date'], axis=1)
Factor_2019onwards = Factor_2019onwards.truncate('2019-02-01')
Factor_2019onwards.to_csv(r'C:/Users/x/Desktop/Python Projects/VLUE_trade_strategy/VLUE_2019onwards.csv')

# technical analysis
Factor['Change'] = Factor['Close'].pct_change()
Factor['30T'] = Factor['Close'].rolling(22).mean()
Factor['30THigh'] = Factor['Close'].rolling(22).mean() + Factor['Close'].rolling(22).std()
Factor['30TLow'] = Factor['Close'].rolling(22).mean() - Factor['Close'].rolling(22).std()
Factor['up'] = Factor['Change'][Factor['Change']>0]
Factor['down'] = Factor['Change'][Factor['Change']<0]
Factor['roll_up'] = Factor['up'].rolling(14, min_periods=1).mean()
Factor['roll_down'] = Factor['down'].rolling(14, min_periods=1).mean()
Factor['RSI'] = 100.0 - (100.0 / (1 + Factor['roll_up']/abs(Factor['roll_down'])))
Factor['MACD'] = Factor['Close'].ewm(span = 12).mean() - Factor['Close'].ewm(span = 26).mean()
Factor['MACDSignal'] = Factor['Close'].ewm(span = 9).mean()
Factor['MACDHistogram'] = Factor['MACD'] - Factor['MACDSignal']
Factor['ConvLine'] = (Factor['Close'].rolling(9).max() + Factor['Close'].rolling(9).min())/2
Factor['BaseLine'] = (Factor['Close'].rolling(26).max() + Factor['Close'].rolling(26).min())/2
Factor['LeadSpanA'] = (Factor['BaseLine'] + Factor['ConvLine'])/2
Factor['LeadSpanB'] = (Factor['Close'].rolling(52).max() + Factor['Close'].rolling(52).min())/2
Factor['LaggingSpan'] = Factor['Close'].shift(periods=26)
Factor['Ichimoko1Buy'] =  (Factor['LaggingSpan'] > Factor['ConvLine']) & (Factor['ConvLine'] > Factor['BaseLine'])
Factor['Ichimoko1Sell'] =  (Factor['LaggingSpan'] < Factor['ConvLine']) & (Factor['ConvLine'] < Factor['BaseLine'])
Factor['BollBuy'] = Factor['30T'] < Factor['30TLow']
Factor['BollSell'] = Factor['30T'] > Factor['30THigh']
Factor['MACDBuy'] = (Factor['MACDSignal'] > Factor['MACD']) &(Factor['MACD']> 0)
Factor['MACDSell'] = (Factor['MACDSignal'] < Factor['MACD']) & (Factor['MACD'] < 0)
Factor['RSIBuy'] = (Factor['RSI'] < 30)
Factor['RSISell'] = (Factor['RSI'] > 70)
Factor = Factor.drop(['Open','High','Low','Close','Volume','30T','30THigh','30TLow','up','down','roll_up','roll_down','RSI','MACD','MACDSignal','MACDHistogram','ConvLine','BaseLine','LeadSpanA','LeadSpanB','LaggingSpan'],axis=1)
Factor = Factor[['Ichimoko1Buy','Ichimoko1Sell','BollBuy','BollSell','MACDBuy','MACDSell','RSIBuy','RSISell','Change']]
Factor.loc[(Factor.Change >= 0.001),"significant_change"] = 1
Factor['significant_change'] = Factor['significant_change'].fillna(0)
Factor = Factor.drop(['Change'],axis=1)

# Split dataframe into training and backtesting
fulldf = pd.concat([df,df2,df3,Factor],axis=1).dropna()
n = 297 # trained until 2018 data
trainingdf = fulldf[:n]
backtestdf = fulldf[n:]

# lag all the variables by 1 T
lag = 1
trainingdf['Ichimoko1Buy'] = trainingdf['Ichimoko1Buy'].shift(lag)
trainingdf['Ichimoko1Sell'] = trainingdf['Ichimoko1Sell'].shift(lag)
trainingdf['BollBuy'] = trainingdf['BollBuy'].shift(lag)
trainingdf['BollSell'] = trainingdf['BollSell'].shift(lag)
trainingdf['MACDBuy'] = trainingdf['MACDBuy'].shift(lag)
trainingdf['MACDSell'] = trainingdf['MACDSell'].shift(lag)
trainingdf['RSIBuy'] = trainingdf['RSIBuy'].shift(lag)
trainingdf['RSISell'] = trainingdf['RSISell'].shift(lag)
trainingdf = trainingdf.dropna()

print(backtestdf)
# implementation of machine learning model
X_train = trainingdf.iloc[:, :-1].values
y_train = trainingdf.iloc[:, -1].values
X_test = backtestdf.iloc[:, :-1].values
y_test = backtestdf.iloc[:, -1].values

# Training the selected classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results with 2019 onwards data inputs
y_pred = classifier.predict(X_test)
print(y_pred)
output = pd.DataFrame(y_pred)

backtestdfdate = backtestdf.reset_index()
backtestdfdate = backtestdfdate.drop(['STLFSI2','WEI','WAAA','Ichimoko1Buy','Ichimoko1Sell','BollBuy','BollSell','MACDBuy','MACDSell','RSIBuy','RSISell','significant_change'],axis=1)

output = pd.concat([backtestdfdate,output],axis=1).dropna()
output['Date'] = output['index'].astype(str).str[0:10]
output ['Date'] = pd.to_datetime(output['Date'])
datetime_index = pd.DatetimeIndex(output['Date'])
output = output.set_index(datetime_index).drop(['Date','index'], axis=1)
output=output.resample('D').ffill()
output = output.reset_index()
output = output.drop(['Date'],axis=1)

Factor_2019onwards_asinputs = pd.read_csv('VLUE_2019onwards.csv')

fulldf2 = pd.concat([Factor_2019onwards_asinputs,output],axis=1).dropna()
fulldf2['Date'] = fulldf2['Date'].astype(str).str[0:10]
fulldf2['Date'] = pd.to_datetime(fulldf2['Date'])
datetime_index = pd.DatetimeIndex(fulldf2['Date'])
fulldf2 = fulldf2.set_index(datetime_index).drop(['Date'], axis=1)

fulldf2.to_csv(r'C:/Users/x/Desktop/Python Projects/VLUE_trade_strategy/VLUE_2019onwards_asdatafeed.csv')


"""**********   backtester   **********"""

class dataFeed(bt.feeds.GenericCSVData):
    params = (
        ('dtformat', '%Y-%m-%d'),
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest',6) # openinterest is pseudonym for machine generated signal 0 or 1
    )

# Create a Strategy
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.signal = self.datas[0].openinterest
        # To keep track of pending orders
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.signal[0] == 1.0:

                    # BUY, BUY, BUY!!! (with default parameters)
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])

                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.buy()

        else:

            # Already in the market ... we might sell
            if self.signal[0] == 0.0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    data = dataFeed(dataname = 'C:/Users/x/Desktop/Python Projects/VLUE_trade_strategy/VLUE_2019onwards_asdatafeed.csv')

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100.0)

    # Set the commission - 1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.plot()