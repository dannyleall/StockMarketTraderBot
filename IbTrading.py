from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
from ibapi.order import *


# *********************************************************************************************************************************************************** #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------- Connecting IB and TWS to Code ------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# *********************************************************************************************************************************************************** #


""" ------------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------------ Connecting to IB TWS --------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

class IbTwsApi(EWrapper, EClient):
    
    def __init__(self):
        
        # Initializes a client instance.
        EClient.__init__(self, self) 


""" ------------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------------ IB Account Functions --------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

class IbFunctions(object):
    
    # Declaring ib variable.
    ib = None

    def __init__(self):
        
        # Connect to IB TWS (see class for reference).
        self.ib = IbTwsApi()

        # Connect using local IP used, port 7497, and 1 for suggested value.
        self.ib.connect("127.0.0.1", 7497, 1) 

        # Placing IBapi on separate thread and updates IB actions.
        ibThread = threading.Thread(target=self.RunLoop, daemon=True) 
        ibThread.start()

        # Allows IBSession take a break between actions.
        time.sleep(1) 


    def BuyOrder(self, ticker, quantity):

        # CREATING ORDER, determines the type, buy/sell, and quantity.
        order = Order()
        order.orderType = "MKT" #order type is on a market
        order.action = "BUY" 
        order.totalQuantity = quantity

        #CONTRACT FOR ORDER, determines what stock, if it is a stock, what exchange to use, currency, and backup exchange.
        contract = Contract()
        contract.symbol = ticker

        # Other types such as futures available.
        contract.secType = "STK" 

        # Automatically determines best exchange to trade in.
        contract.exchange = "SMART" 
        contract.currency = "USD" 

        # Backup exchange.
        contract.primaryExchange = "ISLAND" 
        
        #PLACE THE ORDER, ID MUST BE UNIQUE (uses time).
        self.ib.placeOrder(int(round(time.time())), contract, order) 
        time.sleep(1) 
        

    def SellOrder(self, ticker, quantity):
        
        # CREATING ORDER, determines the type, buy/sell, and amount.
        order = Order()
        order.orderType = "MKT"
        order.action = "SELL"
        order.totalQuantity = quantity

        # CONTRACT FOR ORDER, determines what stock, if it is a stock, what exchange to use, currency, and backup exchange.
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK" 
        contract.exchange = "SMART"
        contract.currency = "USD" 
        contract.primaryExchange = "ISLAND"
       
        #PLACE THE ORDER
        self.ib.placeOrder(int(round(time.time())), contract, order) 
        time.sleep(1) 


    def RunLoop(self):
        self.ib.run()

    
    def TestRun():
        # Initiates IB session.
        ibSession = IbFunctions() 
        ibSession.BuyOrder("AAPL", 54)
        ibSession.SellOrder("AAPL", 34)

        ibSession.BuyOrder("GOOGL", 5)
        ibSession.SellOrder("GOOGL", 1)

