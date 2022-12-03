# Imports and Dependencies.
from termcolor import colored, cprint
from Learners import DeepQNetwork
from Learners import StrategyLearner
from Learners import QLearner
from Utilities import *
from simple_colors import *
import time

import warnings
warnings.filterwarnings('ignore')

def DqnAccuracyOutput(dfTrades):
    """
    Description: This function will interpret the dataframe it receives,
    check the last 10 predictions it made, and check if its predictions where
    accurate or not for the deep q neural network learner.
    """

    accuracyDict = {}
    accuracyDates = []

    for idx in range(2, 35):
        actualIdx = -idx
        date = str(dfTrades.index[actualIdx])
        date = date.split(" ")[0]
        nextDate = dfTrades.iloc[actualIdx + 1]["Return"]
        
        if (nextDate > 0):
            prediction = "ACCURATE"
            accuracyDates.append(actualIdx + 1)
        elif(nextDate < 0):
            prediction = "INACCURATE"

        accuracyDict[date] = prediction

    return accuracyDict, accuracyDates


def StlAccuracyOutput(dfTrades):
    """
    Description: This function will interpret the dataframe it receives,
    check the last 10 predictions it made, and check if its predictions where
    accurate or not for the strategy dyna q learner.
    """

    accuracyDictStl = {}
    accurateDates = []

    for idx in range(2, 30):
        actualIdx = -idx
        date = str(dfTrades.index[actualIdx])
        date = date.split(" ")[0]
        currentDate = dfTrades.iloc[actualIdx]["Port Val"]
        nextDate = dfTrades.iloc[actualIdx + 1]["Port Val"]

        if (nextDate > currentDate):
            prediction = "ACCURATE"
            accurateDates.append(actualIdx + 1)

        elif(nextDate < currentDate):
            prediction = "INACCURATE"
        else:
            continue

        accuracyDictStl[date] = prediction

    return accuracyDictStl, accurateDates


def DqnOutput(symbol, verbose=False):
    """
    Description: This function will basically provide a order recommendation of hold/sell/buy based on a trained Deep Q
    Neural Network Learner. It will output what the order recommendation is based on what specific it believes is best.
    """
    print("\nDeep-Q Neural Network function has begun...")
    
    # Begin timing how long computing takes.
    startTimer = time.time()
    
    # Training time frame.
    trainStart = "2021-07-01"
    trainEnd = "2022-07-01"
    
    # Testing time frame.
    testStart = "2022-07-02"
    testEnd = datetime.today().strftime('%Y-%m-%d')
    
    # Create training df with historical data.
    trainDf = IndividualHistoricalData(
                    symbol,
                    trainStart,
                    trainEnd,
                    keepAllColumns="No")
    
    # Create testing df with historical data.
    testDf = IndividualHistoricalData(
                    symbol,
                    testStart,
                    testEnd,
                    keepAllColumns="No")
    
    # Create DQN instance.
    dqn = DeepQNetwork(
                    alpha=0.004, # Learning rate.
                    stateSize=10, # Amount of states within Q table.
                    actionSize=3, # Buy, sell, hold.
                    verbose=False) # Don't spit random messages.
    
    # Transform dfs into tradable dataframes.
    trainDf = dqn.TransformDf(trainDf)
    testDf = dqn.TransformDf(testDf)

    print("Deep-Q function has begun training...")
    # Create a variety of epochs to train and test data.
    for epoch in range(20):
        dqn.memory = []
        dqn.AddEvidence(
                    trainDf, # Training dataframe.
                    trainEnd, # End training data.
                    dqn.memory)
        if epoch == 10:
            print("Deep-Q function is halfway finished...")
        dqn.Query(testDf)
    print("Deep-Q function has almost finished...")
    
    # End timing for computing.
    endTimer = time.time()
    
    elapsed = round((endTimer - startTimer) / 60)
    
    # Create trades dataframe.
    testTrades = dqn.CreateTradesDf(
                    testDf,
                    dqn)

    # Save the plot for the performance.
    DeepQNetwork().PlotDqnPerformance(testTrades, symbol)

    accuracyDict, accuracyList = DqnAccuracyOutput(testTrades)    
    
    # Create text from action for easier output.
    if (testTrades.iloc[-1]["Trade"] == 0):
        tradingDecision = colored("SELL", "red", attrs=["bold"])
        uncoloredDecision = "SELL"
    elif (testTrades.iloc[-1]["Trade"] == 1):
        tradingDecision = colored("HOLD", "blue", attrs=["bold"])
        uncoloredDecision = "HOLD"
    elif (testTrades.iloc[-1]["Trade"] == 2):
        tradingDecision = colored("BUY", "green", attrs=["bold"])
        uncoloredDecision = "BUY"
          
    # Convert datetime to string.      
    dt = str(testTrades.index[-1])
    dt = dt.split(" ")[0]
        
    if verbose:
        # Print an organized message.
        print(magenta('\nDeep QLearner', ['italic', 'underlined', 'bold']))
        
        # Print computing time.
        cprint("Timing Computation", attrs=["bold", "underline"])
        print("This model took {} minutes to compute.".format(elapsed))
        
        cprint("Technical Indicators", attrs=["bold", "underline"])
        print("This model used the daily return of {} to perform its trades.".format(symbol))
        
        # Print out clean directions of recommended order.
        cprint("Trading Decision", attrs=["bold", "underline"])
        print("Based on the behavior of {} in the time range {} to {},"
            .format(symbol, trainStart, trainEnd))
        print("as well as the closing price on {}, our DQN says you should: {}"
            .format(dt, tradingDecision))

    sym = str(symbol).lower().capitalize()
    PlotAccurateData(accuracyList, figName=f"Images/{sym}DqnPredictionsPlot.png")
    print("Deep-Q function has finished!")

    return uncoloredDecision, testTrades, accuracyDict, dt


def DynaQOutput(symbol, verbose=False):
    """
    Description: This function will basically provide a order recommendation of hold/sell/buy based on a trained Dyna-Q
    Learner that trades based on momentum, standard moving average, and bollinder bandwidth. It will output what the 
    order recommendation is based on what specific computation.
    """
    print("Strategy Dyna-Q function has begun...")

    # Begin timing how long computing takes.
    startTimer = time.time()
    stlTradeDecision = "N/A"
    
    # Training time frame.
    trainStart = "2021-07-01"
    trainEnd = "2022-07-02"
    
    # Testing time frame.
    testStart = "2022-07-02"
    testEnd = datetime.today().strftime('%Y-%m-%d')
    
    # Create benchmark df for trades.
    dfBmTrades = StrategyLearner().CreateBmDfTrades(
        symbol,
        startDate=trainStart,
        endDate=trainEnd,
        numOfShares=1)
    
    # Create dyna Q learner instance.
    learner = QLearner(
        numOfStates=3000, # Number of states in Q table.
        numOfActions=3) # Buy, sell, or hold.
    
    # Create a strategy Q trader.
    stl = StrategyLearner(
        accuracyBool=True,
        numOfShares=1,
        impact=0.00,
        commission=0.00,
        verbose=False,
        learner=learner)
    
    # Train model.
    stl.AddEvidence(
        symbol=symbol,
        startVal=1,
        startDate=trainStart,
        endDate=trainEnd)
    
    # Create trading dataframe with suggestions.
    dfTrades = stl.Query(
        symbol=symbol,
        startDate=testStart,
        endDate=testEnd)

    # Create trading dataframe for training dataset for visuals.
    dfTrainingTrades = stl.Query(
        symbol=symbol,
        startDate=trainStart,
        endDate=trainEnd
    )

    # Check accuracy of learner in past trades.
    training = StrategyLearner().SymbolValueFromTrading(
            dfOrders=dfTrades,
            symbol=symbol,
            startVal=1,
            startDate=testStart,
            endDate=testEnd,
            commision=0,
            impact=0)

    comb = CombineDfs(
        [dfTrades, training],
        testStart,
        testEnd)

    accuracyDict, accuracyList = StlAccuracyOutput(comb)

    # Grab end of data for comparison.
    data = IndividualHistoricalData(symbol, testStart, testEnd, "No")
    endOfData = data.index[-1]
    endOfDataAsString = str(endOfData)
    endOfDataAsString = endOfDataAsString.split(" ")[0]
    
    # Assign the last date of trade to a variable.
    lastTrade = dfTrades.index[-1]   
    
    # Extract trading decision.
    if (lastTrade == endOfData):
        if (dfTrades.iloc[-1]["Shares"] == -1):
            stlTradeDecision = colored("SELL", "red", attrs=["bold"])
            uncoloredDecision = "SELL"
        elif (dfTrades.iloc[-1]["Shares"] == 1):
            stlTradeDecision = colored("BUY", "green", attrs=["bold"])
            uncoloredDecision = "BUY"
        elif (dfTrades.iloc[-1]["Shares"] == 0):
            stlTradeDecision = colored("HOLD", "blue", attrs=["bold"])
            uncoloredDecision = "HOLD"
    # else:
      #  stlTradeDecision = colored("HOLD", "blue", attrs=["bold"])
    
    # End timing for computing.
    endTimer = time.time()
    elapsed = (endTimer - startTimer)
    
    if verbose:
        # Print an organized message.
        print(magenta('\nStrategy QLearner', ['italic', 'underlined', 'bold']))
        
        # Print computing time.
        cprint("Timing Computation", attrs=["bold", "underline"])
        print("This model took {} seconds to compute.".format(round(elapsed)))
        
        cprint("Technical Indicators", attrs=["bold", "underline"])
        print("This model used the standard moving average, momentum, "
            "and bollinger bandwidth of {} to perform its trades.".format(symbol))
        
        # Print out clean directions of recommended order.
        cprint("Trading Decision", attrs=["bold", "underline"])
        print("Based on the behavior of {} in the time range {} to {},"
            .format(symbol, trainStart, trainEnd))
        print("as well as the closing price on {}, our Dyna-Q Learner says you should: {}"
            .format(endOfDataAsString, stlTradeDecision))

    symbolCased = str(symbol).lower().capitalize()

    StrategyLearner().MarketSimulator(dfTrainingTrades, dfBmTrades,
                                          symbol=symbol,
                                          startDate=trainStart,
                                          endDate=trainEnd,
                                          title="QLearner Training",
                                          startVal=1,
                                          commission=0,
                                          impact=0,
                                          saveFigure=True,
                                          figName=f"Images/{symbolCased}StlLearnerVisual.png",
                                          showPlot=False
                                          )

    PlotAccurateData(accuracyList, figName=f"Images/{symbolCased}StlPredictionsPlot.png")
    
    print("Strategy Dyna-Q function has finished!")

    return uncoloredDecision, dfTrades, accuracyDict, endOfDataAsString


if __name__ == "__main__":
    stlDecision, dfTrades, accuracyDict, lastDate = DynaQOutput("AAPL")
    dqnDecision, testTrades, accuracyDictDqn, lastDay = DqnOutput("AAPL")