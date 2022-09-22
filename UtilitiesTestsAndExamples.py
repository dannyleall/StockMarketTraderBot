from Utilities import *

# *********************************************************************************************************************************************************** #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------- TRADERBOT FUNCTION TESTING -------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# *********************************************************************************************************************************************************** #


""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------------------- Dataframes Functions Unit Tests ---------------------------------------------------------- """
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """

# IndividualHistoricalData Tests 
def IndividualHistoricalDataUnitTests():

    # Success.
    print("\n\n\nIndividualHistoricalData Success Tests:")
    print(IndividualHistoricalData('MSFT', "2020-01-01", "2021-09-09"))

    print("\n\n\nIndividualHistoricalData Error Tests\nSymbols Error Test:")
    IndividualHistoricalData("ERROR", "2020-01-01", "2021-09-09")   # symbols error

    print("\n\n\nIndividualHistoricalData Error Tests\nTimeRange Error Test:")
    IndividualHistoricalData('MSFT', "2021-09-09", "2020-01-01")    # TimeRange error



# HistoricalData Tests 
def HistoricalDataUnitTests():

    # Success.
    print("\n\n\nHistoricalData Success Tests:")
    print(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"))

    print("\n\n\nHistoricalData Error Tests\nTimeRange Error Test:")
    HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2021-09-09", "2020-01-01")     # TimeRange error

    print("\n\n\nHistoricalData Error Tests\nsymbolsArray Error Test:")
    HistoricalData(100, "2020-01-01", "2021-09-09")    # symbolsArray error



# NormalizeDfs Tests
def NormalizeDfsUnitTests():

    # Success
    print("\n\n\nNormalizeDfs Success Tests:")
    print(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")))

    print("\n\n\nNormalizeDfs Error Tests\ndfArray Error Test:")
    NormalizeDfs(['AAPL'])   # dfArray error



# CombineDfs Tests 
def CombineDfsUnitTests():
    
    print("\n\n\nCombineDfs Success Tests:")
    print(CombineDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"), "2020-01-01", "2021-09-09"))

    print("\n\n\nCombineDfs Error Tests\nTimeRange Error Test:")
    CombineDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"), "2021-09-09", "2020-01-01")   # TimeRange error
    


# StockReturns Tests 
def StockReturnsUnitTests():
    
    print("\n\n\nStockReturns Daily Success Tests:")
    print(StockReturns(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), "daily"))

    print("\n\n\nStockReturns Monthly Success Tests:")
    print(StockReturns(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), "monthly"))

    print("\n\n\nStockReturns Error Tests\ndfArray Error Test:")
    StockReturns([0.55], "daily")      # dfArray error

    print("\n\n\nStockReturns Error Tests\ndailyOrMonthly Error Test:")
    StockReturns(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), "ERROR")   # dailyOrMonthly error 



# CumulativeReturns Tests
def CumulativeReturnsUnitTests():
    
    print("\n\n\nCumulativeReturns Daily Success Tests:")
    print(CumulativeReturns(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), "daily"))

    print("\n\n\nCumulativeReturns Monthly Success Tests:")    
    print(CumulativeReturns(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), "monthly"))

    print("\n\n\nCumulativeReturns Error Tests\ndailyOrMonthly Error Test:")
    CumulativeReturns(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), "ERROR")      # dailyOrMonthly error



""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------------------- Statistical Functions Unit Tests --------------------------------------------------------- """
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """

# GetMaxClose Tests
def GetMaxCloseUnitTests():
    
    print("\n\n\nGetMaxClose Success Tests:")
    print(GetMaxClose(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")))

    print("\n\n\nGetMaxClose Error Tests\ndfArray Error Test:")
    GetMaxClose([0.25])    # dfArray error



""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ---------------------------------------------------------- Slice And Validate Functions Unit Tests ------------------------------------------------------ """
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """

# SliceRow Tests 
def SliceRowUnitTests():
    
    print("\n\n\nSliceRow Success Tests:")
    print(SliceRow(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"), "2020-01-01", "2020-01-31"))

    print("\n\n\nSliceRow Error Tests\ndfArray Error Test:")
    SliceRow([0.05], "2020-01-01", "2021-09-09") # dfArray error

    print("\n\n\nSliceRow Error Tests\nTimeFrame Error Test:")
    SliceRow(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"), "2021-09-09", "2020-01-01")     # TimeFrame error



# SliceColumnAndRow Tests 
def SliceColumnUnitTests():
    
    print("\n\n\nSliceColumn Success Tests:")
    print(SliceColumn(CombineDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"), "2020-01-01", "2021-09-09"), ['AAPL', 'MSFT']))

    print("\n\n\nSliceColumn Error Tests\ndfArray Error Test:")
    SliceColumn([0.25], ['SPY']) # dfArray error
    
    print("\n\n\nSliceColumn Error Tests\ncolArray Error Test:")
    SliceColumn(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"), 1)     # colArray error
    


# ValidateDates Tests
def ValidateDatesUnitTests():
    
    print("\n\n\nValidateDates Success Tests:")
    ValidateDates("2021-01-01", "2021-12-31")   # Success    

    print("\n\n\nValidateDates Error Tests\n Error Test:")
    ValidateDates("2022-01-22", "2000-01-01")   # Error



""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------------------------ Plot Functions Unit Tests ------------------------------------------------------------ """
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """

# PlotData Tests
def PlotDataUnitTests():
    
    print("\n\n\nPlotData Success Tests:")
    PlotData(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), "Stock Prices", "Date", "Price")    # Success

    print("\n\n\nPlotData Error Tests\ntitle Error Test:")
    PlotData(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), 1, "Date", "Price")    # title error
    
    print("\n\n\nPlotData Error Tests\nx label Error Test:")
    PlotData(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), "Stock Prices", 1, "Price")    # x label error

    print("\n\n\nPlotData Error Tests\ny label Error Test:")
    PlotData(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), "Stock Prices", "Date", 1)     # y label error



# PlotRollingMean Tests 
def PlotRollingMeanUnitTests():
    
    print("\n\n\nPlotRollingMean Success Tests:")
    PlotRollingMean(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), 10)    

    print("\n\n\nPlotRollingMean Error Tests\ndfArray Error Test:")
    PlotRollingMean([0.25], 20)    # dfArray error

    print("\n\n\nPlotRollingMean Error Tests\nwindow Error Test:")
    PlotRollingMean(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), 10.56)  # window error    



# PlotBollingerBands Tests
def PlotBollingerBandsUnitTests():
    
    print("\n\n\nPlotBollingerBands Success Tests:")
    PlotBollingerBands(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), 15)

    print("\n\n\nPlotBollingerBands Error Tests\ndfArray Error Test:")
    PlotBollingerBands([0.25], 10)      # dfArray error

    print("\n\n\nPlotBollingerBands Error Tests\nwindow Error Test:")
    PlotBollingerBands(NormalizeDfs(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09")), 10.56)   # window error



# PlotHistogram Tests
def PlotHistogramUnitTests():
    
    print("\n\n\nPlotHistogramUnitTests No Statistics Success Tests:")
    dailyReturns = StockReturns(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"), "daily")
    PlotHistogram(dailyReturns, "no", 10)      # Success with no statistics on graph

    print("\n\n\nPlotHistogramUnitTests Statistics Success Tests:")
    twoDfsTest = [dailyReturns[0], dailyReturns[2]]
    PlotHistogram(twoDfsTest, "yes", 10)    # Success with statistics on graph

    print("\n\n\nPlotHistogramUnitTests Error Tests\ndfArray Error Test:")
    PlotHistogram([0.25], "yes", 15)   # dfArray error

    print("\n\n\nPlotHistogramUnitTests Error Tests\nplotStatisticsYesOrNo Error Test:")
    PlotHistogram(dailyReturns, "ERROR", 10)    # plotStatisticsYesOrNo error

    print("\n\n\nPlotHistogramUnitTests Error Tests\nBin Error Test:")
    PlotHistogram(dailyReturns, "yes", 10.56)   # Bin error



# PlotScatter Tests
def PlotScatterUnitTests():
    
    print("\n\n\nPlotScatter Success Tests:")
    dailyReturns = StockReturns(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"), "daily")
    twoDfsTest = [dailyReturns[0], dailyReturns[2]]
    PlotScatter(CombineDfs(twoDfsTest, "2020-01-01", "2021-09-09"))     # Success

    print("\n\n\nPlotScatter Error Tests\nCombineDfs Error Test:")
    PlotScatter(twoDfsTest)     # CombineDfs error

    print("\n\n\nPlotScatter Error Tests\ndfArray Error Test:")    
    PlotScatter(dailyReturns)   # dfArray error



# PlotCorrelationMatrix Tests
def PlotCorrelationMatrixUnitTests():
    
    print("\n\n\nPlotCorrelationMatrix Success Tests:")
    dailyReturns = StockReturns(HistoricalData(['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN'], "2020-01-01", "2021-09-09"), "daily")
    PlotCorrelationMatrix(CombineDfs(dailyReturns, "2020-01-01", "2021-09-09"))     # Success 

    print("\n\n\nPlotCorrelationMatrix Error Tests\nCombineDfs Error Test:")
    PlotCorrelationMatrix(dailyReturns)     # CombineDfs error
    



""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------------------- Portfolio Functions Unit Tests ----------------------------------------------------------- """
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """

# ComputePortfolioValue Tests 
def ComputePortfolioValueUnitTests():
    
    print("\n\n\nComputePortfolioValue Success Tests:")
    startVal = 100
    startDate = "2020-01-01"
    endDate = "2021-09-09"
    symbols = ['XOM', 'GE', 'JPM', 'BP', 'AMZN', 'MSFT', 'AAPL', 'GOOGL']
    allocations = [0.05, 0.10, 0.05, 0.10, 0.15, 0.25, 0.15, 0.15]
    print(ComputePortfolioValue(startVal, startDate, endDate, symbols, allocations))      # Success

    print("\n\n\nComputePortfolioValue Error Tests\nstartVal Error Test:")
    ComputePortfolioValue(1000.0101, startDate, endDate, symbols, allocations)      # startVal error

    print("\n\n\nComputePortfolioValue Error Tests\nTimeFrame Error Test:")
    ComputePortfolioValue(startVal, endDate, startDate, symbols, allocations)       # TimeFrame error
    
    print("\n\n\nComputePortfolioValue Error Tests\nsymbols Error Test:")
    ComputePortfolioValue(startVal, startDate, endDate, allocations, allocations)   # symbols error

    print("\n\n\nComputePortfolioValue Error Tests\nallocations Error Test:")
    ComputePortfolioValue(startVal, startDate, endDate, symbols, symbols)       # allocations error



# ComputeSharpeRatio Tests
def ComputeSharpeRatioUnitTests():
    
    print("\n\n\nComputeSharpeRatio Daily Success Tests:")
    price = HistoricalData(['XOM', 'GE', 'JPM', 'BP', 'AMZN', 'MSFT', 'AAPL', 'GOOGL'], "2020-01-01", "2021-09-09")    
    print(ComputeSharpeRatio(price, 252, "2020-01-01", "2021-09-09"))

    print("\n\n\nComputeSharpeRatio Weekly Success Tests:")    
    print(ComputeSharpeRatio(price, 52, "2020-01-01", "2021-09-09"))

    print("\n\n\nComputeSharpeRatio Monthly Success Tests:")
    print(ComputeSharpeRatio(price, 12, "2020-01-01", "2021-09-09"))

    print("\n\n\nComputeSharpeRatio Error Tests\ndfarray Error Test:")
    ComputeSharpeRatio(['XOM', 'GE', 'JPM', 'BP', 'AMZN', 'MSFT', 'AAPL', 'GOOGL'], 252, "2020-01-01", "2021-09-09")       # dfarray error
    
    print("\n\n\nComputeSharpeRatio Error Tests\nk Error Test:")
    ComputeSharpeRatio(price, 100, "2020-01-01", "2021-09-09")     # k error

    print("\n\n\nComputeSharpeRatio Error Tests\nTimeFrame Error Test:")
    ComputeSharpeRatio(price, 252, "2021-09-09", "2020-01-01")     # TimeFrame error



# OptimizePortfolio Tests
def MonteCarloPortfolioOptimizationUnitTest():
    
    print("\n\n\nOptimizePortfolio Success Tests:")
    startDate = "2020-01-01"
    endDate = "2021-09-09"
    symbols = ['XOM', 'GE', 'JPM', 'BP', 'AMZN', 'MSFT', 'AAPL', 'GOOGL']
    allocations = [0.05, 0.10, 0.05, 0.10, 0.15, 0.25, 0.15, 0.15]    

    portVal = ComputePortfolioValue(1, startDate, endDate, symbols, allocations)
    plt.plot(portVal, label="Portfolio Value", color='r')           # Regular portfolio

    optimizedAllocations = MonteCarloPortfolioOptimization(symbols, 20000, startDate, endDate)
    optimizedPort = ComputePortfolioValue(1, startDate, endDate, symbols, optimizedAllocations)
    plt.plot(optimizedPort, label="Optimized Portfolio", color='g')     # Monte Carlo simulation portfolio

    plt.legend(loc="best")
    plt.show()                      # Success
    
    

# ScipyOptimizePortfolio Tests
def ScipyOptimizePortfolioUnitTests():
    
    print("\n\n\nScipyOptimizePortfolio Success Tests:")
    startDate = "2020-01-01"
    endDate = "2021-09-09"
    symbols = ['XOM', 'GE', 'JPM', 'BP', 'AMZN', 'MSFT', 'AAPL', 'GOOGL']
    allocations = [0.05, 0.10, 0.05, 0.10, 0.15, 0.25, 0.15, 0.15]    

    portVal = ComputePortfolioValue(1, startDate, endDate, symbols, allocations)
    plt.plot(portVal, label="Portfolio Value", color='r')           # Regular portfolio

    ScipyAllocations = ScipyOptimizePortfolio(symbols, startDate, endDate)
    scipyVal = ComputePortfolioValue(1, startDate, endDate, symbols, ScipyAllocations)
    plt.plot(scipyVal, label="Scipy Portfolio", color='b')      # Optimize function portfolio

    plt.legend(loc="best")
    plt.show()                      # Success




""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """




# *********************************************************************************************************************************************************** #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------- ML4T EXAMPLES (L 7, 9, 10) -------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# *********************************************************************************************************************************************************** #


""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------------------- PROFESSOR REQUESTED EXAMPLE -------------------------------------------------------------- """
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """

def ML4TProfessorExample():
    
    startDate = "2009-01-01"
    endDate = "2009-12-31"
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.25, 0.25, 0.25, 0.25]

    # SPY index graph
    spyData = HistoricalData(['SPY'], startDate, endDate)
    normSpyData = NormalizeDfs(spyData)
    plt.plot(normSpyData[0], label = "SPY", color = 'g')

    # Previous function (Monte Carlo Simulation)
    prevOptFuncAlloc = MonteCarloPortfolioOptimization(symbols, 20000, startDate, endDate)
    prevOptPortVal = ComputePortfolioValue(1, startDate, endDate, symbols, prevOptFuncAlloc)
    plt.plot(prevOptPortVal, label = "Previous Function", color = 'r')

    # New function (Scipy Optimize)
    newOptFuncAlloc = ScipyOptimizePortfolio(symbols, startDate, endDate)
    newOptPortVal = ComputePortfolioValue(1, startDate, endDate, symbols, newOptFuncAlloc)
    plt.plot(newOptPortVal, label = "New Function", color = 'b')

    # Display labels
    plt.legend(loc = "best")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()


def ProfessorComparisonExample():

    symbols = ['GLD', 'GOOG', 'SPY', 'XOM']
    start = '2009-01-01'
    end = '2011-12-31'
    alloc = [0.4, 0.4, 0.1, 0.1]
    startValue = 1000000

    # Calculate and print portfolio value.
    portfolio = ComputePortfolioValue(startValue, start, end, symbols, alloc)
    print(portfolio)

    # Calculate and print the historical data for the individual stocks within the portfolio.
    combPortData = CombineDfs(HistoricalData(symbols, start, end), start, end)
    print("\n\nPortfolio Data\n", combPortData)

    # Cumulative returns plot of portfolio.
    portCumulRet = CumulativeReturns(portfolio, 'daily')
    PlotData(portCumulRet, "Cumulative Returns", "Date", "Cumulative Return")

    # Print daily return of portfolio.
    portDailyRet = StockReturns(portfolio, 'daily')[0]
    print(portDailyRet)

    # Print cumulative return of portflio.
    print("\nCumulative Return = {}".format(portCumulRet[0].iloc[-1,0] - 1))

    # Print average daily return of portfolio.
    print("Portfolio Average Daily Return = {}".format(portDailyRet.mean()[0]))

    # Print standard deviation of the daily returns for the portfolio.
    print("Portfolio Daily Returns Standard Deviation = {}".format(portDailyRet.std()[0]))

    # Print last three values and sharpe ratio for portfolio value.
    portSr = ComputeSharpeRatio(portfolio, 252, start, end)[0].split("    ")[1]
    print("The last three values plus the portfolio Sharpe Ratio: {}, {}, {}, {}\n\n".format(portCumulRet[0].iloc[-1,0] - 1,
                                                                                        portDailyRet.mean()[0],
                                                                                        portDailyRet.std()[0],
                                                                                        portSr))



""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ----------------------------------------------------------------- LESSON SEVEN EXAMPLES ----------------------------------------------------------------- """
""" --------------------------------------------------------------------------------------------------------------------------------------------------------- """

# SPY stock graph from 2009 tp 2013   /////    Video: #5 How to plot a histogram.  /////   TimeStamp: 0:51 
def SevenExampleOne():
    PlotData(HistoricalData(['SPY'], "2009-01-01", "2012-12-31"), "Stock Prices", "Date", "Price")


# Daily returns graph for SPY.    /////   Video: #5 How to plot a hisogram.   /////   TimeStamp: 0:55  
def SevenExampleTwo():
    PlotData(StockReturns(HistoricalData(['SPY'], "2009-01-01", "2012-12-31"), "daily"), "Daily Returns", "Date", "Daily Returns")


# Daily returns histogram for SPY with 20 bins.    /////   Video: #5 How to plot a histogram.  /////   TimeStamp: 1:14 
def SevenExampleThree():

    PlotHistogram(StockReturns(HistoricalData(['SPY'], "2009-01-01", "2012-12-31"), "daily"), "No", 20)


# Daily returns hist for SPY with statistics.     /////   Video: #6 Computing histogram statistics.   /////   TimeStamp: 2:00 
def SevenExampleFour():
    PlotHistogram(StockReturns(HistoricalData(['SPY'], "2009-01-01", "2012-12-31"), "daily"), "Yes", 20)


# Comparable daily returns histogram for SPY and XOM.     /////    Video: #8 Plot two histograms together.    /////   TimeStamp: 1:13 
def SevenExampleFive():
    PlotHistogram(StockReturns(HistoricalData(['SPY', 'XOM'], "2009-01-01", "2012-12-31"), "daily"), "no", 20)


# Scatter plot of daily returns of SPY vs XOM/GLD.    /////   Video: #13 Scatterplots in python.      /////   TimeStamp: 3:36 and 4:25 
def SevenExampleSix():
    PlotScatter(CombineDfs(StockReturns(HistoricalData(['SPY', 'XOM'], "2009-01-01", "2012-12-31"), 'daily'), "2009-01-01", "2012-12-31"))

    PlotScatter(CombineDfs(StockReturns(HistoricalData(['SPY', 'GLD'], "2009-01-01", "2012-12-31"), 'daily'), "2009-01-01", "2012-12-31"))


# Stock graph of SPY, XOM, and GLD.   /////   Video: #13 Scatterplots in python.      /////   TimeStamp: 3:41 
def SevenExampleSeven():
    PlotData(HistoricalData(['SPY', 'XOM', 'GLD'], "2009-01-01", "2012-12-31"), "Stock Prices", "Date", "Price")


# Correlation matrix of SPY, XOM, and GLD.    /////   Video: #13 Scatterplots in python.  /////   TimeStamp: 4:20
def SevenExampleEight():        
    PlotCorrelationMatrix(CombineDfs(StockReturns(HistoricalData(['SPY', 'XOM', 'GLD'], "2009-01-01", "2012-12-31"), "daily"), "2009-01-01", "2012-12-31"))



""" -------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ----------------------------------------------------------------- LESSON NINE EXAMPLES ----------------------------------------------------------------- """
""" -------------------------------------------------------------------------------------------------------------------------------------------------------- """

# Minimize example.     /////   Video: #3 Minimizer in python.  /////   TimeStamp: 2:13 
def NineExampleOne():
    
    def f(X):
        # Given a scalar X, return some value (a real number).
        Y = (X - 1.5) ** 2 + 0.5
        # For tracing.
        print("X = {}, Y = {}".format(X,Y))     
        return Y
    
    # Initial guess.
    Xguess = 2.0
    minResult = spo.minimize(f, Xguess, method="SLSQP", options={'disp':True})

    print("Minima Found At: ")
    print("X = {}, Y = {}".format(minResult.x, minResult.fun))


# Minimize example plotted.     /////   Video: #3 Minimizer in python.    /////   TimeStamp: 3:04
def NineExampleTwo():
    
    def f(X):
        # Given a scalar X, return some value (a real number).
        Y = (X - 1.5) ** 2 + 0.5
        # For tracing.
        print("X = {}, Y = {}".format(X,Y))     
        return Y
    
    # Initial guess.
    Xguess = 2.0
    minResult = spo.minimize(f, Xguess, method="SLSQP", options={'disp':True})

    print("Minima Found At: ")
    print("X = {}, Y = {}".format(minResult.x, minResult.fun))

    Xplot = np.linspace(0.5, 2.5, 21)
    Yplot = f(Xplot)
    plt.plot(Xplot, Yplot)
    plt.plot(minResult.x, minResult.fun, 'ro')
    plt.title("Minima of an objective function")
    plt.show()
    

# Fitting a line to a given set of data points using optimization.      /////   Video: #9 Fit a line to given data points.      /////   TimeStamp: 5:17
def NineExampleThree():

    # Computes error between given line model and observed data.
    def Error(line, data):
        err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
        return err

    # Fit a line to given data.
    def FitLine(data, errorFunc):
        # Initial guess with m=0 and intercept is mean of y values.
        l = np.float32([0, np.mean(data[:, 1])])

        # Plot initial guess.
        x_ends = np.float32([-5, 5])
        plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth = 2.0, label = "Initial guess")

        # Optimize function to minimize error function.
        result = spo.minimize(errorFunc, l, args=(data), method='SLSQP', options={'disp':True})
        return result.x

    # Original line
    l_orig = np.float32([4, 2])
    print("Orignal Line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1]))
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]

    plt.plot(Xorig, Yorig, 'b--', linewidth = 2.0, label = "Original line")

    # Generate random noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label = "Data points")

    # Try to fit a line to this data
    l_fit = FitLine(data, Error)
    print("Fitted Line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1]))
    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', linewidth = 2.0, label = "Fitted line")

    plt.legend(loc="best")
    plt.show()


# Fitting a polynomial line using optimization.     /////   Video: #10 And it works for polynomials too!    /////   TimeStamp: 0:00
def NineExampleFour():

    # Computes error between polynomial and observed data.
    def ErrorPoly(C, data):
        err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
        return err

    # Fits a polynomial to given data.
    def FitPoly(data, errorFunc, degree = 3):

        # Initial guess.
        Cguess = np.poly1d(np.ones(degree + 1, dtype = np.float32))

        # Plot initial guess.
        x = np.linspace(-5, 5)
        plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth = 2.0, label = "Initial guess")

        # Optimizer function.
        result = spo.minimize(errorFunc, Cguess, args=(data,), method='SLSQP', options={'disp':True})
        return np.poly1d(result.x)

    # Original line
    p_orig = np.poly1d([1.5, -10, -5, 60, 50])
    print("\nOrignal Line: C0 = {}, C1 = {}, C2 = {}, C3 = {}, C4 = {}".format(p_orig[4], p_orig[3], p_orig[2], p_orig[1], p_orig[0]))
    Xorig = np.linspace(-5, 5, 25)
    Yorig = (p_orig[4] * (Xorig ** 4)) + (p_orig[3] * (Xorig ** 3)) + (p_orig[2] * (Xorig ** 2)) + (p_orig[1] * Xorig) + p_orig[0]

    plt.plot(Xorig, Yorig, 'b--', linewidth = 2.0, alpha = 0.6, label = "Original line")

    # Generate random noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label = "Data points")

    # Try to fit a line to this data
    p_fit = FitPoly(data, ErrorPoly, degree=4)
    print("Fitted Line: C0 = {}, C1 = {}, C2 = {}, C3 = {}, C4 = {}".format(p_fit[4], p_fit[3], p_fit[2], p_fit[1], p_fit[0]))

    p_graph = (p_fit[4] * (data[:, 0] ** 4)) + (p_fit[3] * (data[:,0] ** 3)) + (p_fit[2] * (data[:,0] ** 2)) + (p_fit[1] * (data[:,0])) + p_fit[0]
    plt.plot(data[:, 0], p_graph, 'r--', linewidth = 1.5, alpha=0.7, label = "Fitted line")


    plt.legend(loc="best")
    plt.show()
    


""" -------------------------------------------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------------------------ LESSON TEN EXAMPLES ----------------------------------------------------------------- """
""" -------------------------------------------------------------------------------------------------------------------------------------------------------- """

# Graph of GOOG, AAPL, GLD, and XOM.     /////   Video: #2 The difference optimization can make.     /////   TimeStamp: 0:05
def TenExampleOne():
    port = ComputePortfolioValue(1, "2010-01-01", "2010-12-31", ['GOOGL', 'AAPL', 'GLD', 'XOM'], [0.25, 0.25, 0.25, 0.25])
    spyNorm = NormalizeDfs(HistoricalData(['SPY'], "2010-01-01", "2010-12-31"))
    PlotData([spyNorm[0], port], "Daily Portfolio Value and SPY", "Dates", "Price")


# Graph of optimized GOOG, AAPL, GLD, and XOM.      /////   Video: #2 The difference optimization can make.     ////    TimeStamp: 0:48
def TenExampleTwo():
    port = ComputePortfolioValue(1, "2010-01-01", "2010-12-31", ['GOOGL', 'AAPL', 'GLD', 'XOM'], [0.00, 0.40, 0.60, 0.00])
    spyNorm = NormalizeDfs(HistoricalData(['SPY'], "2010-01-01", "2010-12-31"))
    PlotData([spyNorm[0], port], "Daily Portfolio Value and SPY", "Dates", "Price")


# Optimized portfolio of GOOG, AAPL, GLD, and XOM.      
def TenExampleThree():

    # SPY index graph
    plt.plot(NormalizeDfs(HistoricalData(['SPY'], "2009-01-01", "2009-12-31"))[0], label = "SPY", color = 'g')

    # New function (Scipy Optimize)
    newOptFuncAlloc = ScipyOptimizePortfolio(['GOOG', 'AAPL', 'GLD', 'XOM'], "2009-01-01", "2009-12-31")
    newOptPortVal = ComputePortfolioValue(1, "2009-01-01", "2009-12-31", ['GOOG', 'AAPL', 'GLD', 'XOM'], newOptFuncAlloc)
    plt.plot(newOptPortVal, label = "Scipy Optimized", color = 'b')

    # Display labels
    plt.legend(loc = "best")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()

