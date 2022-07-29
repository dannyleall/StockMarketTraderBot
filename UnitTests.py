from TraderBot import *


""" 
This UnitTests.py file serves to test all functions in the TraderBot.py file.
This file will also test what message is displayed by the function of there is 
an invalid parameter.
"""


#------------------------------------- Main -------------------------------------#
def Main():
    
    """ Portfolio """
    startVal = 100
    startDate = "2020-01-01"
    endDate = "2021-09-09"
    symbols = ['AAPL', 'MSFT', 'TWTR', 'IBM', 'AMZN']
    allocations = [0.05, 0.20, 0.20, 0.50, 0.05]

    # """ IndividualHistoricalData Tests """
    # msftHistoricalData = IndividualHistoricalData('MSFT', startDate, endDate)   # Success
    # # print(msftHistoricalData)
    # # IndividualHistoricalData("ERROR", startDate, endDate)   # symbols error
    # # IndividualHistoricalData('MSFT', endDate, startDate)    # TimeRange error

    # """ HistoricalData Tests """
    price = HistoricalData(symbols, startDate, endDate)    # Success
    # # print(price)
    # # HistoricalData(symbols, endDate, startDate)     # TimeRange error
    # # HistoricalData(startVal, startDate, endDate)    # symbolsArray error

    # """ NormalizeDfs Tests"""
    normed = NormalizeDfs(price)    # Success
    # # print(normed)
    # # NormalizeDfs(symbols)   # dfArray error

    # """ CombineDfs Tests """
    # combined = CombineDfs(price, startDate, endDate)     # Success
    # # print(combined)
    # # CombineDfs(price, endDate, startDate)   # TimeRange error

    # """ StockReturns Tests """
    # dailyReturns = StockReturns(normed, "daily")    # Success daily
    # print(dailyReturns)
    # monthlyReturns = StockReturns(normed, "monthly")    # Success monthly
    # print(monthlyReturns)
    # StockReturns(allocations, "daily")      # dfArray error
    # StockReturns(normed, "ERROR")   # dailyOrMonthly error 

    # """ CumulativeReturns Tests"""
    # cumulDailyReturns = CumulativeReturns(normed, "daily")      # Success daily
    # print(cumulDailyReturns)
    # cumulMonthlyReturns = CumulativeReturns(normed, "monthly")      # Success monthly
    # print(cumulMonthlyReturns)
    # CumulativeReturns(allocations, "daily")     # dfArray error
    # CumulativeReturns(normed, "ERROR")      # dailyOrMonthly error

    """ GetMaxClose Tests """
    maxClose = GetMaxClose(price)   # Success 
    print(maxClose)
    # GetMaxClose(allocations)    # dfArray erro

    # """ SliceRow Tests """
    # slicedRow = SliceRow(price, startDate, "2009-01-31")    # Success
    # # print(slicedRow)
    # # SliceRow(allocations, startDate, endDate) # dfArray error
    # # SliceRow(price, endDate, startDate)     # TimeFrame error

    # """ SliceColumnAndRow Tests """
    # slicedCol = SliceColumn(price, ['SPY'])
    # # print(slicedCol)
    # # SliceColumn(allocations, ['SPY']) # dfArray error
    # # SliceColumn(price, 1)     # colArray error

    # """ ValidateDates Tests """
    # # ValidateDates("2021-01-01", "2021-12-31")   # Success
    # # ValidateDates("2022-01-22", "2000-01-01")   # Error

    # """ PlotData Tests """
    # # PlotData(normed, "Stock Prices", "Date", "Price")    # Success
    # # PlotData(normed, 1, "Date", "Price")    # title error
    # # PlotData(normed, "Stock Prices", 1, "Price")    # x label error
    # # PlotData(normed, "Stock Prices", "Date", 1)     # y label error

    # """ PlotRollingMean Tests """
    # # PlotRollingMean(normed, 10)     # Success
    # # PlotRollingMean(allocations, 20)    # dfArray error
    # # PlotRollingMean(normed, 10.56)  # window error

    # """ PlotBollingerBands Tests """
    # # PlotBollingerBands(normed, 15)    # Success
    # # PlotBollingerBands(allocations, 10)     # dfArray error
    # # PlotBollingerBands(normed, 10.56)   # window error

    # """ PlotHistogram Tests """
    # twoDfsTest = [dailyReturns[0], dailyReturns[2]]
    # # PlotHistogram(dailyReturns, "no", 10)      # Success with no statistics on graph
    # # PlotHistogram(twoDfsTest, "yes", 10)    # Success with statistics on graph
    # # PlotHistogram(allocations, "yes", 15)   # dfArray error
    # # PlotHistogram(dailyReturns, "ERROR", 10)    # plotStatisticsYesOrNo error
    # # PlotHistogram(dailyReturns, "yes", 10.56)   # Bin error

    # """ PlotScatter Tests """
    # # PlotScatter(CombineDfs(twoDfsTest, startDate, endDate))     # Success
    # # PlotScatter(twoDfsTest)     # CombineDfs error
    # # PlotScatter(dailyReturns)   # dfArray error

    # """ PlotCorrelationMatrix Tests """
    # # PlotCorrelationMatrix(CombineDfs(dailyReturns, startDate, endDate))     # Success 
    # # PlotCorrelationMatrix(dailyReturns)     # CombineDfs error

    # """ ComputePortfolioValue Tests """
    # portfolioValue = ComputePortfolioValue(startVal, startDate, endDate, symbols, allocations)
    # # print(portfolioValue)      # Success
    # # ComputePortfolioValue(1000.0101, startDate, endDate, symbols, allocations)      # startVal error
    # # ComputePortfolioValue(startVal, endDate, startDate, symbols, allocations)       # TimeFrame error
    # # ComputePortfolioValue(startVal, startDate, endDate, allocations, allocations)   # symbols error
    # # ComputePortfolioValue(startVal, startDate, endDate, symbols, symbols)       # allocations error

    # """ ComputeSharpeRatio Tests """
    # yearlySr = ComputeSharpeRatio(price, 252, startDate, endDate, allocations)      # Success yearly 
    # # print(yearlySr)
    # monthlySr = ComputeSharpeRatio(price, 52, startDate, endDate, allocations)      # Success monthly 
    # # print(monthlySr)
    # weeklySr = ComputeSharpeRatio(price, 12, startDate, endDate, allocations)     # Success weekly
    # # print(weeklySr)
    # # ComputeSharpeRatio(symbols, 252, startDate, endDate, allocations)       # dfarray error
    # # ComputeSharpeRatio(price, 100, startDate, endDate, allocations)     # k error
    # # ComputeSharpeRatio(price, 252, endDate, startDate, allocations)     # TimeFrame error
    # # ComputeSharpeRatio(price, 252, startDate, endDate, symbols)     # allocations error

    # """ OptimizePortfolio Tests """
    # optimizedAllocations = OptimizePortfolio(symbols, 20000, startDate, endDate)
    # portVal = ComputePortfolioValue(1, startDate, endDate, symbols, allocations)
    # improvedPort = ComputePortfolioValue(1, startDate, endDate, symbols, optimizedAllocations)
    # plt.plot(portVal, label="Portfolio Value", color='r')
    # plt.plot(improvedPort, label="Optimized Portfolio", color='g')
    # plt.legend(loc="best")
    # plt.show()                      # Success
    # # # OptimizePortfolio("ERROR", 20000, startDate, endDate)   # symbol error
    # # # OptimizePortfolio(symbols, 10.5, startDate, endDate)    # numOfSim error
    # # # OptimizePortfolio(symbols, 10, endDate, startDate)      # time frame error

    

Main()