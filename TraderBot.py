import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo 

# ---------------------------------------------- Returns Adjusted Dataframes ---------------------------------------------- #

#---------- Single Historical Data ----------#
def IndividualHistoricalData(symbol, startDate, endDate):
        
    # Validate symbol
    if type(symbol) is not str: 
        return print("\nsymbol Variable: Invalid\n\tsymbol must be a string.\n")

    # Validate Dates
    ValidateDates(startDate, endDate)

    # Returns df for data of symbol
    data = yf.Ticker(symbol)
    dataDf = data.history(period='1d', start=startDate, end=endDate)

    # Only keep row with 'close' prices
    dataDf = dataDf.iloc[: , [3]].copy()
    dataDf = dataDf.rename(columns={'Close':symbol})
    return dataDf

#-------- Historical Data --------#
def HistoricalData(symbolsArray, startDate, endDate):
    
    # Validate symbolsArray
    if type(symbolsArray) is not list:
        return print("\nsymbolsArray Variable: Invalid\n\tVariable is not of type array.\n")

    # Validate Dates
    ValidateDates(startDate, endDate)

    dfs = []

    # Returns dfArray that iterated through symbols to get data for each
    for symbol in range(len(symbolsArray)):
        # Validate symbol variable
        if type(symbolsArray[symbol]) is not str:
            return print("\nsymbols Variable: Invalid\n\tSymbol does not exist.\n")

        tempDf = IndividualHistoricalData(symbolsArray[symbol], startDate, endDate)
        dfs.append(tempDf)

    return dfs


#------ Normalize Dataframes ------#
def NormalizeDfs(dfArray):

    # Validate dfArray
    if type(dfArray) is pd.DataFrame:
        return NormalizeDfs([dfArray])

    elif type(dfArray) is not list:
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    normalizedDfs = []

    for df in range(len(dfArray)):
        tempDf = dfArray[df]

        if type(tempDf) is not pd.DataFrame:
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")

        normal = tempDf/tempDf.iloc[0,:]
        normalizedDfs.append(normal)
    
    return normalizedDfs


#------ Combine Many Dataframes ------#
def CombineDfs(dfArray, startDate, endDate):

    # Validate dfArray
    if type(dfArray) is not list:
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    # Validate dates
    ValidateDates(startDate, endDate)

    timeFrame = pd.date_range(start=startDate, end=endDate)
    # For loop iterates through dfs and combines
    finalDf = pd.DataFrame(index=timeFrame)

    for df in range(len(dfArray)):

        if type(dfArray[df]) is not pd.DataFrame:
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")

        dfTemp = dfArray[df].iloc[: , 0].copy()

        finalDf = finalDf.join(dfTemp, how='inner')

    return finalDf.dropna()


# ------ Daily/Monthly Returns ------#
def StockReturns(dfArray, dailyOrMonthly):

    # Validate dfArray
    if type(dfArray) is pd.DataFrame:
        return StockReturns([dfArray], dailyOrMonthly)
    
    elif type(dfArray) is not list:
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    returnsDf = []    

    # Validate dailyOrMonthly
    if dailyOrMonthly.lower() != "daily" and dailyOrMonthly.lower() != "monthly": 
        return print("\ndailyOrMonthly Variable: Invalid\n\tMust be 'daily' or 'monthly'.\n")

    if dailyOrMonthly.lower() == "daily": 
        
        for df in range(len(dfArray)):
            tempDf = dfArray[df]

            if type(tempDf) is not pd.DataFrame: 
                return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")

            tempDf = tempDf.pct_change()
            returnsDf.append(tempDf.dropna())

    elif dailyOrMonthly.lower() == "monthly": 

        for df in range(len(dfArray)):
            tempDf = dfArray[df]

            if type(tempDf) is not pd.DataFrame: 
                return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")
               
            tempDf = tempDf.resample('M').ffill().pct_change()
            returnsDf.append(tempDf.dropna())
    
    return returnsDf


#------ Cumulative Returns ------#
def CumulativeReturns(dfArray, dailyOrMonthly):
    
    # Validate dfArray
    if type(dfArray) is pd.DataFrame:
        return CumulativeReturns([dfArray], dailyOrMonthly)

    elif type(dfArray) is not list:
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    cumulativeDfs = []

    # Validate dailyOrMonthly
    if dailyOrMonthly.lower() != "daily" and dailyOrMonthly.lower() != "monthly": 
        return print("\ndailyOrMonthly Variable: Invalid\n\tMust be 'daily' or 'monthly'.\n")

    if dailyOrMonthly == "daily": 
        
        for df in range(len(dfArray)):
            tempDf = dfArray[df]

            if type(tempDf) is not pd.DataFrame: 
                return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")

            tempDf = (tempDf.pct_change() + 1).cumprod()
            cumulativeDfs.append(tempDf.dropna())
            
    if dailyOrMonthly == "monthly": 

        for df in range(len(dfArray)):
            tempDf = dfArray[df]

            if type(tempDf) is not pd.DataFrame: 
                return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")

            tempDf = (tempDf.resample('M').ffill().pct_change() + 1).cumprod()
            cumulativeDfs.append(tempDf.dropna())
        
    return cumulativeDfs



# ---------------------------------------------- Statistical Functions ---------------------------------------------- #

#------ Max Close ------$
def GetMaxClose(dfArray):
    
    if type(dfArray) is pd.DataFrame:
        return GetMaxClose([dfArray])
    
    elif type(dfArray) is not list: 
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    # Returns a dict with symbol and value
    maxCloseDf = {}

    for df in range(len(dfArray)):

        tempDf = dfArray[df]

        if type(tempDf) is not pd.DataFrame:
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")

        symbol = tempDf.columns[0]
        maxDf = (tempDf[symbol].max())
        
        maxCloseDf[symbol] = maxDf

    return maxCloseDf



#------------------------------------------------- Slice Dataframe functions -------------------------------------------#

def SliceRow(dfArray, startDate, endDate):
    
    # Validate dfArray
    if type(dfArray) is pd.DataFrame:
        return SliceRow([dfArray], startDate, endDate)

    elif type(dfArray) is not list:
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    # Validate startDate and endDate
    ValidateDates(startDate, endDate)

    slicedDfs = []

    for df in range(len(dfArray)):
        tempDf = dfArray[df]

        if type(tempDf) is not pd.DataFrame:
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")

        sliced = tempDf.loc[startDate:endDate]
        slicedDfs.append(sliced)
    
    return slicedDfs


def SliceColumn(dfArray, colArray):

    # Validate dfArray
    if type(dfArray) is pd.DataFrame:
        return SliceRow([dfArray], colArray)

    elif type(dfArray) is not list:
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    # Validate colArray
    if type(colArray) is str:
        return SliceColumn(dfArray, [colArray])

    elif type(colArray) is not list: 
        return print("\ncolArray Variable: Invalid\n\tcolArray must be of type list.")

    slicedDfs = []

    for df in range(len(dfArray)):
        tempDf = dfArray[df]

        if type(tempDf) is not pd.DataFrame:
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")
 
        tempDfCol = tempDf.columns.values.tolist()

        if set(colArray).issubset(set(tempDfCol)):
            sliced = tempDf[colArray]
            slicedDfs.append(sliced)

        else: 
            print("\ncolArray Variable: Invalid\n\tColumn(s) ", colArray, " is not in " + tempDf.columns[0] + " dataframe\n")

    return slicedDfs



#------------------------------------------------ Validation Functions -------------------------------------------------#

#------ Date Validation ------#
def ValidateDates(startDate, endDate):

    beg = startDate.split("-")
    fin = endDate.split("-")

    if fin[0] < beg[0]: 
        return print("\nTime Range: Invalid\n")
    
    elif fin[0] == beg[0]:

        if fin[1] < beg[1]: 
            return print("\nTime Range: Invalid\n")

        elif fin[1] == beg[1]:
            if fin[2] < beg[2]: 
                return print("\nTime Range: Invalid\n")

            elif fin[2] == beg[2]: 
                return print("\nTime Range: Invalid\n")



#-------------------------------------------------- Plot Functions -----------------------------------------------------#

#------ Dataframe Plot ------#
def PlotData(dfArray, title, x, y):
    
    # Validate dfArray variable
    if type(dfArray) is pd.DataFrame: 
        return PlotData([dfArray], title, x, y)

    elif type(dfArray) is not list: 
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    # Validate title
    if type(title) is not str:
        return print("\ntitle Variable: Invalid\n\tTitle must be of type string.\n")

    # Validate x
    if type(x) is not str:
        return print("\nx Variable: Invalid\n\tx must be of type string.\n")

    # Validate y
    if type(y) is not str:
        return print("\ny Variable: Invalid\n\ty must be of type string.\n")

    for df in range(len(dfArray)):
        tempDf = dfArray[df]
            
        if type(tempDf) is not pd.DataFrame: 
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")
            
        tempDf.fillna(method="ffill", inplace=True)
        plt.plot(tempDf, label=tempDf.columns[0])

    plt.title("Graphs of " + title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(loc='best') 

    plt.show()


#------ Rolling Mean ------# 
def PlotRollingMean(dfArray, window):

    # Validate dfArray variable
    if type(dfArray) is pd.DataFrame: 
        return PlotRollingMean([dfArray], window)

    elif type(dfArray) is not list:
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    # Validate window variable
    if type(window) is not int: 
        return print("\nWindow Variable: Invalid\n\tNot an int.\n")

    for df in range(len(dfArray)):
        tempDf = dfArray[df]
            
        if type(tempDf) is not pd.DataFrame: 
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")
            
        symbol = tempDf.columns[0]
        tempDf.fillna(method="ffill", inplace=True)
        plt.plot(tempDf, label=symbol)

        rmSymbol = tempDf.rolling(window).mean()
        plt.plot(rmSymbol, label='Rolling Mean', linestyle='dotted')

        plt.title(symbol + " Rolling Mean")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Price")
        plt.legend(loc='best') 
        plt.show()


#------ Bollinger Bands ------# 
def PlotBollingerBands(dfArray, window):
    
    # Validate dfArray
    if type(dfArray) is pd.DataFrame: 
        return PlotBollingerBands([dfArray], window)

    elif type(dfArray) is not list:
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")

    # Validate window
    if type (window) is not int: 
        return print("\nWindow Variable: Invalid\n\tNot an integer.\n")

    for df in range(len(dfArray)):  
        tempDf = dfArray[df] 

        if type(tempDf) is not pd.DataFrame: 
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")

        tempDf.fillna(method="ffill", inplace=True)
        plt.plot(tempDf, label=tempDf.columns[0], color='black', linewidth=1.0)
                
        rm = tempDf.rolling(window).mean()
        plt.plot(rm, label="Rolling Mean", linestyle='dotted', color='grey', linewidth=2.5)

        bbSymbol = tempDf.rolling(window).std()

        upperBand = rm + bbSymbol * 1.75
        plt.plot(upperBand, label="Upper BB", linestyle='dotted', color='green', linewidth=2.5)

        lowerBand = rm - bbSymbol * 1.75
        plt.plot(lowerBand, label="Lower BB", linestyle='dotted', color='red', linewidth=2.5)

        plt.title(tempDf.columns[0] + " Bollinger Bands")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(loc='best') 

        plt.show()


#------ Histogram ------# 
def PlotHistogram(dfArray, plotStatisticsYesOrNo, bin):
    
    # Validate dfArray
    if type(dfArray) is pd.DataFrame: 
        return PlotHistogram([dfArray], plotStatisticsYesOrNo, bin)

    elif type(dfArray) is not list:
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")    
        
    # Validate plotstatisticsYesOrNo
    if plotStatisticsYesOrNo.lower() != 'yes' and plotStatisticsYesOrNo.lower() != 'no':
        return print("\nplotStatisticsYesOrNo Variable: Invalid\n\t'Must be 'yes' or 'no'.\n")

    # Validate bin
    if type(bin) is not int: 
        return print("\nBin variable: Invalid\n\tBin must be an int.\n")

    # Express that you cannot plot more than 2 dataframes AND statistics because of excess lines.
    if len(dfArray) > 2: 
        print("\nBecause dfArray has more than 2 dataframes, statistics will NOT be displayed.\n")
        print("If you wish to have statistics displayed, try the method again with only 2 dataframes.\n")
        plotStatisticsYesOrNo = "no"

    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    count = 0

    for df in range(len(dfArray)):
        tempDf = dfArray[df]
         
        if type(tempDf) is not pd.DataFrame: 
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")
            
        symbol = tempDf.columns[0]

        tempDf[symbol].hist(bins=bin, alpha=0.4, label=symbol)

        mean = tempDf[symbol].mean()
        std = tempDf[symbol].std()
        kurt = tempDf[symbol].kurtosis()

        print("\n" + symbol + " Mean = ", mean, "\n" + symbol + " Std = ", std, "\n"
            + symbol + " Kurtosis = ", kurt, "\n")

        if plotStatisticsYesOrNo.lower() == 'yes':
            plt.axvline(mean, color=colors[count], linestyle='dashed', linewidth=2, label=symbol+ " Mean")
            count +=1 if count <= 7 else count == 0
            plt.axvline(std, color=colors[count], linestyle='dashed', linewidth=2, label=symbol+ " +Std")
            count +=1 if count <= 7 else count == 0
            plt.axvline(-std, color=colors[count], linestyle='dashed', linewidth=2, label=symbol+ " -Std")
            count +=1 if count <= 7 else count == 0

    plt.legend(loc='best')
    plt.show()


#------ Scatter Plot ------# 
def PlotScatter(df):        

    # Validate dfArray type and size
    if type(df) is pd.DataFrame: 
        return PlotScatter([df])

    elif type(df) is not list: 
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")    

    if len(df) == 2:
        return print("\nPlease use the 'CombineDfs' method on these two dataframes first.\n")
    
    elif len(df) == 1:
    
        if len(df[0].columns) != 2: 
            return print("\nDataframe can only have 2 columns.\n")
    
    else:
        return print("\ndfArray Variable: Invalid\n\tArray can only have two dataframes as content.\n")
        
    df = df[0]
    xSymbol = df.columns[0]
    ySymbol = df.columns[1]
    plt.scatter(df[xSymbol], df[ySymbol])
    
    plt.xlabel(xSymbol)
    plt.ylabel(ySymbol)

    beta, alpha = np.polyfit(df[xSymbol], df[ySymbol], 1)
    print("\nBeta Value for " + ySymbol + " = ", beta, "\nAlpha Value for " + ySymbol + " = ", alpha, "\n")
    plt.plot(df[xSymbol], beta * df[xSymbol] + alpha, '-', color='r')

    plt.show()


#------ Correlation Matrix Plot ------# 
def PlotCorrelationMatrix(df):

    # Validate dfArray type and size
    if type(df) is pd.DataFrame: 
        return PlotCorrelationMatrix([df])

    elif type(df) is not list: 
        return print("\ndfArray Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")    

    if type(df[0]) is not pd.DataFrame: 
        return print("\ndfArray Variable: Invalid\n\tdfArray content is not in dataframe format.\n")

    if len(df) > 1:
        return print("\ndfArray Variable: Invalid\n\tPlease use the 'CombineDfs' method on these dataframes first.\n")

    elif len(df) == 1:

        if len(df[0].columns) < 2: 
            return print("\ndfArray Variable: Invalid\n\tDataframe must have more than one column.\n")

    df = df[0]
    
    matrix = df.corr(method='pearson')
    print("\n", matrix, "\n")
    plt.matshow(matrix, cmap=plt.get_cmap('binary'))
    
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=8, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=8)
    
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title('Correlation Matrix', fontsize=12)
    plt.show()
    


#---------------------------------------------------- Important Functions ---------------------------------------------#

# Get portfolio value
def ComputePortfolioValue(startValue, startDate, endDate, symbols, allocations):

    # Validate startValue
    if type(startValue) == float:
        decLen = len(str(startValue).split(".")[1])
        
        if decLen > 2:
            return print("\nStart Value: Invalid\n\t'startValue' can only have 2 decimal places.\n")
        
    elif type(startValue) != int: 
        return print("Start Value: Invalid\n\t'startValue' must be a valid number.\n")
    
    # Validate startDate and endDate
    ValidateDates(startDate, endDate)

    # Validate Symbols array
    if type(symbols) is str or type(allocations) is int:
        return ComputePortfolioValue(startValue, startDate, endDate, [symbols], allocations)
    
    # Validate allocations array
    if type(allocations) is int: 
        return ComputePortfolioValue(startValue, startDate, endDate, symbols, [allocations])
    
    for x in range(len(allocations)):

        if type(allocations[x]) is not float:
            return print("\nallocations Variable: Invalid\n\tContents of allocation must of type float.\n")

    if len(symbols) != len(allocations):
        return print("Symbols: Invalid\n\tLength of symbols array must equal lengths of allocations array.\n")

    sum = 0

    for x in range(len(allocations)):
        sum += allocations[x]
    
    if sum < 0.98 or sum > 1.02:
        return print("Allocations: Invalid\n\tAllocations must add to 1.0.\n")

    print("Computing Portfolio Value..........")

    # Get prices
    prices = HistoricalData(symbols, startDate, endDate)

    # Normalize prices
    normed = NormalizeDfs(prices)

    # Allocations dataframe
    alloc = [normed[x] * allocations[x] for x in range(len(normed))]

    # Position values dataframe
    positionVal = [alloc[x] * startValue for x in range(len(alloc))]
    
    # Combine each position dataframe into one big dataframe
    combined = CombineDfs(positionVal, startDate, endDate)

    # Sum each row to get portfolio value per day
    portfolioVal = combined.sum(axis=1)

    print("Portfolio value successfully computed.\n")
    df = pd.DataFrame(portfolioVal, columns = ['Portfolio Value'])
    return df


# Get Sharpe Ratio
def ComputeSharpeRatio(historicalData, k, startDate, endDate, allocations):
    
    # Validate historical data
    if type(historicalData) is pd.DataFrame:
        return ComputeSharpeRatio([historicalData], k, startDate, endDate, allocations)

    elif type(historicalData) is not list:
        return print("\nhistoricalData Variable: Invalid\n\tdfArray is not a dataframe or array of dataframes.\n")
    
    # Validate k variable
    if k != 252 and k != 52 and k != 12 or type(k) is not int:
        return print("\nk Variable: Invalid\n\tk must be an int val of 252, 52, or 12.\n")

    # Validate Dates
    ValidateDates(startDate, endDate)

    # Validate allocations array
    if type(allocations) is not list:
        return print("\nAllocations Variable: Invalid\n\tVariable must be an array.\n")
    
    else:

        for x in range(len(allocations)):
        
            if type(allocations[x]) is not float:
                return print("\nAllocations Variable: Invalid\n\tArray contents must be of type float.\n")

    if len(historicalData) != len(allocations):
        return print("\nhistoricalData Variable: Invalid\n\tLength of historicalData array must equal lengths of allocations array.\n")

    sum = 0

    for x in range(len(allocations)):
        sum += allocations[x]
    
    if sum != 1.0:
        return print("\nAllocations: Invalid\n\tAllocations must add to 1.0.\n")

    sharpe = []

    for df in range(len(historicalData)):

        if type(historicalData[df]) is not pd.DataFrame:
            return print("\ndfArray Variable: Invalid\n\tContent of array is not of type dataframe.\n")

        tempDf = StockReturns(historicalData[df], "daily")
        sharpeRatio = tempDf[0].mean()/tempDf[0].std()

        sharpe.append(str(np.sqrt(k) * sharpeRatio).split("\n")[0])

    return sharpe


# Optimize Portfolio using Monte Carlo Simulation
def OptimizePortfolio(symbols, numOfSim, startDate, endDate):

    # Validate symbols
    if type(symbols) is not list:
        return print("\nsymbols Variable: Invalid\n\tsymbols variable must be of type list.\n")

    else:
        
        for x in range(len(symbols)):

            if type(symbols[x]) is not str:
                    return print("\nsymbols Variable: Invalid\n\tsymbols variable must be of type list.\n")

    # Validate numOfSim
    if type(numOfSim) is not int:
        return print("\numOfSim Variable: Invalid\n\tMust be of type integer.\n")

    # Validate timeframe
    ValidateDates(startDate, endDate)

    # HistoricalData for symbols
    price = HistoricalData(symbols, startDate, endDate)

    # Combine the list of dataframes into one dataframe
    combined = CombineDfs(price, startDate, endDate)

    # Daily returns of the dataframe
    returns = np.log(combined/combined.shift())

    # initialize len of symbols list
    n = numOfSim
    symLength = int(len(symbols))
    count = 0

    # Develop empty numpy array
    weights = np.zeros((n, symLength))
    expectedReturns = np.zeros(n)
    expectedVolatility = np.zeros(n)
    sharpeRatios = np.zeros(n)

    # Random allocations to see which is most profitable based on sr
    for i in range(n):
        # Print computing percent completed
        count += 1
        percent = count/n * 100

        if percent % 10 == 0:
            print("Optimizing Portfolio..........", percent, "% Complete")

        # fill weights array with random content
        weight = np.random.random(symLength)

        # ensure weight array adds up to 1
        weight /= weight.sum()

        weights[i] = weight
        
        # calculate the returns, voltality, and sharpe ratio of specific iteration
        expectedReturns[i] = np.sum(returns.mean() * weight) * 252
        expectedVolatility[i] = np.sqrt(np.dot(weight.T, np.dot(returns.cov() * 252, weight)))
        sharpeRatios[i] = expectedReturns[i]/expectedVolatility[i]

    # Create a list of the optimized allocations
    optimizedAllocations = []

    for x in range(len(weights[sharpeRatios.argmax()])):
        optimizedAllocations.append(float(weights[sharpeRatios.argmax()][x]))

    print("Portfolio Optimized!\n\nMaximum Sharpe Ratio of the", n, "Simulations: ", sharpeRatios.max(), 
            "\nMost Profitable Allocations are: ", optimizedAllocations, "\n\n")

    return optimizedAllocations


# Optimize Portoflio using Scipy function
def ScipyOptimizePortfolio(symbols, startDate, endDate):

    # Validate symbols
    if type(symbols) is not list:
        return print("\nsymbols Variable: Invalid\n\tsymbols variable must be of type list.\n")

    else:
        
        for x in range(len(symbols)):

            if type(symbols[x]) is not str:
                    return print("\nsymbols Variable: Invalid\n\tsymbols variable must be of type list.\n")

    # Validate timeframe
    ValidateDates(startDate, endDate)

    # Combine the dataframes
    combined = CombineDfs(HistoricalData(symbols, startDate, endDate), startDate, endDate)

    # Compute the daily log returns 
    logReturns = np.log(combined/combined.shift(1)).dropna()

    # Define the negative sharpe ratio function that will be used inside of the spo.minimize call
    def Function(weights):

        # Create a numpy array of the list of weights
        weights = np.array(weights)

        # Compute expected returns, volatility, and sharpe ratio
        expectedReturns = np.sum(logReturns.mean() * weights) * 252
        expectedVolatility = np.sqrt(np.dot(weights.T, np.dot(logReturns.cov() * 252, weights)))
        sharpeRatio = expectedReturns/expectedVolatility

        """ 
        If we want to optimize a portfolio based on sharpe ratio using the minimize function, we must use the 
        negative sharpe ratio rather than the regular sharpe ratio. By using the 'negative sharpe ratio', the function
        will now 'maximize' rather than 'minimize'.

        """

        # Convert to numpy array of the metrics, then return the negative sharpe ratio
        return np.array([expectedReturns, expectedVolatility, sharpeRatio])[2] * -1

    # Create a random numpy array for the allocations
    random = np.array(np.random.random(len(symbols)))

    # Make sure the sum of allocations is 1.0 and compute expected return
    rebalance = random / np.sum(random)
    expectedReturn = np.sum((logReturns.mean() * rebalance) * 252)

    # Constraints
    constraints = (
        # eq ensures our weights sums up to 1.0
        { 'type':'eq', 'fun':lambda x: np.sum(x) - 1}, 
        
        # ineq ensures non-negative allocations while also aiming for our expected return to ideally be 1.5 above original stock return
        {'type':'ineq', 'fun':lambda x: np.sum(logReturns.mean() * x) - (expectedReturn + (expectedReturn * 1.5))}
        )

    # Creates a tuple that allows for the allocations to be from 0 to 1
    bounds = tuple((0,1) for x in range(logReturns.shape[1]))

    # Create an intial even distribution guess to begin the minimization process
    initialGuess = len(symbols) * [1/len(symbols)]

    # Optimize using the function, guess, bounds, and constraint
    optimizedResults = spo.minimize(Function, initialGuess, method="SLSQP", bounds=bounds, constraints=constraints)

    # Initialize the optimized allocations array
    optimizedAllocations = []

    # Convert values from numpy.float64 to float
    for x in range(len(optimizedResults.x)):
        optimizedAllocations.append(float(optimizedResults.x[x]))

    return optimizedAllocations
