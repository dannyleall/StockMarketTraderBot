import numpy as np
from Utilities import *
from finta import TA
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import random 

""" ------------------------------------------------------------------------------------------------------------------------- """
""" ----------------------------------------------- Utilities for TestLearners.ipynb ---------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

# Trains, tests, and calculate RMSEs and correlations for learners.
def TrainTestLearner(trainX, trainY, testX, testY, learnerArgs, 
                    iterations=1, maxLeafSize=None, maxBagSize=None, verbose=False, **kwargs):
    
    """
    Description: This function serves to train and test a learner.

    Params:
        trainX (np.Array): Training data for X.
        trainY (np.Array): Training data for Y.
        testX (np.Array): Testing data for X.
        testY (np.Array): Testing data for Y.
        learnerArgs (Object): Learner being used (i.e., DTLearner)
        iterations (int): Times data is trained and tested.
        maxLeafSize (int): Max leaf size range for training tree learner.
        maxBagSize (int): Max value for bag size range when training bag learner.
        **kwargs: Additional arguments for learner constructors.

    Returns: RMSEs and Correlations for training and testing data.
    """

    # Make sure leag size and/or bag size is not none.
    if maxLeafSize is None and maxBagSize is None:
        print("\nmaxLeafSize and/or maxBagSize cannot be none.")
        print("Fake data with zeros will be used for now.")
        return np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))

    maxVal = maxLeafSize or maxBagSize

    # Initialize two NDArrays for training and testing RMSEs.
    trainingRMSE = np.zeros((maxVal, iterations))
    testingRMSE = np.zeros((maxVal, iterations))

    # Initialize two NDArrays for training and testing correlations.
    trainingCorr = np.zeros((maxVal, iterations))
    testingCorr = np.zeros((maxVal, iterations))

    # Train learner and record RMSEs.
    for i in range(1, maxVal):
        for j in range(iterations):
            
            if verbose:
                print("Leaf Number {} of {}, Iteration {} of {}"
                .format(i, maxVal, j, iterations))
            # Create a learner and train it.
            if maxLeafSize is not None:
                if learnerArgs == LinRegLearner() or learnerArgs == KNearestNeighborLearner():
                    Learner = learnerArgs(**kwargs)
                
                else:
                    Learner = learnerArgs(leafSize=i, **kwargs)
            
            elif maxBagSize is not None:
                Learner = learnerArgs(bags=i, **kwargs)
            
            # Train learner.
            Learner.AddEvidence(trainX, trainY)

            # Evaluate train data.
            predY = Learner.Query(trainX)
            
            RMSE = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
            trainingRMSE[i, j] = RMSE
            CORR = np.corrcoef(predY, y=trainY)
            trainingCorr[i,j] = CORR[0,1]

            # Evaluate test data.
            predY = Learner.Query(testX)
            
            RMSE = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
            testingRMSE[i,j] = RMSE
            CORR = np.corrcoef(predY, y=testY)
            testingCorr[i,j] = CORR[0,1]

    # Retrieve the average RMSEs from all iterations.
    trainRMSEMean = np.mean(trainingRMSE, axis=1)
    testingRMSEMean = np.mean(testingRMSE, axis=1)

    # Retrieve the medians of correlations from all iterations.
    trainCorrMedian = np.median(trainingCorr, axis=1)
    testCorrMedian = np.median(testingCorr, axis=1)

    return trainRMSEMean, testingRMSEMean, trainCorrMedian, testCorrMedian


# *********************************************************************************************************************************************************** #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------ Algorithmic Trading Using Technical Indicators ----------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# *********************************************************************************************************************************************************** #


""" ------------------------------------------------------------------------------------------------------------------------- """
""" ---------------------------------------------------- Bollinger Bands Learner -------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

class BollingerBandsLearner(object):

    # Constructor.    
    def __init__(self):
        pass


    # Adjust dataframe for bollinger bands trading.
    def BollingerBandsDf(self, ticker, startDate, endDate, window=8, stdVal=1.3):

        tickerDf = IndividualHistoricalData(ticker, startDate, endDate, 'Yes')
        bbUpper = tickerDf['Close'].rolling(window).mean() + tickerDf['Close'].rolling(window).std() * stdVal
        bbLower = tickerDf['Close'].rolling(window).mean() - tickerDf['Close'].rolling(window).std() * stdVal

        tickerDf['SMA'] = TA.SMA(tickerDf, window)
        tickerDf['BBU'] = bbUpper
        tickerDf['BBL'] = bbLower    

        return tickerDf


    # Acquire sell prices and dates based on bollinger bands.
    def BbSellPricesAndDates(self, checkDf):

        # Drop na to avoid confusing algorithm.
        df = checkDf.dropna()
        sellPrice, sellDate = [], []
        overBB = None
        for index in range(len(df)):

            # If stock value goes outside of bollinger bands, make var True.
            if (df.iloc[index, 0] or df.iloc[index, 3]) > df.iloc[index, 6]:
                overBB = True
            
            else: overBB is False

            # If the value comes back in from being above BB, sell it.
            if overBB is True and (df.iloc[index, 0] < df.iloc[index, 6]):
                overBB = False
                sellPrice.append(df.iloc[index, 3])
                sellDate.append(df.index[index])

            elif overBB is True and (df.iloc[index, 3] < df.iloc[index, 6]):
                overBB = False
                sellPrice.append(df.iloc[index + 1, 0])
                sellDate.append(df.index[index + 1])            

        return sellPrice, sellDate


    # Acquire buy prices and dates based on bollinger bands.
    def BbBuyPricesAndDates(self, checkDf):
        
        # Drop na to avoid confusing algorithm.
        df = checkDf.dropna()
        buyPrice, buyDate = [], []
        underBB = None
        for index in range(len(df)):

            # Check if value is underneath BB.
            if (df.iloc[index, 0] or df.iloc[index, 3]) < df.iloc[index, 7]:
                underBB = True
            
            else: underBB is False

            # If value re enters the BB, buy it.
            if underBB is True and (df.iloc[index, 0] > df.iloc[index, 6]):
                underBB = False
                buyPrice.append(df.iloc[index, 3])
                buyDate.append(df.index[index])

            elif underBB is True and (df.iloc[index, 3] > df.iloc[index, 6]):
                underBB = False

                try:
                    buyPrice.append(df.iloc[index + 1, 0])
                    buyDate.append(df.index[index + 1])
                except:
                    break            

        return buyPrice, buyDate


    # Format time for aesthetic purposes.
    def FormatTime(self, time):
        return str(time).split(' ')[0]


    # Calculate the trading results.
    def BbTradeResults(self, tickerDf, ticker):    

        # Make first row the first value.
        startVal = tickerDf.iloc[0, 0]

        # Get sell and buy prices and dates.
        sp, sd = self.BbSellPricesAndDates(tickerDf)
        totSp = sum(sp)
        
        bp, bd = self.BbBuyPricesAndDates(tickerDf)
        totBp = sum(bp)    

        # Compute transaction fees.
        totTranFee = (totSp + totBp) * 0.12

        # Calculate total shares needed along with start value.
        totalShares = len(sp)
        totalValue = totalShares * startVal

        # Calculate profit after all transactions and performance.
        finalValue = totalValue - totBp + totSp - totTranFee
        profit = ((finalValue/totalValue - 1)*100)
        profit = round(profit, 2)

        print("\n\nThe total amount of sells: {}\n"
                "The total amount of buys: {}\n\n"
                    "After transaction fees of about 12%, considering your portfolio had {} total shares of {} to\n" 
                    "invest from {} to {}, my algorithm could have made you profitable by {}%\n\n"
                        .format(len(sp), 
                                len(bp), 
                                totalShares,
                                ticker,    
                                self.FormatTime(tickerDf.index[0]), 
                                self.FormatTime(tickerDf.index[len(tickerDf) - 1]),
                                profit))


    # Plot the trading results.
    def BbVisualizeTrades(self, tickerDf, ticker):

        bp, bd = self.BbBuyPricesAndDates(tickerDf)
        sp, sd = self.BbSellPricesAndDates(tickerDf)

        up = tickerDf[tickerDf.Close >= tickerDf.Open]
        down = tickerDf[tickerDf.Close < tickerDf.Open]

        # Plot a candelstick graph.
        plt.figure()
        plt.bar(up.index, up.Close - up.Open, 1, bottom=up.Open, color='black')
        plt.bar(up.index, up.High - up.Close, 0.25, bottom=up.Close, color="black")
        plt.bar(up.index, up.Low - up.Open, 0.25, bottom=up.Open, color="black")

        # Plot the regulat stock graph in there as well.
        plt.plot(tickerDf['Close'], label=ticker, color='purple', linestyle='dashed')
        plt.bar(down.index, down.Close - down.Open, 1, bottom=down.Open, color='steelblue')
        plt.bar(down.index, down.High - down.Open, 0.25, bottom=down.Open, color='steelblue')
        plt.bar(down.index, down.Low - down.Close, 0.25, bottom=down.Close, color='steelblue')

        # Plot buy and sell datapoints in the graph as well.
        plt.xticks(rotation=45, ha='right')
        plt.scatter(bd, bp, label='BUY', marker='^', color='Green', s=70)
        plt.scatter(sd, sp, label='SELL', marker='v', color='Red', s=70)
        plt.legend(loc='best')
        plt.show()


    # Perform all operations given a time range and stock.
    def StockTradeBb(self,
                ticker='GOOGL', 
                startDate='2022-01-01', 
                endDate=datetime.today().strftime('%Y-%m-%d'),
                window=8, 
                stdVal=1.3):

        tickerDf = self.BollingerBandsDf(ticker, startDate, endDate, window, stdVal)
        
        self.BbTradeResults(tickerDf, ticker)

        self.BbVisualizeTrades(tickerDf, ticker)




# *********************************************************************************************************************************************************** #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------ Machine Learning Algorithm Learners ---------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# *********************************************************************************************************************************************************** #


""" ------------------------------------------------------------------------------------------------------------------------- """
""" --------------------------------------------------- Linear Regression Learner ------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

class LinRegLearner(object):

    def __init__(self, verbose=False):
        """
        Description: This is the constructor for the LinRegLearner class
        that simply initializes a Linear Regression Learner.

        Params:
            verbose (bool): Print process or not.

        Returns: Initializes variables.
        """
        
        self.modelCoefficients = None
        self.residuals = None
        self.rank = None
        self.s = None
        self.verbose = verbose

        if verbose:
            print("Initialization Complete.")
            self.GetLearnerInfo()


    def AddEvidence(self, X, Y):
        """
        Description: This function trains a linear regression learner
        when given training dataframes X and Y.

        Params:
            X (pd.DataFrame): Dataframe X.
            Y (pd.DataFrame): Dataframe Y.

        Returns: A trained model and its variables.
        """

        # Add a column of 1s so that linear regression finds a constant term.
        newX = np.ones([X.shape[0], X.shape[1] + 1])
        newX[:, 0:X.shape[1]] = X

        # Build and save model.
        self.modelCoefficients, self.residuals, self.rank, self.s = np.linalg.lstsq(newX, Y)

        if self.verbose:
            print("Post Linear Regression Training")
            self.GetLearnerInfo()

    
    def Query(self, points):
        """
        Description: This function tests the learner that was trained by estimating a set
        of test points given the model we built before.

        Params:
            points(np.Array): Represents row queries.

        Returns: Estimated values according to trained model.
        """

        # Predict the models performance.
        return (self.modelCoefficients[:-1] * points).sum(axis=1) + self.modelCoefficients[-1]


    def GetLearnerInfo(self):
        """
        Description: This function serves to simply print out data from the learner.
        """
        print("Model Coefficient Matrix: ", self.modelCoefficients,
              "\nSums of Residuals: ", self.residuals, "\n")



""" ------------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------- K-Nearest Neighbor Learner ------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """


class KNearestNeighborLearner(object):

    def __init__(self, K=4, verbose=False):
        """
        Description: This is the constructor for the KNearestNeighborLearner class
        that simply initializes a k nearest neighbor Learner.

        Params:
            K (int): K nearest neighbors value.
            verbose (bool): Print process or not.

        Returns: Initializes variables.
        """
        
        self.K = K
        self.verbose = verbose


    def AddEvidence(self, X, Y):
        """
        Description: This function trains a knn learner
        when given training dataframes X and Y.

        Params:
            X (pd.DataFrame): Dataframe X.
            Y (pd.DataFrame): Dataframe Y.

        Returns: Variables designated to their respective class variables.
        """

        # Split into training and testing.
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=12345)
        
        # Create instance of a KNR but just to compare with Prof.
        model = KNeighborsRegressor(n_neighbors=self.K)
        model.fit(xTrain, yTrain)

        # Predictions and RMSEs.
        xTrainPred = model.predict(xTrain)
        trainRmse = sqrt(mean_squared_error(yTrain, xTrainPred))
        xTestPred = model.predict(xTest)
        testRmse = sqrt(mean_squared_error(yTest, xTestPred))

        # Hyper parameterize.
        hp = dict(n_neighbors=list(range(1,10)),
                  weights=['uniform', 'distance'])

        self.model = GridSearchCV(KNeighborsRegressor(), hp)
        self.model.fit(xTrain, yTrain)

        return xTrain, xTrainPred, xTest, xTestPred, trainRmse, testRmse


    def Query(self, xTest):
        """
        Description: This function tests the learner that was trained by estimating a set
        of test points given the model we built before.

        Params:
            points(np.Array): Represents row queries.

        Returns: Estimated values according to trained model.
        """
        yTest = self.model.predict(xTest)
        return yTest



""" ------------------------------------------------------------------------------------------------------------------------- """
""" ----------------------------------------------------- Decision Tree Learner --------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """


class DecisionTreeLearner(object):

    def __init__(self, leafSize=1, verbose=False, tree=None):
        """
        Description: This function serves to initialize a Decision Tree Learner
        and all its respective variables.

        Params:
            leafSize (int): Maximum number of samples to be aggregated to a leaf.
            verbose (bool): Print process or not.
            tree (np.Array): Number of trees.

        Returns: Initialized variables.
        """

        self.leafSize = leafSize
        self.verbose = verbose
        self.tree = tree

        if verbose:
            print("Initialization complete.")
            self.GetLearnerInfo()

        
    def BuildTree(self, X, Y, leafSize):
        """
        Description: This function builds a decision tree using recursion by choosing the
        best column feature to split along with best value to split. Usually the best 
        feature has the highest correlation with data Y. If they are all the same however, 
        then select the first feature. Typically, the best value to split is based on the median 
        of the data according to its best determined feature.

        Params:
            X (np.Array): X values at every decision tree node.
            Y (np.1DArray): Y values at every decision tree node.

        Returns: A numpy NDArray that represents a tree. 
        """

        numOfSamp, numOfFeat = X.shape

        if numOfSamp <= leafSize or len(set(Y)) == 1:
            return [["Leaf", Y.mean(), None, None]]

        bestFeat = None
        maxInfoGain = float('-inf')

        for i in range(numOfFeat):
            iter = X[:, i]
            splitVal = np.median(iter)

            mask = iter <= splitVal
            
            a= sum(mask)
            b = mask.shape[0] - a

            if a == 0 or b == 0:
                infoGain = 0
            
            else:
                infoGain = np.var(Y) - (a / (a + b) * np.var(Y[mask])) - (b / (a + b) * np.var(Y[~mask]))

            if infoGain > maxInfoGain:
                maxInfoGain = infoGain
                bestFeat = i

        splitVal = np.mean(X[:, bestFeat])

        if maxInfoGain == 0:
            return [["Leaf", Y.mean(), None, None]]

        leftIndex = X[:, bestFeat] <= splitVal
        rightIndex = np.logical_not(leftIndex)

        leftTree = self.BuildTree(X[leftIndex], Y[leftIndex], leafSize=leafSize)
        rightTree = self.BuildTree(X[rightIndex], Y[rightIndex], leafSize=leafSize)

        root = [[bestFeat, splitVal, 1, len(leftTree) + 1]]

        return root + leftTree + rightTree


    def TreeSearch(self, point, row):
        """
        Description: This function serves to be used alongside the Predict (Query) function
        as it recursively searches the decision tree matrix.

        Params:
            point (pd.1DArray): 1D Array of test query.
            row (list): Row of decision tree matrix to search.

        Returns:  A predicted value for a given point.
        """

        # Acquire feature on row and corresponding split value.
        feature, splitVal = self.tree[row, 0:2]

        # If splitting value is -1, we reached leaf.
        if feature == -1:
            return splitVal

        # If corresponding feature is less than split value, go to left tree.
        elif point[int(feature)] <= splitVal:
            prediction = self.TreeSearch(point, row + int(self.tree[row, 2]))

        # Otherwise, go to right tree.
        else:
            prediction = self.TreeSearch(point, row + int(self.tree[row, 3]))

        return prediction


    def AddEvidence(self, X, Y):
        """
        Description: This function serves to add training data to the 
        decision tree learner.

        Params:
            X (np.NDArray): X values of data to add.
            Y (np.1DArray): Y training values.

        Returns: Updated tree matrix for Decision Tree Learner.
        """

        # Build a tree based on the data.
        self.tree = self.BuildTree(X, Y, self.leafSize)


    def Query(self, points):
        """
        Description: This function serves to estimate a set of test points given
        a model we created. Basically, this is a test function for our model.

        Params:
            points (np.NDArray): Test queries.

        Returns: Predictions in a numpy 1D array of estimated values.
        """
        yPred = np.array([])

        for row in points:
            index = 0
            node = self.tree[index]

            while node[0] != "Leaf":
                fact, splitVal, left, right = node
                X = row[fact]

                if X <= splitVal:
                    index += left

                else:
                    index += right

                node = self.tree[index]

            yPred = np.append(yPred, node[1])
        
        return yPred


    def GetLearnerInfo(self):
        """
        Description: This function serves to print out the 
        data for the decision tree learner.
        """

        print("Leaf Size = ", self.leafSize)

        if self.tree is not None:
            print("Tree Shape = ", self.tree.shape)
            print("Tree as a Matrix: ")

            # Create a dataframe that is easier to visualize.
            dfTree = pd.DataFrame(
                                  self.tree,
                                  columns=['Factor', 'Split Value', 'Left Tree', 'Right Tree']
                                  )
            
            dfTree.index.name = 'Node'
            print(dfTree)

        else:
            print("Tree has no data yet.")

        print("\n")


    def DTExample(self):
        """
        Description: This is a simple decision tree application on a pre-defined dataset.
        """

        print("\nThis is a Decision Tree Learner.\n")
    
        # Data to test DecisionTreeLearner.
        X0 = np.array([0.885, 0.725, 0.560, 0.735, 0.610, 0.260, 0.500, 0.320])
        X1 = np.array([0.330, 0.390, 0.500, 0.570, 0.630, 0.630, 0.680, 0.780])
        X2 = np.array([9.100, 10.900, 9.400, 9.800, 8.400, 11.800, 10.500, 10.000])
        
        X = np.array([X0, X1, X2]).T
        Y = np.array([4.000, 5.000, 6.000, 5.000, 3.000, 8.000, 7.000, 6.000])

        # Create a decision tree learner and train using X and Y.
        DT = DecisionTreeLearner(verbose=True, leafSize=1)
        print("\nAdding data...")
        DT.AddEvidence(X, Y)

        # Create a sub tree.
        print("\nCreating another tree from existing...")
        DST = DecisionTreeLearner(tree=DT.tree)
        print("Done!")

        # Check if DST and DT have the same tree.
        print("\nChecking if DST and DT have the same tree...")
        if np.any(DT.tree == DST.tree): print("True.")

        # Print info for DST.
        print("\nPrinting sub tree information...")
        DST.GetLearnerInfo()

        # Modify DST and confirm DT is not affected.
        print("Modifying DST and confirming DT is unmodified...")
        DST.tree[0] = np.arange(DST.tree.shape[1])
        if np.any(DT.tree != DST.tree): print("True.")

        # Use query command on dummy data.
        DT.Query(np.array
                        (
                            [
                                [1, 2, 3],
                                [0.2, 12, 12]
                            ]
                        )
                )
        
        print("\nTesting a dummy dataset...")
        X2 = np.array(
                        [
                            [  0.26,    0.63,   11.8  ],
                            [  0.26,    0.63,   11.8  ],
                            [  0.32,    0.78,   10.0  ],
                            [  0.32,    0.78,   10.0  ],
                            [  0.32,    0.78,   10.0  ],
                            [  0.735,   0.57,    9.8  ],
                            [  0.26,    0.63,   11.8  ],
                            [  0.61,    0.63,    8.4  ]
                        ]
                    )
        Y2 = np.array(
                        [ 8.0,  8.0,  6.0,  6.0,  6.0,  5.0,  8.0,  3.0 ]
                    )
        DT = DecisionTreeLearner(verbose=True)
        DT.AddEvidence(X2, Y2)



""" ------------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------------ Random Tree Learner ---------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """


class RandomTreeLearner(object):

    def __init__(self, leafSize=1, verbose=False, tree=None):
        """
        Description: This function serves to initialize a Random Tree Learner
        and all its respective variables.

        Params:
            leafSize (int): Maximum number of samples to be aggregated to a leaf.
            verbose (bool): Print process or not.
            tree (np.Array): Number of trees.

        Returns: Initialized variables.
        """

        self.leafSize = leafSize
        self.verbose = verbose
        self.tree = tree

        if verbose:
            print("Initialization complete.")
            self.GetLearnerInfo()


    def BuildTree(self, X, Y, leafSize):
        """
        Description: This function builds a decision tree using recursion by choosing a random
        feature to split. Also, the splitting value is the mean of feature values of two rows.
    
        Params:
            X (np.Array): X values at every decision tree node.
            Y (np.1DArray): Y values at every decision tree node.

        Returns: A numpy NDArray that represents a tree. 
        """
        numOfSamp, numOfFeat = X.shape

        if numOfSamp <= leafSize or len(set(Y)) == 1:
            return [["Leaf", Y.mean(), None, None]]

        feat = np.random.choice(np.arange(numOfFeat))
        splitVal = np.mean(X[:, feat])

        mask = X[:, feat] <= splitVal
        
        a= sum(mask)
        b = mask.shape[0] - a
        maxInfoGain = np.var(Y) - (a / (a + b) * np.var(Y[mask])) - (b / (a + b) * np.var(Y[~mask]))

        if maxInfoGain == 0:
            return [["Leaf", Y.mean(), None, None]]

        leftIndex = X[:, feat] <= splitVal
        rightIndex = np.logical_not(leftIndex)

        leftTree = self.BuildTree(X[leftIndex], Y[leftIndex], leafSize=leafSize)
        rightTree = self.BuildTree(X[rightIndex], Y[rightIndex], leafSize=leafSize)

        root = [[feat, splitVal, 1, len(leftTree) + 1]]

        return root + leftTree + rightTree


    def TreeSearch(self, point, row):
        """
        Description: This function serves to be used alongside the Predict (Query) function
        as it recursively searches the decision tree matrix.

        Params:
            point (pd.1DArray): 1D Array of test query.
            row (list): Row of decision tree matrix to search.

        Returns:  A predicted value for a given point.
        """
        # Acquire feature on row and corresponding split value.
        feature, splitVal = self.tree[row, 0:2]

        # If splitting value is -1, we reached leaf.
        if feature == -1:
            return splitVal

        # If corresponding feature is less than split value, go to left tree.
        elif point[int(feature)] <= splitVal:
            prediction = self.TreeSearch(point, row + int(self.tree[row, 2]))

        # Otherwise, go to right tree.
        else:
            prediction = self.TreeSearch(point, row + int(self.tree[row, 3]))

        return prediction


    def AddEvidence(self, X, Y):
        """
        Description: This function serves to add training data to the 
        random tree learner.

        Params:
            X (np.NDArray): X values of data to add.
            Y (np.1DArray): Y training values.

        Returns: Updated tree matrix for Random Tree Learner.
        """
        self.tree = self.BuildTree(X, Y, self.leafSize)

        if self.verbose:
            print("Post Random Tree Training.")
            self.GetLearnerInfo()


    def Query(self, points):
        """
        Description: This function serves to estimate a set of test points given
        a model we created. Basically, this is a test function for our model.

        Params:
            points (np.NDArray): Test queries.

        Returns: Predictions in a numpy 1D array of estimated values.
        """

        yPred = np.array([])

        for row in points:
            index = 0
            node = self.tree[index]

            while node[0] != "Leaf":
                fact, splitVal, left, right = node
                X = row[fact]

                if X <= splitVal:
                    index += left

                else:
                    index += right

                node = self.tree[index]

            yPred = np.append(yPred, node[1])
        
        return yPred


    def GetLearnerInfo(self):
        """
        Description: This function serves to print out the 
        data for the random tree learner.
        """

        print("Leaf Size = ", self.leafSize)

        if self.tree is not None:
            print("Tree Shape = ", self.tree.shape)
            print("Tree as a Matrix: ")

            # Create a dataframe that is easier to visualize.
            dfTree = pd.DataFrame(
                                  self.tree,
                                  columns=['Factor', 'Split Value', 'Left Tree', 'Right Tree']
                                  )
            
            dfTree.index.name = 'Node'
            print(dfTree)

        else:
            print("Tree has no data yet.")

        print("\n")



""" ------------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------- Bootstrap Aggregating Learner ----------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """


class BootstrapAggregatingLearner(object):

    def __init__(self, learner, bags=20, boost=False,
                 verbose=False, **kwargs):

        """
        Description: This function serves to initialize a Boostrap Aggregating Learner
        and all its respective variables.

        Params:
            learner (object): LRL, DTL, or RTL.
            bags (int): Quantity of learners to be trained.
            boost (bool): Applies boosting.
            verbose (bool): Print process or not.
            **kwargs: Additional arguments.

        Returns: Initialized variables.
        """
        
        Learners = []

        # Add the amount of learners to learners array depending bag size.
        for i in range(bags):
            Learners.append(learner(**kwargs))

        self.Learners = Learners
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.kwargs = kwargs
        
        if self.verbose:
            print("Initialization complete.")
            self.GetLearnerInfo()

    
    def AddEvidence(self, X, Y):
        """
        Description: This function serves to add training data to the 
        bootstrap aggregating learner.

        Params:
            X (np.NDArray): X values of data to add.
            Y (np.1DArray): Y training values.

        Returns: Updated training data for BagLearner.
        """

        # Get the number of samples based on the shape of X data.
        numOfSamples = X.shape[0]

        # For every iteration of bag, grab a random amount of training data and train it.
        for learner in self.Learners:
            index = np.random.choice(numOfSamples, numOfSamples)

            bagX = X[index]
            bagY = Y[index]
            learner.AddEvidence(bagX, bagY)

        if self.verbose:
            print("Post Bag Learner Training.")
            self.GetLearnerInfo()


    def Query(self, points):
        """
        Description: This function serves to estimate a set of test points given
        a model we created. Basically, this is a test function for our model.

        Params:
            points (np.NDArray): Test queries.

        Returns: Predictions in a numpy 1D array of estimated values.
        """

        # Use a for loop to predict a value using the mean of all the learners for that given points.
        predictions = np.array([learner.Query(points) for learner in self.Learners])
        return np.mean(predictions, axis=0)


    def GetLearnerInfo(self):
        """
        Description: This function serves to print out the 
        data for the BagLearner.
        """
        learnerName = str(type(self.Learners[0]))[8:-2]
        print("This Boostrap Aggregating Learner is made up of "
                " {} {}.".format(self.bags, learnerName))

        print("Kwargs = {}\nBoost = {}".format(self.kwargs, self.boost))

        for i in range (1, self.bags + 1):
            print("{} #{}.\n".format(learnerName, i))
            self.Learners[i-1].GetLearnerInfo()



""" ------------------------------------------------------------------------------------------------------------------------- """
""" --------------------------------------------------------- Insane Learner ------------------------------------------------ """
""" ------------------------------------------------------------------------------------------------------------------------- """


class InsaneLearner(object):

    def __init__(self, learner=LinRegLearner, iterations=20, verbose=False, **kwargs):
        """
        Description: This function serves to initialize an InsaneLearner
        and all its respective variables.

        Params:
            learner (object): KNNL, LRL, DTL, or RTL.
            verbose (bool): Print process or not.
            **kwargs: Additional arguments.

        Returns: Initialized variables.
        """

        self.verbose = verbose
        self.learners = [BootstrapAggregatingLearner(learner=learner(**kwargs))] * iterations


    def AddEvidence(self, X, Y):
        """
        Description: This function serves to add training data to the 
        insane learner.

        Params:
            X (np.NDArray): X values of data to add.
            Y (np.1DArray): Y training values.

        Returns: Updated training data for insane learner.
        """

        for learner in self.learners:
            learner.AddEvidence(X, Y)


    def Query(self, points):
        """
        Description: This function serves to estimate a set of test points given
        a model we created. Basically, this is a test function for our model.

        Params:
            points (np.NDArray): Test queries.

        Returns: Predictions in a numpy 1D array of estimated values.
        """

        results = []
        
        for learner in self.learners:
            results.append(learner.Query(points))
        
        results = np.mean(np.array(results), axis=0)

        return results




# *********************************************************************************************************************************************************** #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------- Reinforcement Learning Algorithm Learners ------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# *********************************************************************************************************************************************************** #


""" ------------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------------------ Q Learner -------------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

class QLearner(object):

    def __init__(self, numOfStates=100, numOfActions=4, 
                 alpha=0.2, gamma=0.9, rar=0.5, radr=0.99,
                 dyna=0, verbose=False):
        """
        Description: This function serves as the constructor for a Dyna QLearner instance.
        Params:
            numOfStates (int): Number of states within a Q Table.
            numOfActions (int): Number of actions within a Q Table.
            alpha (float): Value for learning rate.
            gamma (float): Value of future reward.
            rar (float): Random action rate (Probability of selection a random action at each step).
            radr (float): Random action decay rate (After each update, rar = rar * radr).
            dyna (int): Number of dyna updates.
            verbose (bool): Display info or not.
        Returns: Initialized variables.
        """
        # Declare number of actions and states for Q table.
        self.numOfStates = numOfStates
        self.numOfActions = numOfActions

        # Initialize the Q table.
        self.state = 0
        self.action = 0
        self.reward = (np.ones([self.numOfStates, self.numOfActions]) * -1.0)
        self.Q = (np.random.uniform(-1.0, 1.0, size=(numOfStates, numOfActions)))
        
        # Declare the other intagibles.
        self.alpha = alpha
        self.gamma = gamma
        self.rar= rar
        self.radr = radr

        # Initialize Dyna variables.
        self.thetaCount = (np.ones([self.numOfStates, self.numOfActions, self.numOfStates]) * 0.00001)
        self.theta = (np.zeros([self.numOfStates, self.numOfActions, self.numOfStates]))
        self.dyna = dyna

        self.verbose = verbose


    def SetState(self, state):
        """
        Description: This function updates the state before updating the entire table.
        Params: 
            state (float): The new state to be set.
        Returns: The selected action.
        """
        # Set the state.
        self.state = state

        # Create a random number.
        rand = np.random.random(1)

        # Choose random action or choose the best action depending on table decay radr.
        if rand > self.rar:
            # Choose action based on Q table and state.
            action = np.argmax(self.Q[self.state])

        else:
            # Choose a random action.
            action = random.randint(0, self.numOfActions - 1)

        self.action = action

        if self.verbose:
            print("\nState = {}, Action = {}".format(self.state, self.action))
        
        return action


    def UpdateTable(self, statePrime, reward):
        """
        Description: This function serves to update the Q table and select the best action.
        Params:

        """


    def Hallucinations(self):
        pass


    def Query(self):
        pass




if __name__ == '__main__':
    
    BollingerBandsLearner().StockTradeBb('GLD')