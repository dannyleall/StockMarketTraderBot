import numpy as np
import numpy.ma as MaskedArray
from Utilities import *
from finta import TA
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from copy import deepcopy
from collections import deque
import random as rand
import math
from finta import TA
import seaborn as sns

""" DQN Imports. """
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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

    def __init__(self, leafSize=20, verbose=False):
        """
        Description: This function serves to initialize a Decision Tree Learner
        and all its respective variables.

        Params:
            leafSize (int): Maximum number of samples to be aggregated to a leaf.
            verbose (bool): Print process or not.

        Returns: Initialized variables.
        """

        self.leafSize = leafSize
        self.verbose = verbose

        
    def BuildTree(self,data):
        """
        Description: This function builds a decision tree using recursion by choosing the
        best column feature to split along with best value to split. Usually the best 
        feature has the highest correlation with data Y. If they are all the same however, 
        then select the first feature. Typically, the best value to split is based on the median 
        of the data according to its best determined feature.

        Params:
            data (numpy.Array): The data being used to build the decision tree.

        Returns: A numpy NDArray that represents a tree. 
        """
        
        dataY = data[:, -1]

        # If there is only one row, return just a leaf with the average of y.
        if data.shape[0] <= self.leafSize or len(data.shape) == 1:
            return np.array([['leaf',
                              np.mean(dataY),
                              -1,
                              -1]])

        # Or, if all the data is the same, return null leaf.
        elif np.all(dataY == data[0, -1]):
            return np.array([['leaf',
                              data[0, -1],
                              -1,
                              -1]])

        # Otherwise, find the best feature to slit on. Based on JR Quinlan Decision
        # Tree algorithm, the best feature X should have the highest correlation to Y.
        else:
            bestDtlFeat = 0
            highestCorrelation = -1

            # For loop across all features (X values).
            for idx in range (data.shape[1] - 1):
                # Get best absolute correlation.
                correlation = MaskedArray.corrcoef(
                    MaskedArray.masked_invalid(data[:, idx]), # Mask where invalid values occur.
                    MaskedArray.masked_invalid(dataY))[0, 1] # Mask where invalid values occur.

                # Absolute value.
                correlation = abs(correlation)

                # Replace correlation if condition passes.
                if correlation > highestCorrelation:
                    highestCorrelation = correlation
                    bestDtlFeat = idx

            # Split down the middle and check its not just 2 rows.
            splitValue = np.median(data[:, bestDtlFeat], axis=0)
            if splitValue == max(data[:, bestDtlFeat]):
                return np.array([['leaf',
                                  np.mean(dataY),
                                  -1,
                                  -1]])

            # Create left tree.
            leftTree = self.BuildTree(
                data[data[:, bestDtlFeat] <= splitValue]
            )

            # Create right tree.
            rightTree = self.BuildTree(
                data[data[:, bestDtlFeat] > splitValue]
            )

            # Establish root of tree and create a decision tree from it.
            root = np.array([[bestDtlFeat, splitValue, 1, leftTree.shape[0] + 1]])
            decisionTree = np.vstack((np.vstack((root, leftTree)), rightTree))

            return decisionTree


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
        data = np.hstack((X, Y.reshape(-1, 1)))
        self.tree = self.BuildTree(data)


    def Query(self, points):
        """
        Description: This function serves to estimate a set of test points given
        a model we created. Basically, this is a test function for our model.

        Params:
            points (np.NDArray): Test queries.

        Returns: Predictions in a numpy 1D array of estimated values.
        """
        yPred = []
        root = self.tree

        for row in range(points.shape[0]):
            
            node = 0

            while root[node, 0] != 'leaf':
                idx = root[node, 0]
                splitValue = root[node, 1]

                if points[row, int(float(idx))] <= float(splitValue):
                    left = int(float(root[node, 2]))
                    node = node + left

                else:
                    right = int(float(root[node, 3]))
                    node = node + right

            result = root[node, 1]
            yPred.append(float(result))

        return np.array(yPred)



""" ------------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------------ Random Tree Learner ---------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """


class RandomTreeLearner(object):

    def __init__(self, leafSize=20, verbose=False):
        """
        Description: This function serves to initialize a Random Tree Learner
        and all its respective variables.

        Params:
            leafSize (int): Maximum number of samples to be aggregated to a leaf.
            verbose (bool): Print process or not.

        Returns: Initialized variables.
        """

        self.leafSize = leafSize
        self.verbose = verbose

        
    def BuildTree(self,data):
        """
        Description: This function builds a decision tree using recursion by choosing a
        random column feature to split along with best value to split. Usually the random 
        feature has does not perform as well as the DTL. If they are all the same however, 
        then we select the first feature. Typically, the best value to split is based on the median 
        of the data according to its best determined feature.

        Params:
            data (numpy.Array): The data being used to build the decision tree.

        Returns: A numpy NDArray that represents a tree. 
        """
        
        dataY = data[:, -1]

        # If there is only one row, return just a leaf with the average of y.
        if data.shape[0] <= self.leafSize or len(data.shape) == 1:
            return np.array([['leaf',
                              np.mean(dataY),
                              -1,
                              -1]])

        # Or, if all the data is the same, return null leaf.
        elif np.all(dataY == data[0, -1]):
            return np.array([['leaf',
                              data[0, -1],
                              -1,
                              -1]])

        # Otherwise, find the best feature to slit on. Based on JR Quinlan Decision
        # Tree algorithm, the best feature X should have the highest correlation to Y.
        else:
            numOfFeat = data.shape
            randRtlFeat = np.random.randint(0, data.shape[1] - 2)

            # Split down the middle and check its not just 2 rows.
            splitValue = np.median(data[:, randRtlFeat], axis=0)
            if splitValue == max(data[:, randRtlFeat]):
                return np.array([['leaf',
                                  np.mean(dataY),
                                  -1,
                                  -1]])

            # Create left tree.
            leftTree = self.BuildTree(
                data[data[:, randRtlFeat] <= splitValue]
            )

            # Create right tree.
            rightTree = self.BuildTree(
                data[data[:, randRtlFeat] > splitValue]
            )

            # Establish root of tree and create a decision tree from it.
            root = np.array([[randRtlFeat, splitValue, 1, leftTree.shape[0] + 1]])
            randomTree = np.vstack((np.vstack((root, leftTree)), rightTree))

            return randomTree


    def AddEvidence(self, X, Y):
        """
        Description: This function serves to add training data to the 
        random tree learner.

        Params:
            X (np.NDArray): X values of data to add.
            Y (np.1DArray): Y training values.

        Returns: Updated tree matrix for Random Tree Learner.
        """

        # Build a tree based on the data.
        data = np.hstack((X, Y.reshape(-1, 1)))
        self.tree = self.BuildTree(data)


    def Query(self, points):
        """
        Description: This function serves to estimate a set of test points given
        a model we created. Basically, this is a test function for our model.

        Params:
            points (np.NDArray): Test queries.

        Returns: Predictions in a numpy 1D array of estimated values.
        """
        yPred = []
        root = self.tree

        for row in range(points.shape[0]):
            
            node = 0

            while root[node, 0] != 'leaf':
                idx = root[node, 0]
                splitValue = root[node, 1]

                if points[row, int(float(idx))] <= float(splitValue):
                    left = int(float(root[node, 2]))
                    node = node + left

                else:
                    right = int(float(root[node, 3]))
                    node = node + right

            result = root[node, 1]
            yPred.append(float(result))

        return np.array(yPred)



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

    def __init__(self, verbose=False, **kwargs):
        """
        Description: This function serves to initialize an InsaneLearner
        and all its respective variables.

        Params:
            verbose (bool): Print process or not.
            **kwargs: Additional arguments.

        Returns: Initialized variables.
        """

        self.verbose = verbose
        self.learners = [BootstrapAggregatingLearner(learner=DecisionTreeLearner, bags=20)] * 20


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
# -------------------------------------------------------- Reinforcement Learning Algorithm Learners -------------------------------------------------------- #
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
        self.numOfStates = numOfStates
        self.numOfActions = numOfActions

        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        
        self.dyna = dyna
        self.verbose = verbose

        # Double ended queue data structure that allows insert and delete at both ends.
        self.memory = deque(maxlen=2000)

        # Keep track of the latest state and action.
        self.state = 0
        self.action = 0

        # Initialize a Q-table that records and updates q values for each action/state.
        self.Q = np.zeros(shape=(numOfStates, numOfActions))

        # Keep track of transitions from s to sprime when performing an aciton in Dyna-Q.
        self.T = {}

        # Keep track of reward for each action in each state when doing Dyna-Q.
        self.R = np.zeros(shape=(numOfStates, numOfActions))

    
    def RememberQValues(self, state, action, reward, nextState, done):
        """
        Description: Allows for remember the Q values and appends to deque data structure.
        Params:
            state (int): State of Q table.
            action (int): Action to perform for respective state.
            reward (float): Reward for specific aciton.
            nextState (int): Subsequent state of Q table.
            done (bool): If q value acquisition is complete.
        """
        self.memory.append((state, action, reward, nextState, done))
        

    def Act(self, state, reward, done=False, update=True):
        """
        Description: Peforms a query operation depending on current status of Q table.
        Params:
            state (int): Current state to perform query on.
            reward (float): Immediate reward from previous action.
            done (bool): If acting has been performed.
            update (bool): Update Q table based on values.
        Returns: Query.
        """
        if update:
            return self.Query(state, reward, done=done)
        
        else:
            return self.QueryState(state)


    def QueryState(self, state):
        """
        Description: Find the next action to take in state s. Update the latest state and action 
        without updating the Q table.
        Parameters:
            state (int): The new state
        Returns: The selected action to take in state.
        """
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.numOfActions - 1)
        
        else:
            action = self.Q[state, :].argmax()

        self.state = state
        self.action = action

        if self.verbose:
            print("\nState = {}, Action = {}".format(state, action))

        return action


    def Query(self, statePrime, reward, done=False):
        """
        Find the next action to take in state s_prime. Update the latest state 
        and action and the Q table. Update rule:
        Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmax a'(Q[s', a'])]).
        Parameters:
            statePrime (int): New state.
            reward (float): Immediate reward for taking the previous action.
        Returns: The selected action to take in statePrime.
        """
        self.RememberQValues(self.state, self.action, reward, statePrime, done)

        # Update Q table.
        self.Q[self.state, self.action] = (
            (1 - self.alpha) * self.Q[self.state, self.action] + 
            self.alpha * (reward + self.gamma * self.Q[statePrime, self.Q[statePrime, :].argmax()])
        )

        # Implement Dyna-Q.
        if self.dyna > 0:
            # Update reward table.
            self.R[self.state, self.action] = (
                (1 - self.alpha) * self.R[self.state, self.action] + self.alpha * reward
            )

            if (self.state, self.action) in self.T:
                if statePrime in self.T[(self.state, self.action)]:
                    self.T[(self.state, self.action)][statePrime] += 1

                else:
                    self.T[(self.state, self.action)][statePrime] = 1

            else:
                self.T[(self.state, self.action)] = { statePrime: 1 }

            Q = deepcopy(self.Q)

            # Hallucinations.
            for i in range (self.dyna):
                dummyState = rand.randint(0, self.numOfStates - 1)
                dummyAction = rand.randint(0, self.numOfActions - 1)

                if (dummyState, dummyAction) in self.T:
                    # Find the most common statePrime as a result of taking action.
                    dummyStatePrime = max(self.T[(dummyState, dummyAction)], key=lambda x: self.T[(dummyState, dummyAction)][x])

                    # Update temp table.
                    Q[dummyState, dummyAction] = (
                        (1 - self.alpha) * Q[dummyState, dummyAction] + 
                        self.alpha * (self.R[dummyState, dummyAction] + 
                        self.gamma * Q[dummyStatePrime, Q[dummyStatePrime, :].argmax()])
                    )

            # Update once dyna is complete.
            self.Q = deepcopy(Q)

        # Find the next action to take and update.
        nextAction = self.QueryState(statePrime)
        self.rar *= self.radr

        if self.verbose:
            print("\nState = {}, Action = {}, Reward = {}".format(statePrime, nextAction, reward))

        return nextAction


""" ------------------------------------------------------------------------------------------------------------------------- """
""" ---------------------------------------------------------- Deep Q Network ----------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

class DeepQNetwork(object):
    
    ACTIONS = { 0:-1, 1:0, 2:1 }

    def __init__(self, stateSize=20, actionSize=20, 
                 alpha=0.001, gamma=0.95, epsilon=1.0,
                 minEpsilon=0.9, epsilonDecay=0.9, verbose=False):
        """
        Description: Constructor for deep nueral network q learner.
        Params:
            stateSize(int): Number of states.
            actionSize (int): Number of actions.
            alpha (float): Learning rate.
            gamma (float): Value of future reward.
            epsilon (float): Exploration rate.
            minEpsilon (float): Minimum exploration rate.
            epsilonDecay (float): Decay rate for exploration.
            verbose (bool): Print info out.
        Returns: Initialized variables.
        """

        self.stateSize = stateSize
        self.actionSize = actionSize

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.minEpsilon = minEpsilon
        self.epsilonDecay = epsilonDecay

        self.memory = deque(maxlen=2000)
        
        # Build neural network.
        self.model = self.BuildModel()
        
        if verbose:
            self.verbose = 1
        
        else:
            self.verbose = 0

    
    def BuildModel(self):
        """
        Description: Builds the neural networks for our deep-q learning model.
        """
        # Prepares/Initializes NN layers.
        model = Sequential()
        model.add(Dense(60, input_dim=self.stateSize, activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(3, activation='linear'))
        
        # Configures model for training using MSE loss function and Adam optimizer.
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        return model


    def Remember(self, state, action, reward, nextState, done):
        """
        Description: Appends information to memory to remember performance and values.
        Params:
            state (int): Current state.
            action (int): Current action.
            reward (int): Current reward.
            nextState (int): Next state.
            done (bool): Completed or not.
        """
        self.memory.append((state, action, reward, nextState, done))


    def Act(self, state):
        """
        Description: Makes decision on best action to perform based on state parameter.
        Params:
            state (int): Current state.
        Returns: Action to perform.
        """
        if np.random.rand() <= self.epsilon:
            return rand.randrange(self.actionSize)

        actVals = self.model.predict(np.asarray([state]), verbose=self.verbose)

        return np.argmax(actVals[0])


    def RewardTarget(self, memory):
        """
        Description: Makes model understand what the best outcome is and reward algorithm accordingly.
        Params:
            memory (deque): Model memory.
        Returns: Target reward/action.
        """

        states = []
        targRwds = []

        for state, action, reward, nextState, done in memory:
            target = reward

            if not done:
                target = (
                    reward + self.gamma *
                    np.amax(self.model.predict(np.asarray([nextState]), 
                    verbose=self.verbose)[0])
                    )
            
            tempTarget = self.model.predict(np.asarray([state]), verbose=self.verbose)
            tempTarget[0][action] = target

            states.append(state)
            targRwds.append(tempTarget)

        return (states, targRwds)

    
    def Query(self, df):
        """
        Description: Fits model.
        Params:
            df (pd.DataFrame): Data to be tested.
        Returns: Trained model.
        """
        validMem = self.AddEvidence(df, df.index[-1], memory=[])

        validStates, validTargetRewards = self.RewardTarget(validMem)

        dfStates, dfTargetRewards = self.RewardTarget(self.memory)

        self.model.fit(np.asarray(dfStates),
                       np.asarray(dfTargetRewards)[:, 0, :],
                       validation_data=(np.asarray(validStates), np.asarray(validTargetRewards)[:, 0, :]),
                       epochs=10, 
                       verbose=self.verbose)

        if self.epsilon > self.minEpsilon:
            self.epsilon *= self.epsilonDecay


    def AddEvidence(self, df, endDate, memory=[]):
        """
        Description: Creates memory for training purposes.
        Param:
            df (pd.DataFrame): Dataframe to create memory from.
            endDate (string): End data of dataframe.
            memory (deque): Memory.
        Returns: Created memory.
        """
        

        for idx, (domain, range) in enumerate(df.iloc[self.stateSize:-2].iterrows()):

            state = np.asarray(df.iloc[idx: idx + self.stateSize]['Return'])

            action = self.ACTIONS[self.Act(state)]

            nextState = np.asarray(df.iloc[idx + 1: idx + self.stateSize + 1]['Return'])

            done = domain == endDate

            reward = action * df.iloc[idx + self.stateSize]['Return']

            memory.append((state, action, reward, nextState, done))
        
        return memory


    def TransformDf(self, df, windowSize=2):
        """
        Description: Transforms dataframe into a tradeable one using daily returns for the 
        DQN to use as reward system.
        Params:
            df (pd.DataFrame): Dataframe to transform.
            windowSize (int): Window size for return computing.
        Returns: Transformed dataframe.
        """
        
        df['Return'] = df[df.columns[0]].rolling(
            window=windowSize).apply(lambda x: x[1] / x[0] - 1)

        return df


    def CreateTradesDf(self, df, learner):

        dfTrades = { "Trade": []}
        cumRet = 1

        df = df.append(df.iloc[-1])

        for idx, (domain, range) in enumerate(
            df[learner.stateSize:-2].iterrows()):
        
            state = np.asarray(df.iloc[
                idx: idx + learner.stateSize
            ]['Return'])
            
            position = learner.Act(state)
            
            reward = position * df.iloc[ 
                idx + learner.stateSize + 1
            ]['Return']

            dfTrades["Trade"].append(position)
            cumRet *= 1 + reward

        dfTrades = pd.DataFrame(dfTrades, 
                    index=df.index[
                        learner.stateSize + 1:-1]).join(df)
        dfTrades["Portfolio Return"] = (dfTrades["Trade"] * dfTrades["Return"])
        dfTrades["DQNLearner"] = (1 + dfTrades["Portfolio Return"]).cumprod()
        dfTrades[df.columns[0]] = dfTrades[df.columns[0]] / dfTrades.iloc[0][df.columns[0]]
   
        return dfTrades


    def PlotDqnPerformance(self, testTrades, symbol):
        plt.plot(testTrades[[symbol]], label=symbol, color="maroon")
        plt.plot(testTrades[["DQNLearner"]], label="DQN Learner", color="green")
        plt.title("DQN Test Plot")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend(loc="best")

        fig = plt.gcf()
        fig.set_size_inches(9, 4)
        sym = str(symbol).lower().capitalize()
        plt.savefig(f"Images/{sym}DqnLearnerVisual.png")
        plt.close()


""" ------------------------------------------------------------------------------------------------------------------------- """
""" -------------------------------------------------------- QStrategyLearner ---------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

class StrategyLearner(object):
    
    BUY = 1
    HOLD = 0
    SELL = -1

    def __init__(self, accuracyBool=False, numOfShares=1, epochs=100, numOfSteps=10,
                 impact=0.0, commission=0.0, verbose=False,
                 learner=QLearner(numOfStates=3000, numOfActions=3)):
        """
        Description: This function serves to create a StrategyLearner that can learn a trading policy.
        Params:
            numOfShares (int): Number of shares that can be traded in one order.
            epochs (int): The number of times to train the learner.
            numOfSteps (int): Steps in getting discretization thresholds.
            impact (float): Difference between learner and actual data.
            commision (float): Amount charged per transaction.
            verbose (bool): Print info or not.
            learner (object): Learner to implement strategy on.
        Returns: Initialized variables.
        """

        self.numOfShares = numOfShares
        self.epochs = epochs
        self.numOfSteps = numOfSteps
        self.impact = impact
        self.commision = commission
        self.verbose = verbose
        self.accuracyBool = accuracyBool
        self.QLearner = learner


    def GetTechnicalIndicators(self, df):
        """
        Description: This function implements the technical indicators and features of a position and feeds it into the Q Learner.
        Params:
            df (pd.DataFrame): Dataframe to compute tech indicators.
        Returns: A pandas dataframe of the technical indicators.
        """
        df['MOMENTUM'] = TA.MOM(df, period=5)
        df['SMA'] = TA.SMA(df, period=5)
        df['BBWIDTH'] = TA.BBWIDTH(df, period=5) 
        df.drop(columns={'Open', 'High', 'Low', 'Close', 'Volume'}, inplace=True)
        df.dropna(inplace=True)
        return df


    def GetThresholds(self, dfTechnicalIndicators, numOfSteps):
        """
        Description: Computes thresholds to be used for discretization and 
        returns a 2-d numpy array where the first dimension indicates the index
        of features in dfTechnicalIndicators and second dimension refers to the value
        of a feature at the particular threshold.
        """
        stepSize = round(dfTechnicalIndicators.shape[0] / numOfSteps)

        tempDf = dfTechnicalIndicators.copy()

        thresholds = np.zeros(shape=(dfTechnicalIndicators.shape[1], numOfSteps))

        for idx, features in enumerate(dfTechnicalIndicators.columns):
            tempDf.sort_values(by=[features], inplace=True)

            for step in range(numOfSteps):
                if step < numOfSteps - 1:
                    thresholds[idx, step] = tempDf[features].iloc[(step + 1) * stepSize]

                else:
                    thresholds[idx, step] = tempDf[features].iloc[-1]

        return thresholds


    def Discretize(self, dfTechnicalIndicators, nonNegativePosition, thresholds):
        """
        Description: This function serves to discretize the upcoming values of the deep q network. 
        In applied mathematics, discretization is the process of transferring continuous functions, 
        models, variables, and equations into discrete counterparts. This process is usually 
        carried out as a first step toward making them suitable for numerical evaluation and 
        implementation on digital computers
        Params:
            dfTechnicalIndicators (pd.DataFrame): Dataframe with technical indicators.
            nonNegativePosition (int): Positions of DQN.
            thresholds (float): Threshold computed from previous function.
        Returns: State in Q Table from which we query the action.
        """
        state = nonNegativePosition * pow(self.numOfSteps, len(dfTechnicalIndicators))

        for idx in range(len(dfTechnicalIndicators)):
            threshold = thresholds[idx][thresholds[idx] >= dfTechnicalIndicators[idx]][0]

            thresholdIdx = np.where(thresholds == threshold)[1][0]

            state += thresholdIdx * pow(self.numOfSteps, idx)

        return state


    def GetPosition(self, prevPosition, signal):
        """
        Description: This function serves to find a new position based on the previous 
        position and the given signal. Signal is the action that results from querying
        a state which comes from discretize in the q table. Action is either 0, 1, 2 and is 
        the second index of the q table.
        """
        newPosition = self.HOLD

        if prevPosition < self.BUY and signal == self.BUY:
            newPosition = self.BUY

        elif prevPosition > self.SELL and signal == self.SELL:
            newPosition = self.SELL

        return newPosition


    def GetDailyReward(self, prevPrice, currPrice, position):
        """
        Description: This function serves to calculate the daily reward of the dataframe
        as a percentage change in prices.
        """

        return position * ((currPrice / prevPrice) - 1)

    
    def CheckConverged(self, cumReturns, patience=10):
        """
        Description: This function serves to check if the cumulative returns has converged. 
        Patience is the number of epochs with no improvements in cumulative returns. This
        will return either true or false.
        """

        if patience > len(cumReturns):
            return False

        lastFewReturns = cumReturns[-patience:]

        if len(set(lastFewReturns)) == 1:
            return True
        
        maxReturn = max(cumReturns)

        if maxReturn in lastFewReturns:
            if maxReturn not in cumReturns[:len(cumReturns) - patience]:
                return False

            else:
                return True

        return True


    def CreateDfTrades(self, orders, numOfShares,
                       hold=0, buy=1, sell=-1):
        """
        Description: This function serves to simply create a dataframe for 
        orders executed to simulate trading.
        """

        trades = []

        if self.accuracyBool == False:
            buyOrSell = orders[orders != hold]

            for date in buyOrSell.index:
                if buyOrSell.loc[date] == buy:
                    trades.append((date, numOfShares))

                elif buyOrSell.loc[date] == sell:
                    trades.append((date, -numOfShares))

        elif self.accuracyBool == True:
            buyOrSell = orders
            
            for date in buyOrSell.index:
                if buyOrSell.loc[date] == buy:
                    trades.append((date, numOfShares))

                elif buyOrSell.loc[date] == sell:
                    trades.append((date, -numOfShares))

                elif buyOrSell.loc[date] == hold:
                    trades.append((date, 0))

        dfTrades = pd.DataFrame(trades, columns=["Date", "Shares"])
        dfTrades.set_index("Date", inplace=True)

        return dfTrades


    def CreateBmDfTrades(self, symbol, startDate, endDate, numOfShares):
        """
        Description: This function serves to simply createa an empty df that can be used to test
        against the dfTrades dataframe and establishes a benchmark.
        """

        bmPrices = NormalizeDfs(IndividualHistoricalData(symbol, startDate, endDate, keepAllColumns="NO"))[0]

        dfBmTrades = pd.DataFrame(
            data=[
                (bmPrices.index.min(), numOfShares),
                (bmPrices.index.max(), -numOfShares)
            ], 
            columns=["Date", "Shares"]
        )

        dfBmTrades.set_index("Date", inplace=True)

        return dfBmTrades


    def SymbolValueFromTrading(self, dfOrders, symbol, startDate, endDate,
                               startVal=1, commision=9.95, impact=0.05):
        """
        Description: This function serves to simulate trading a stock 
        based on the orders performed and symbol given. This returns
        a column of the portfolio value given that one stock after every 
        action performed.
        """
        dfOrders.sort_index(ascending=True, inplace=True)

        dfPrices = NormalizeDfs(IndividualHistoricalData(symbol, startDate, endDate, keepAllColumns="NO"))[0]

        dfPrices["Cash"] = 1.0

        dfPrices.fillna(method="ffill", inplace=True)
        dfPrices.fillna(method="bfill", inplace=True)
        dfPrices.fillna(1.0, inplace=True)

        dfTrades = pd.DataFrame(np.zeros((dfPrices.shape)), dfPrices.index,
                                dfPrices.columns)

        for idx, row in dfOrders.iterrows():
            tradedShareVal = dfPrices.loc[idx, symbol] * row["Shares"]
            transactionCost = commision + impact * dfPrices.loc[idx, symbol] * abs(row["Shares"])

            if row["Shares"] > 0:
                dfTrades.loc[idx, symbol] = dfTrades.loc[idx, symbol] + row["Shares"]
                dfTrades.loc[idx, "Cash"] = dfTrades.loc[idx, "Cash"] - tradedShareVal - transactionCost

            elif row["Shares"] < 0:
                dfTrades.loc[idx, symbol] = dfTrades.loc[idx, symbol] + row["Shares"]
                dfTrades.loc[idx, "Cash"] = dfTrades.loc[idx, "Cash"] - tradedShareVal - transactionCost

        dfHoldings = pd.DataFrame(np.zeros((dfPrices.shape)), dfPrices.index,
                                  dfPrices.columns)

        for rowCount in range(len(dfHoldings)):
            if rowCount == 0:
                dfHoldings.iloc[0, :-1] = dfTrades.iloc[0, :-1].copy()
                dfHoldings.iloc[0, -1] = dfTrades.iloc[0, -1] + startVal

            else:
                dfHoldings.iloc[rowCount] = dfHoldings.iloc[rowCount - 1] + dfTrades.iloc[rowCount]

            rowCount += 1

        dfVal = dfPrices * dfHoldings
        
        portVals = pd.DataFrame(dfVal.sum(axis=1), dfVal.index, ["Port Val"])
        
        return portVals


    def MarketSimulator(self, dfOrders, dfOrdersBm, symbol,
                        startDate, endDate, title, startVal=1,
                        commission=9.95, impact=0.005,
                        saveFigure=False, figName="Plot.png",
                        showPlot=False):
        """
        Description: This function serves to mimic the market simulator project from ML4T
        university course by Tucker Balch. In summary, this function intakes dfOrders
        dataframe that executes the trades and displays the portfolio value respectively.
        """
        portVals = self.SymbolValueFromTrading(dfOrders=dfOrders, symbol=symbol,
                                               startVal=startVal, startDate=startDate,
                                               endDate=endDate, commision=commission,
                                               impact=impact)

        portValsBm = self.SymbolValueFromTrading(dfOrders=dfOrdersBm, symbol=symbol,
                                                 startVal=startVal, startDate=startDate,
                                                 endDate=endDate, commision=commission,
                                                 impact=impact)

        portValsBm.rename(columns={"Port Val": symbol}, inplace=True)

        temp = []
        temp.append(portVals)
        temp.append(portValsBm)
        df = temp[0].join(temp[1])

        plt.plot(df.loc[:, df.columns[1]], label=symbol, color="maroon")
        plt.plot(df.loc[:, df.columns[0]], label="QLearner", color="darkgreen")
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Normalized Prices")
        plt.legend(loc="best")

        fig = plt.gcf()
        fig.set_size_inches(9, 4)
        
        if saveFigure:
            plt.savefig(figName)

        if showPlot:
            plt.show()

        plt.close()
        
            
    def AddEvidence(self, symbol='GLD', startDate="2021-01-01", 
                    endDate="2022-01-01", startVal=1):
        """
        Description: This function serves to add training data to the 
        Strategy learner.
        Params:
            Self explanatory.
        Returns: Updated training data for Strategy learner.
        """
        tempDf = NormalizeDfs(IndividualHistoricalData(symbol=symbol, startDate=startDate,
                                      endDate=endDate, keepAllColumns="YES"))[0]

        dfPrices = NormalizeDfs(IndividualHistoricalData(symbol=symbol, startDate=startDate,
                                      endDate=endDate, keepAllColumns="NO"))[0]

        dfFeatures = self.GetTechnicalIndicators(tempDf)

        dfThres = self.GetThresholds(dfFeatures, self.numOfSteps)

        cumReturns = []

        for epoch in range(1, self.epochs + 1):

            # Initial position is hold.
            position = self.HOLD

            # Create pandas series that captures order signals.
            orders = pd.Series(index=dfFeatures.index)

            for day, date in enumerate(dfFeatures.index):
                # Get a state.
                state = self.Discretize(dfFeatures.loc[date],
                                        position + 1,
                                        dfThres)

                # Get action, do not update table if first time.
                if date == dfFeatures.index[0]:
                    action = self.QLearner.Act(state, 0.0, update=False)

                # Otherwise, calculate reward and update table.
                else:
                    prevPrice = dfPrices[symbol].iloc[day - 1]
                    currPrice = dfPrices[symbol].loc[date]

                    reward = self.GetDailyReward(prevPrice, currPrice, position)
                    action = self.QLearner.Act(state, reward, 
                                               done=date==dfFeatures.index[-1],
                                               update=True)

                # If last day, sell.
                if date == dfFeatures.index[-1]:
                    newPosition = -position

                else:
                    newPosition = self.GetPosition(position, action - 1)

                orders.loc[date] = newPosition

                position += newPosition

            dfTrades = self.CreateDfTrades(orders, self.numOfShares)
            portVals = self.SymbolValueFromTrading(dfOrders=dfTrades,
                                                   symbol=symbol,
                                                   startDate=startDate,
                                                   endDate=endDate,
                                                   startVal=startVal,
                                                   commision=self.commision,
                                                   impact=self.impact)

            cr = portVals.iloc[-1, 0] / portVals.iloc[0, 0] - 1
            cumReturns.append(cr)

            if self.verbose:
                print("Epoch: {}, Cumulative Return: {}\n".format(epoch, cr))

            if epoch > 20:
                if self.CheckConverged(cumReturns):
                    break

        if self.verbose:
            sns.heatmap(self.QLearner.Q, cmap='Blues')
            plt.plot(cumReturns)
            plt.xlabel("Epochs")
            plt.ylabel("Cumulative Return (%)")
            plt.show()

    
    def Query(self, symbol='GLD', startDate="2022-01-02", 
              endDate=datetime.today().strftime('%Y-%m-%d'), 
              ):
        """
        Description: This function serves to test the existing policy on a new data set.
        """

        tempDf = NormalizeDfs(IndividualHistoricalData(symbol=symbol,
                                            startDate=startDate,
                                            endDate=endDate,
                                            keepAllColumns="YES"))[0]                

        dfFeatures = self.GetTechnicalIndicators(tempDf)

        thresholds = self.GetThresholds(dfTechnicalIndicators=dfFeatures,
                                        numOfSteps=self.numOfSteps)

        position = self.HOLD

        orders = pd.Series(index=dfFeatures.index)

        for date in dfFeatures.index:
            state = self.Discretize(dfFeatures.loc[date],
                                    position + 1,
                                    thresholds)

            action = self.QLearner.Act(state, 0.0, update=False)

            if date == dfFeatures.index[-1]:
                newPosition = -position

            else:
                newPosition = self.GetPosition(position, action - 1)

            orders.loc[date] = newPosition

            position += newPosition

        dfTrades = self.CreateDfTrades(orders=orders,
                                       numOfShares=self.numOfShares)

        return dfTrades
        



# *********************************************************************************************************************************************************** #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------ DefeatLearners for LinRegLearner and DTLearner ----------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# *********************************************************************************************************************************************************** #


class DefeatLearners(object):

    def BestForLinRegLearner(self, seed=1489683273):
        """
        Description: This finds the best data for a linear regression learner.
        Params:
            seed (int): Input seed value to repeat random number.
        Returns: Optimized data for a linear regression learner.
        """
        np.random.seed(seed) 

        X = np.random.rand(100,4)
        Y = X[:, 0] + X[:,1]*3 + X[:,2]**2 - X[:,3]*4

        return X, Y 


    def BestForDecisionTreeLearner(self, seed=1489683273):
        """
        Description: This finds the best data for a decision tree learner.
        Params:
            seed (int): Input seed value to repeat random number.
        Returns: Optimized data for a decision tree learner.
        """

        np.random.seed(seed)
        # Data must contain between 10 to 1000 (inclusive) entries.
        rows = np.random.randint(10, 1001)

        # Number of features must be between 2 and 10 (inclusive).
        cols = np.random.randint(2, 11)

        # Generate random x data.
        x = np.random.rand(rows, cols)

        # Defeat linear regression learner with a non linear (exponential) function y = x0^5 + x1^2.
        y = (-1/8) * ( 
                        ((x[:, 0] - 2)**3) * ((x[:, 1] + 1)**2) * ((x[:, 2] - 4) )
            )

        return x, y


    def CompareRmses(self, learnerOne, learnerTwo, X, Y):
        """
        Description: This function serves to simply compare the two learners.

        Params:
            X (array): X axis values.
            Y (array): Y axis values.
            learnerOne/learnerTwo (object): LRL, KNN, DTL, RTL, etc.

        Returns: Rmses for both learners.
        """

        trainRows = int(math.floor(0.6 * X.shape[0]))
        testRows = X.shape[0] - trainRows

        train = np.random.choice(X.shape[0], size=trainRows, replace=False)
        test = np.setdiff1d(np.array(range(X.shape[0])), train)

        trainX = X[train, :]
        trainY = Y[train]

        testX = X[test, :]
        testY = Y[test]

        learnerOne.AddEvidence(trainX, trainY)
        learnerTwo.AddEvidence(trainX, trainY)

        predY = learnerOne.Query(testX)
        rmseOne = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])

        predY2 = learnerTwo.Query(testX)
        rmseTwo = math.sqrt(((testY - predY2) ** 2).sum() / testY.shape[0])

        return rmseOne, rmseTwo
     

    def TestDefeatLearners(self):
        lrl = LinRegLearner()
        dtl = DecisionTreeLearner(leafSize=1)

        X, Y = self.BestForLinRegLearner()
        rmseLrl, rmseDtl = self.CompareRmses(lrl, dtl, X, Y)
        print("\nBest for LRL Results.\nRMSE LRL: {}\nRMSE DTL: {}"
        .format(rmseLrl, rmseDtl))

        if rmseLrl < 0.9 * rmseDtl:
            print("LRL < 0.9 DTL: Pass.")
        else:
            print("LRL >= 0.9 DTL: Fail.")

        X, Y = self.BestForDecisionTreeLearner()
        rmseLrl, rmseDtl = self.CompareRmses(lrl, dtl, X, Y)
        print("\nBest for DTL Results.\nRMSE LRL: {}\nRMSE DTL: {}"
        .format(rmseLrl, rmseDtl))

        if rmseDtl < 0.9 * rmseLrl:
            print("DTL < 0.9 LRL: Pass.\n")
        else:
            print("DTL >= 0.9 LRL: Fail.\n")
   