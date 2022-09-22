import numpy as np
from Utilities import *
from Learners import DecisionTreeLearner, LinRegLearner

# *********************************************************************************************************************************************************** #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------- Defeat Learners Using GenerateData and DefeatLearners -------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------------------------------------- #
# *********************************************************************************************************************************************************** #


""" ------------------------------------------------------------------------------------------------------------------------- """
""" ------------------------------------------------------ Generate Data Class ---------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

class GenerateData(object):

    def __init__(self):
        pass


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


""" ------------------------------------------------------------------------------------------------------------------------- """
""" ----------------------------------------------------- Defeat Learners Class --------------------------------------------- """
""" ------------------------------------------------------------------------------------------------------------------------- """

class DefeatLearners(object):

    def TestDefeatLearners():
        lrl = LinRegLearner()
        dtl = DecisionTreeLearner(leafSize=1)

        X, Y = GenerateData().BestForLinRegLearner()
        rmseLrl, rmseDtl = CompareRmses(lrl, dtl, X, Y)
        print("\nBest for LRL Results.\nRMSE LRL: {}\nRMSE DTL: {}"
        .format(rmseLrl, rmseDtl))

        if rmseLrl < 0.9 * rmseDtl:
            print("LRL < 0.9 DTL: Pass.")
        else:
            print("LRL >= 0.9 DTL: Fail.")

        X, Y = GenerateData().BestForDecisionTreeLearner()
        rmseLrl, rmseDtl = CompareRmses(lrl, dtl, X, Y)
        print("\nBest for DTL Results.\nRMSE LRL: {}\nRMSE DTL: {}"
        .format(rmseLrl, rmseDtl))

        if rmseDtl < 0.9 * rmseLrl:
            print("DTL < 0.9 LRL: Pass.\n")
        else:
            print("DTL >= 0.9 LRL: Fail.\n")


if __name__ == '__main__':
    DefeatLearners.TestDefeatLearners()