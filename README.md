# Table of Contents
- [Table of Contents](#table-of-contents)
  - [Installing Dependencies ](#installing-dependencies-)
  - [Running the Project](#running-the-project)
    - [First Window](#first-pop-up-window)
    - [Second Window](#second-pop-up-window)
    - [Predictions Window](#algorithmic-predictions-windows)
  - [Other Functionalities](#other-functionalities)
    - [Implementing More Buttons](#implementing-more-buttons)
  - [Repository File Explanations](#repository-file-explanations)
    - [Data/Istanbul.csv](#dataistanbulcsv)
    - [Images/{ImageName}.png](#imagesimagenamepng)
    - [IbTrading.py](#ibtradingpy)
    - [Learners.py](#learnerspy)
    - [Output.py](#outputpy)
    - [README.md](#readmemd)
    - [Requirements.txt](#requirementstxt)
    - [RunMe.py](#runmepy)
    - [Software Documentation.docx](#software-documentationdocx)
    - [TestLearners.pynb](#testlearnersipynb)
    - [UserInterface.py](#userinterfacepy)
    - [Utilities.py](#utilitiespy)
    - [UtilitiesTestsAndExamples](#utilitiestestandexamplespy)

<br>

## Installing Dependencies <a name="Introduction"></a>
1. In order to run this program without errors, all the necessary libraries must be installed as outlined per [Requirements.txt](https://github.com/dannyleall/StockMarketTraderBot/blob/main/Requirements.txt) file. For the easiest installation process: 
   - Open this project in [VS Code](https://code.visualstudio.com/download) and open the terminal by pressing 
``
Ctrl+Shift+`
``.
   - Then, install the dependecies with pip and the terminal using:
  
     ```
     pip3 install -r Requirements.txt
     ```
<br>

## Running the Project
1. Once all dependencies have been installed, you are ready to run the project.
    - **Using VS Code:** Navigate to the project's [RunMe.py]([##RunMe.py](https://github.com/dannyleall/StockMarketTraderBot/blob/main/RunMe.py)) file, and run the file. 
        
        **NOTE:** There will be a 5 to 10 minute delay after you type your stock in the first window in order to train and test the algorithms.
        <br>

        ### First Pop-Up Window
        - Contains entry box for user to input stock. Once typed, and enter is selected, algorithms will begin computing.
  
          ![](Images/FirstWindow.png)
          
          <br>

        ### Second Pop-Up Window
        - Here, you can select which algorithm you would like to see the prediction for.
        
            **Dark Mode**
          ![](Images/DarkSecondWindow.png)

          <br>

        ### Algorithmic Predictions Windows
        - Lastly, you can now access all the information relevant to the learner you selected.
 
            **Deep-Q Algorithm (Light Mode)**
          ![](Images/LightLastWindow.png)  
          <br>

## Other Functionalities
### Implementing More Buttons
1. The user interface has two buttons after selecting the learner: BUY and SELL. Currently, this project as is does not incorporate BUY and SELL of the stock you inputted in the fist pop-up window.

    - If you **do not** wish to use the BUY and SELL buttons functionality, ignore this section and [run the project](#running-the-project)! Otherwise, follow these instructions:
    
        **Step One:** Install [Trader WorkStation (TWS) API](https://www.interactivebrokers.com/en/trading/tws.php#tws-software).
        
        **Step Two:** Create an [InteractiveBrokers account](https://gdcdyn.interactivebrokers.com/Universal/Application) and ensure a funded account.

        **Step Three:** Un-comment out lines [562-564](https://github.com/dannyleall/StockMarketTraderBot/blob/main/UserInterface.py#L562-L564), [568-570](https://github.com/dannyleall/StockMarketTraderBot/blob/main/UserInterface.py#L568-L570), [951-953](https://github.com/dannyleall/StockMarketTraderBot/blob/main/UserInterface.py#L951-L953), and [957-959](https://github.com/dannyleall/StockMarketTraderBot/blob/main/UserInterface.py#L951-L953) of [UserInterface.py](https://github.com/dannyleall/StockMarketTraderBot/blob/main/UserInterface.py).

        **Step Four:** Follow short instructions on `Connecting Code to TWS` section of the [Software Documentation.docx](https://github.com/dannyleall/StockMarketTraderBot/blob/main/Software%20Documentation.docx) to ensure an established Interactive Brokers connection.

        **Step Five:** [Run the project](#running-the-project)!

<br>

## Repository File Explanations
- This section provides a very general description and overall purpose of every single file present in this project. For a more intense, deeper technical understanding of the files, refer to the [Software Documentation.docx](https://github.com/dannyleall/StockMarketTraderBot/blob/main/Software%20Documentation.docx) that provides more in-depth descriptions of all code in this project.


   ### Data/Istanbul.csv
  - **Description:** This dataset includes the returns of multiple worldwide indexes for a number of days in history. 
  - **Purpose:** Utilized in [TestLearners.ipynb](https://github.com/dannyleall/StockMarketTraderBot/blob/main/TestLearners.ipynb) to assess our learners per the ML4T [university course project by Tucker Balch](https://quantsoftware.gatech.edu/Spring_2020_Project_3:_Assess_Learners).
    - The overall objective is to predict what the return for the MSCI Emerging Markets (EM) index will be on the basis of the other index returns. Y in this case is the last column to the right, and the X values are the remaining columns to the left (except the first column which is the date).

   ### Images/{ImageName}.png
  - **Description:** This is a folder that simply contains the images for our `README.md` and [UserInterface.py](https://github.com/dannyleall/StockMarketTraderBot/blob/main/UserInterface.py) files. 
  - **Purpose:** Improve project appearance and instructions.

   ### IbTrading.py
  - **Description:** This file contains all functions dealing with connecting our code to a Interactive Brokers trading account and submitting buy/sell orders accordingly.
  - **Purpose:** Code that allows for buy/sell orders through the click of a button.

   ### Learners.py
  - **Description:** Contains all of our algorithms including a decision tree, random tree, bootstrap aggregating learner, insane learner, Dyna-Q, Strategy Dyna-Q, and Deep-Q.
  - **Purpose:** Incorporate machine learning and deep learning to predict the behavior of a stock.

   ### Output.py
  - **Description:** Ties in together all of our utility functions and algorithms together into one clean output. 
  - **Purpose:** Convert a regression-based prediction (e.g., Stock Price will increase $25 tomorrow) to a classification output (e.g., BUY/SELL/HOLD).

   ### README.md
  - **Description:** Clear instructions on how a random user can use this project. 
  - **Purpose:** Facilitate the user's experience when using this repository.

   ### Requirements.txt
  - **Description:** Contains all necessary dependencies to run this project. 
  - **Purpose:** For user to easily install necessary libraries in one terminal command.
  
   ### RunMe.py
  - **Description:** Connects entire project into four lines of code. 
  - **Purpose:** For user to easily identify how to run the project.

   ### Software Documentation.docx
  - **Description:** Has a description, parameters, and return types (if applicable) of all functions in the entire project. 
  - **Purpose:** To technically document all functions in a manner that it will be easy to incorporate/understand the code in the future.

   ### TestLearners.ipynb
  - **Description:** Tests the performance of all our learners in the [Learners.py](https://github.com/dannyleall/StockMarketTraderBot/blob/main/Learners.py). 
  - **Purpose:** To visually understand the differences in performance of all our algorithms.

   ### UserInterface.py
  - **Description:** Has classes and functions necessary to construct our graphical user interface. 
  - **Purpose:** To tie in all the work into a clean application/GUI.

   ### Utilities.py
  - **Description:** Contains all utility functions necessary for the project. Also contains all utility functions from the [ML4T Udacity course](https://www.udacity.com/course/machine-learning-for-trading--ud501) by Tucker Balch. 
  - **Purpose:** Create convenient, easily callable functions that can be used throughout the project files. 

   ### UtilitiesTestAndExamples.py
  - **Description:** Contains all testing for the utility functions' functionality. Also contains all of the examples in the form of a function respective to lessons 7 through 10. 
  - **Purpose:** Ensure utility functions work as expected and grasp a better understanding of the ML4T Udacity Coursework.
