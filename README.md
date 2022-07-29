# Stock Market Trader Bot
This repo is the contemporary status of the senior design project for my undergraduate Computer Engineering studies at Florida International University. This project is still in the making, however, this repo will updated with the progress as more intricate implementations are processed and tested. <br><br><br>

# Project Description
### Current Plan
I am currently training a machine learning algorithm using k-nearest neighbor, Q-Learning, Dyna, and historical pricing data of various stocks to formulate a predictive model that analyzes trends of varying positions in a portfolio. By training this model, I plan to have an algorithm that accurately forecasts the future beahvior of a stock and potentially an entire portfolio. Amongst this model, multiple high-level trading algorithms will be coherently used to optimize a portfolio and create thresholds of executing buy/sell orders to act as psychologically  uninclined as possible.

### Objectives
In the design phase of the Bot, the main objectives include safety, progressive training, marketability, and user friendliness. More specifically, these main objectives specifically quantify that the end product aims to ensure the product will have a success measurement high enough for the product to be generalized as profitable (>60% accurate predictions and order decisions), it will have the ability to opt-in and provide two functional options for trading (Automatic or Manual Trading), and it will be easy to utilize and maneuver by the user with minimal prior trading and stock exchange background. <br>

### Operation
The Stock Market Trader Bot will have a clear user interface that allows for easy handling of the Bot’s functional trading operations. Prior to any trading, the system will require a direct and established connection to a properly funded, compatible bank to allow for legal and effective connection, ensuring the ability to manipulate a portfolio's buy/sell orders. From this friendly interface, the client will be able to select whether it wants its current portfolio to be handled completely by the Bot using the Automatic Transaction Handling option or wants to work alongside the Bot using the Client/User Intervention option.<br><br><br>

# End-Product Description
The End Product Description explains how the Stock Market Trader Bot will operate along with the functions it will perform and can be categorized by its two functional operations: 1) Automatic Transaction Handline and 2) Client/User Intervention.

![image](https://user-images.githubusercontent.com/92603066/181686542-476d1349-8256-418b-ab07-0f44ae13ca6e.png)


### Automatic Transaction Handling
For this form of operation, the Bot will proceed to perform buy/sell orders of a predetermined portfolio of varying stocks based off algorithmically driven decision-making models. This specific option of operation for the Bot does not require any user intervention other than selecting a pool of stocks in which the user wishes to allow the Bot to manipulate through varying orders and exchanges. From here on forth, the Bot will use the library of machine learning algorithms it has been trained with to predict a stock’s price based on historical data analysis ranging from approximately 25 years prior. Furthermore, the Bot will also use economic theories to formulate a trend and moving average to understand the behavior of a select stock.

### Client/User Intervention
If the client was to select the alternative option, “Client/User Intervention,” the Bot continues to apply all its machine learning models, historical data analysis, Q Learning/Dyna algorithms, and economic theories. However, rather than automatically handling buy/sell orders and the decision-making process when controlling stock orders in a portfolio, this option will leave the choice to the user. In this functional option, the primary position and tasks of the system is to merely provide suggestions and notifications to the user of when it recommends acting on an investment.
