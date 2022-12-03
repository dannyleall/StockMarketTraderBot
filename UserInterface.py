import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
from Output import DynaQOutput, DqnOutput
from Utilities import *
from IbTrading import *

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"


class SelectionInterface(customtkinter.CTkToplevel):

    def __init__(self, symbol, stlDecision, stlLastDate, stlTrades, stlAccuracyDict,
                 dqnDecision, dqnLastDate, dqnTrades, dqnAccuracyDict):
        super().__init__()

        self.symbol = symbol
        self.stlTradingDecision = stlDecision
        self.stlLastDate = stlLastDate
        self.stlTrades = stlTrades
        self.stlAccuracyDict = stlAccuracyDict

        self.dqnDecision = dqnDecision
        self.dqnLastDate = dqnLastDate
        self.dqnTrades = dqnTrades
        self.dqnAccuracyDict = dqnAccuracyDict

        # House keeping functions.
        self.title("Choose Your Learner!")
        self.geometry("600x680")
        self.protocol("WM_DELETE_WINDOW", self.CloseWindow)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Images.
        imageOne = Image.open("Images/StlButtonImg.png").resize((35, 35))
        self.ImageOne = ImageTk.PhotoImage(imageOne)
        imageTwo = Image.open("Images/DqnButtonImg.png").resize((35, 35))
        self.ImageTwo = ImageTk.PhotoImage(imageTwo)

        # Frames.
        self.FrameOne = customtkinter.CTkFrame(
            master=self,
            height=580)

        self.FrameOne.pack(
            pady=20,
            padx=60,
            fill="both",
            expand=True)

        # Labels.
        self.LabelOne = customtkinter.CTkLabel(
            master=self.FrameOne,
            justify=tkinter.CENTER,
            text="\nWelcome to the\n" +
            "Stock Market Trader Bot!",
            text_font=("Roboto", -30)
        )
        self.LabelOne.pack(pady=12, padx=10)

        self.LabelTwo = customtkinter.CTkLabel(
            master=self.FrameOne,
            justify=tkinter.LEFT,
            text="Compute Time*:\t~10 seconds\n" +
            "Accuracy*:\t~53%\n",
            text_font=("Roboto", -14)
        )
        self.LabelTwo.place(
            relx=0.055,
            rely=0.33
        )

        self.LabelThree = customtkinter.CTkLabel(
            master=self.FrameOne,
            justify=tkinter.LEFT,
            text="Compute Time*:\t~8 minutes\n" +
            "Accuracy*:\t~59%\n",
            text_font=("Roboto", -14)
        )
        self.LabelThree.place(
            relx=0.055,
            rely=0.54
        )

        self.LabelFour = customtkinter.CTkLabel(
            master=self.FrameOne,
            justify=tkinter.LEFT,
            text="*Computing time is calculated based on " +
            "actual timing tests performed by various " +
            "PCs\nwith average components and power.\n" +
            "*Accuracy is based on the algorithms daily " +
            "return the day after a prediction was made. " +
            "\nThis is performed over the 25 latest algorithmic " +
            "predictions of the AMZN, GME, TSLA\n" + 
            "and NVDA stock. From here, the accurate " +
            "predictions is divided by 100 to" +
            " acquire\nthe percentage.",
            text_font=("Roboto", -11)
        )

        self.LabelFour.place(
            relx=0.055,
            rely=0.75
        )

        #Buttons.
        self.FirstButton = customtkinter.CTkButton(
            master=self.FrameOne,
            width=150,
            height=70,
            corner_radius=6,
            text="Dyna-QLearner",
            image=self.ImageOne,
            text_font=("Roboto", -15),
            compound="right",
            command=self.OpenStlLearner
        )
        self.FirstButton.image = self.ImageOne

        self.FirstButton.place(
            relx=0.745,
            rely=0.36,
            anchor=tkinter.CENTER
        )

        self.SecondButton = customtkinter.CTkButton(
            master=self.FrameOne,
            width=120,
            height=70,
            corner_radius=6,
            text="Deep-QLearner",
            text_font=("Roboto", -15),
            image=self.ImageTwo,
            compound="right",
            command=self.OpenDqnLearner
        )
        self.SecondButton.image = self.ImageTwo

        self.SecondButton.place(
            relx=0.745,
            rely=0.56,
            anchor=tkinter.CENTER
        )

        # Drop down options.
        self.AppearanceOptions = customtkinter.CTkOptionMenu(
            master=self.FrameOne,
            values=["Light", "Dark"],
            text_color=("white", "white"),
            fg_color=("#FF8473", "#B22222"),
            button_color=("#FF8473", "#B22222"),
            button_hover_color=("#FF6955", "#460000"),
            dropdown_text_color=("white", "white"),
            dropdown_color=("#FF8473", "#460000"),
            dropdown_hover_color=("#FF6955", "#2a0000"),
            command=self.AppearanceMode)

        self.AppearanceOptions.set("Dark")
        self.AppearanceOptions.place(
            relx=0.5,
            rely=0.95,
            anchor=tkinter.CENTER    
        )

    def AppearanceMode(self, mode):
        customtkinter.set_appearance_mode(mode)

    def CloseWindow(self):
        self.destroy()

    def OpenStlLearner(self):
        stlApp = QLearnerInterface(
            self.symbol,
            self.stlTradingDecision, 
            self.stlLastDate, 
            self.stlTrades, 
            self.stlAccuracyDict)

        stlApp.mainloop()

    def OpenDqnLearner(self):
        dqnApp = DqnLearnerInterface(
            self.symbol,
            self.dqnDecision,
            self.dqnLastDate,
            self.dqnTrades,
            self.dqnAccuracyDict)
        dqnApp.mainloop()


class QLearnerInterface(customtkinter.CTkToplevel):

    trainStart = "2021-07-01"
    trainEnd = "2022-07-02"
    learnerImages = []

    def __init__(self, symbol, decision, lastDate, dfTrades, accuracyDict):
        super().__init__()

        self.symbol = symbol
        self.tradingDecision = decision
        self.lastTradingDay = lastDate
        self.accuracyDict = accuracyDict
        
        if str(decision) == "BUY":
            self.decisionColor = ("#006400", "#006400")
        elif str(decision) == "SELL":
            self.decisionColor = ("#DB3E39", "#821D1A")
        elif str(decision) == "HOLD":
            self.decisionColor = ("blue", "blue")

        if (dfTrades.iloc[-1]["Shares"] == -1):
            self.firstDay = "SELL"
        elif (dfTrades.iloc[-1]["Shares"] == 1):
            self.firstDay = "BUY"
        elif (dfTrades.iloc[-1]["Shares"] == 0):
            self.firstDay = "HOLD"

        if (dfTrades.iloc[-2]["Shares"] == -1):
            self.secondDay = "SELL"
        elif (dfTrades.iloc[-2]["Shares"] == 1):
            self.secondDay = "BUY"
        elif (dfTrades.iloc[-2]["Shares"] == 0):
            self.secondDay = "HOLD"

        if (dfTrades.iloc[-3]["Shares"] == -1):
            self.thirdDay = "SELL"
        elif (dfTrades.iloc[-3]["Shares"] == 1):
            self.thirdDay = "BUY"
        elif (dfTrades.iloc[-3]["Shares"] == 0):
            self.thirdDay = "HOLD"

        if (dfTrades.iloc[-4]["Shares"] == -1):
            self.fourthDay = "SELL"
        elif (dfTrades.iloc[-4]["Shares"] == 1):
            self.fourthDay = "BUY"
        elif (dfTrades.iloc[-4]["Shares"] == 0):
            self.fourthDay = "HOLD"

        if (dfTrades.iloc[-5]["Shares"] == -1):
            self.fifthDay = "SELL"
        elif (dfTrades.iloc[-5]["Shares"] == 1):
            self.fifthDay = "BUY"
        elif (dfTrades.iloc[-5]["Shares"] == 0):
            self.fifthDay = "HOLD"

        if (dfTrades.iloc[-6]["Shares"] == -1):
            self.sixthDay = "SELL"
        elif (dfTrades.iloc[-6]["Shares"] == 1):
            self.sixthDay = "BUY"
        elif (dfTrades.iloc[-6]["Shares"] == 0):
            self.sixthDay = "HOLD"

        if (dfTrades.iloc[-7]["Shares"] == -1):
            self.seventhDay = "SELL"
        elif (dfTrades.iloc[-7]["Shares"] == 1):
            self.seventhDay = "BUY"
        elif (dfTrades.iloc[-7]["Shares"] == 0):
            self.seventhDay = "HOLD"

        if (dfTrades.iloc[-8]["Shares"] == -1):
            self.eigthDay = "SELL"
        elif (dfTrades.iloc[-8]["Shares"] == 1):
            self.eigthDay = "BUY"
        elif (dfTrades.iloc[-8]["Shares"] == 0):
            self.eigthDay = "HOLD"

        self.dateOne = list(accuracyDict)[0]
        self.dateTwo = list(accuracyDict)[1]
        self.dateThree = list(accuracyDict)[2]
        self.dateFour = list(accuracyDict)[3]
        self.dateFive = list(accuracyDict)[4]
        self.dateSix = list(accuracyDict)[5]
        self.dateSeven = list(accuracyDict)[6]
        self.dateEight = list(accuracyDict)[7]

        self.accuracyOne = list(accuracyDict.values())[0]
        self.accuracyTwo = list(accuracyDict.values())[1]
        self.accuracyThree = list(accuracyDict.values())[2]
        self.accuracyFour = list(accuracyDict.values())[3]
        self.accuracyFive = list(accuracyDict.values())[4]
        self.accuracySix = list(accuracyDict.values())[5]
        self.accuracySeven = list(accuracyDict.values())[6]
        self.accuracyEight = list(accuracyDict.values())[7]

        # House keeping functions.
        self.title("Strategy Dyna-Q Algorithm")
        self.geometry(f"{950}x{720}")
        self.protocol("WM_DELETE_WINDOW", self.StlCloseWindow)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        #-----------------------------------------------------------------------------------------------#
        #---------------------------------------Left Side of UI-----------------------------------------#
        #-----------------------------------------------------------------------------------------------#
        
        # Right frame area.
        self.FrameOne = customtkinter.CTkFrame(master=self, width=700)
        self.FrameOne.grid(row=0, column=2, columnspan=1, sticky="nswe", padx=20, pady=20)
        self.FrameOne.grid_rowconfigure(0)
        self.FrameOne.grid_rowconfigure(1)
        self.FrameOne.grid_rowconfigure(2)
        self.FrameOne.grid_rowconfigure(3)
        self.FrameOne.grid_rowconfigure(4)
        self.FrameOne.grid_rowconfigure(5)
        self.FrameOne.grid_rowconfigure(6)
        self.FrameOne.grid_rowconfigure(7)
        self.FrameOne.grid_rowconfigure(8)
        self.FrameOne.grid_rowconfigure(9)
        self.FrameOne.grid_rowconfigure(11, weight=1)

        # Labels.
        self.LabelOne = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"\nRecent Predictions\nfor {self.symbol}\n",
            text_font=("Roboto", -18))
        
        self.LabelTwo = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateOne}\n" +
            f"    Prediction:\t              {self.firstDay}\n   Result:" +
            f"                    {self.accuracyOne}!\n",
            text_font=("Roboto", -13))

        self.LabelThree = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateTwo}\n" +
            f"    Prediction:\t              {self.secondDay}\n   Result:" +
            f"                    {self.accuracyTwo}!\n",
            text_font=("Roboto", -13))

        self.LabelFour = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateThree}\n" +
            f"    Prediction:\t              {self.thirdDay}\n   Result:" +
            f"                    {self.accuracyThree}!\n",
            text_font=("Roboto", -13))

        self.LabelFive = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateFour}\n" +
            f"    Prediction:\t              {self.fourthDay}\n   Result:" +
            f"                    {self.accuracyFour}!\n",
            text_font=("Roboto", -13))

        self.LabelSix = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateFive}\n" +
            f"    Prediction:\t              {self.fifthDay}\n   Result:" +
            f"                    {self.accuracyFive}!\n",
            text_font=("Roboto", -13))

        self.LabelSeven = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateSix}\n" +
            f"    Prediction:\t              {self.sixthDay}\n   Result:" +
            f"                    {self.accuracySix}!\n",
            text_font=("Roboto", -13))

        self.LabelEight = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateSeven}\n" +
            f"    Prediction:\t              {self.seventhDay}\n   Result:" +
            f"                    {self.accuracySeven}!\n",
            text_font=("Roboto", -13))

        self.LabelNine = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateEight}\n" +
            f"    Prediction:\t              {self.eigthDay}\n   Result:" +
            f"                    {self.accuracyEight}!\n",
            text_font=("Roboto", -13))

        # Drop-down options.
        self.AppearanceOptions = customtkinter.CTkOptionMenu(
            master=self.FrameOne,
            values=["Light", "Dark"],
            width=75,
            command=self.StlAppearance)

        self.AppearanceOptions.set("Dark")

        # Buttons.
        self.SeventhButton = customtkinter.CTkButton(
            master=self.FrameOne,
            width=50,
            text="More Predictions",
            command=self.StlPredictionsButton,
            fg_color=("#a9a9ff", "#192841"),
            border_color=("white", "gray38"),
            hover_color="#152238"
        )

        # Grids.
        self.LabelOne.grid(row=0, column=0, pady=0, padx=15)
        self.LabelTwo.place(anchor="w")
        self.LabelTwo.grid(row=1, column=0, pady=0, padx=15, sticky="w")
        self.LabelThree.place(anchor="w")
        self.LabelThree.grid(row=2, column=0, pady=0, padx=15, sticky="w")
        self.LabelFour.place(anchor="w")
        self.LabelFour.grid(row=3, column=0, pady=0, padx=15, sticky="w")
        self.LabelFive.place(anchor="w")
        self.LabelFive.grid(row=4, column=0, pady=0, padx=15, sticky="w")
        self.LabelSix.place(anchor="w")
        self.LabelSix.grid(row=5, column=0, pady=0, padx=15, sticky="w")
        self.LabelSeven.place(anchor="w")
        self.LabelSeven.grid(row=6, column=0, pady=0, padx=15, sticky="w")
        self.LabelEight.place(anchor="w")
        self.LabelEight.grid(row=7, column=0, pady=0, padx=15, sticky="w")
        self.LabelNine.place(anchor="w")
        self.LabelNine.grid(row=8, column=0, pady=0, padx=15, sticky="w")
        self.SeventhButton.place(relx=0.056, rely=0.93)
        self.AppearanceOptions.place(relx=0.62, rely=0.93)


        #-----------------------------------------------------------------------------------------------#
        #---------------------------------------Left Side of UI-----------------------------------------#
        #-----------------------------------------------------------------------------------------------#
        
        # Frames.
        self.FrameTwo = customtkinter.CTkFrame(master=self, width=200)
        self.FrameTwo.grid(row=0, column=1, columnspan=1, sticky="nswe", padx=20, pady=20)
        self.FrameTwo.rowconfigure((0, 1, 2, 3), weight=1)
        self.FrameTwo.rowconfigure(7, weight=10)
        self.FrameTwo.columnconfigure((0, 1), weight=1)
        self.FrameTwo.columnconfigure(2, weight=0)
        
        self.FirstEmbeddedFrame = customtkinter.CTkFrame(master=self.FrameTwo)
        self.FirstEmbeddedFrame.grid(row=0, column=0, columnspan=2, rowspan=1, pady=20, padx=20, sticky="nsew")
        self.FirstEmbeddedFrame.rowconfigure(4, weight=1)
        self.FirstEmbeddedFrame.columnconfigure(0, weight=1)
        
        self.SecondEmbeddedFrame = customtkinter.CTkFrame(master=self.FrameTwo, height=270)
        self.SecondEmbeddedFrame.grid(row=1, column=0, columnspan=2, rowspan=4, pady=0, padx=20, sticky="nsew")
        self.SecondEmbeddedFrame.rowconfigure(2, weight=1)
        self.SecondEmbeddedFrame.columnconfigure(0, weight=1)
        
        # Labels.
        self.FirstEmbeddedLabel = customtkinter.CTkLabel(
            master=self.FirstEmbeddedFrame,
            text="Strategy Dyna-Q Learner",
            height=100,
            corner_radius=6,
            text_font=("Roboto", -20),
            text_color=("#DB3E39", "#821D1A"),
            justify=tkinter.LEFT)

        self.SecondLabel = customtkinter.CTkLabel(
            master=self.FirstEmbeddedFrame,
            text=
            "\n\n\n\n\n   Description: Reinforcement learning algorithm " +
            "that finds the best action to take based   \n   on its " +
            f"current state and total maximized daily reward/return of {self.symbol}." +
            "\n\n   Technical Indicators: Momentum, standard moving " +
            f"average, and bollinger bandwidth\n    of {self.symbol}\n\n" +
            f"    Trading Decision: Based on the behavior of {self.symbol} " +
            f"in the time range {self.trainStart} to\n "+
            f"    {self.trainEnd}, " +
            f"as well as the closing price on {self.lastTradingDay}, our Dyna " +
            "Q Learner says\n     you should:\n\n",
            height=100,
            corner_radius=6,
            text_font=("Roboto", -13),
            fg_color=("white", "gray38"),
            justify=tkinter.LEFT)
        
        self.ThirdLabel = customtkinter.CTkLabel(
            master=self.FirstEmbeddedFrame,
            text=
            "Strategy Dyna-Q Algorithm",
            height=50,
            corner_radius=6,
            fg_color=("white", "gray38"),
            bg_color=("white", "gray38"),
            text_font=("Roboto", -20),
            justify=tkinter.LEFT)

        self.FourthLabel = customtkinter.CTkLabel(
            master=self.FirstEmbeddedFrame,
            text=f"{self.tradingDecision}",
            text_color=self.decisionColor,
            width=20,
            height=10,
            fg_color=("white", "gray38"),
            bg_color=("white", "gray38"),
            text_font=("Roboto", -20))

        # Images.
        symbolCased = self.symbol.lower().capitalize()
        imageThree = Image.open(f"Images/{symbolCased}StlLearnerVisual.png").resize((570, 260))
        self.ImageThree = ImageTk.PhotoImage(imageThree)

        self.SixthButton = customtkinter.CTkButton(
            master=self.FrameTwo,
            image=self.ImageThree,
            fg_color=("white", "gray38"),
            bg_color=("white", "gray38"),
            hover_color=("white", "gray38"),
            text=""
        )

        # Buttons.
        self.FourthButton = customtkinter.CTkButton(
            master=self.FrameTwo,
            text="SELL",
            border_width=2,
            width=250,
            fg_color=("#DB3E39", "#821D1A"),
            hover_color="#8B0000",
            command=self.StlSellStock(self.symbol, 1))
            
        self.FifthButton = customtkinter.CTkButton(
            master=self.FrameTwo,
            text="BUY",
            border_width=2,
            width=250,
            fg_color=("#7cb69d", "#006400"),
            hover_color="#002800",
            command=self.StlBuyStock(self.symbol, 1))

        # Grids.                       
        self.FourthButton.place(relx=0.545, rely=0.93)

        self.FifthButton.place(relx=0.075, rely=0.93)

        self.FirstEmbeddedLabel.grid(column=0, row=0,
            sticky="nwe", padx=15, pady=20)
        
        self.SecondLabel.grid(row=0, column=0, padx=10, pady=20)

        self.ThirdLabel.place(relx=0.095, rely=0.1)

        self.SixthButton.place(relx=0.066, rely=0.5)

        self.FourthLabel.place(relx=0.225, rely=0.741)

    def StlPredictionsButton(self):
        # PlotAccurateData(accuracyList)
        predictionsApp = customtkinter.CTkToplevel()
        symbolCased = self.symbol.lower().capitalize()
        predictionsApp.title("Here is the Predictions Plot!")
        predictionsApp.geometry("1200x750")
        imageThree = Image.open(f"Images/{symbolCased}StlPredictionsPlot.png").resize((1200, 750))
        PredictionsImage = ImageTk.PhotoImage(imageThree)
        
        PredictionsButton = customtkinter.CTkButton(
            master=predictionsApp,
            image=PredictionsImage,
            fg_color=("white", "gray38"),
            bg_color=("white", "gray38"),
            hover_color=("white", "gray38"),
            text=""
        )
        PredictionsButton.place(relx=0, rely=0)

        predictionsApp.mainloop()
                
    def StlBuyStock(self, symbol, quantity):
        # Uncomment while TWS is opened and ready to trade.
        # ibSession = IbFunctions() 
        # ibSession.BuyOrder(symbol, quantity)
        x=1 # Dummy to fill empty function.
        
    def StlSellStock(self, symbol, quantity):
        # Uncomment while TWS is opened and ready to trade.
        # ibSession = IbFunctions() 
        # ibSession.SellOrder(symbol, quantity)
        x=1 # Dummy to fill empty function.

    def StlAppearance(self, mode):
        customtkinter.set_appearance_mode(mode)

    def StlCloseWindow(self):
        self.destroy()

      
class DqnLearnerInterface(customtkinter.CTkToplevel):

    trainStart = "2021-07-01"
    trainEnd = "2022-07-02"
    learnerImages = []

    def __init__(self, symbol, decision, lastDate, dfTrades, accuracyDict):
        super().__init__()

        self.symbol = symbol
        self.tradingDecision = decision
        self.lastTradingDay = lastDate
        self.accuracyDict = accuracyDict
        
        if str(decision) == "BUY":
            self.decisionColor = ("#006400", "#006400")
        elif str(decision) == "SELL":
            self.decisionColor = ("#DB3E39", "#821D1A")
        elif str(decision) == "HOLD":
            self.decisionColor = ("blue", "blue")

        if (dfTrades.iloc[-1]["Trade"] == 0):
            self.firstDay = "SELL"
        elif (dfTrades.iloc[-1]["Trade"] == 2):
            self.firstDay = "BUY"
        elif (dfTrades.iloc[-1]["Trade"] == 1):
            self.firstDay = "HOLD"

        if (dfTrades.iloc[-2]["Trade"] == 0):
            self.secondDay = "SELL"
        elif (dfTrades.iloc[-2]["Trade"] == 2):
            self.secondDay = "BUY"
        elif (dfTrades.iloc[-2]["Trade"] == 1):
            self.secondDay = "HOLD"

        if (dfTrades.iloc[-3]["Trade"] == 0):
            self.thirdDay = "SELL"
        elif (dfTrades.iloc[-3]["Trade"] == 2):
            self.thirdDay = "BUY"
        elif (dfTrades.iloc[-3]["Trade"] == 1):
            self.thirdDay = "HOLD"

        if (dfTrades.iloc[-4]["Trade"] == 0):
            self.fourthDay = "SELL"
        elif (dfTrades.iloc[-4]["Trade"] == 2):
            self.fourthDay = "BUY"
        elif (dfTrades.iloc[-4]["Trade"] == 1):
            self.fourthDay = "HOLD"

        if (dfTrades.iloc[-5]["Trade"] == 0):
            self.fifthDay = "SELL"
        elif (dfTrades.iloc[-5]["Trade"] == 2):
            self.fifthDay = "BUY"
        elif (dfTrades.iloc[-5]["Trade"] == 1):
            self.fifthDay = "HOLD"

        if (dfTrades.iloc[-6]["Trade"] == 0):
            self.sixthDay = "SELL"
        elif (dfTrades.iloc[-6]["Trade"] == 2):
            self.sixthDay = "BUY"
        elif (dfTrades.iloc[-6]["Trade"] == 1):
            self.sixthDay = "HOLD"

        if (dfTrades.iloc[-7]["Trade"] == 0):
            self.seventhDay = "SELL"
        elif (dfTrades.iloc[-7]["Trade"] == 2):
            self.seventhDay = "BUY"
        elif (dfTrades.iloc[-7]["Trade"] == 1):
            self.seventhDay = "HOLD"

        if (dfTrades.iloc[-8]["Trade"] == 0):
            self.eigthDay = "SELL"
        elif (dfTrades.iloc[-8]["Trade"] == 2):
            self.eigthDay = "BUY"
        elif (dfTrades.iloc[-8]["Trade"] == 1):
            self.eigthDay = "HOLD"

        self.dateOne = list(accuracyDict)[0]
        self.dateTwo = list(accuracyDict)[1]
        self.dateThree = list(accuracyDict)[2]
        self.dateFour = list(accuracyDict)[3]
        self.dateFive = list(accuracyDict)[4]
        self.dateSix = list(accuracyDict)[5]
        self.dateSeven = list(accuracyDict)[6]
        self.dateEight = list(accuracyDict)[7]

        self.accuracyOne = list(accuracyDict.values())[0]
        self.accuracyTwo = list(accuracyDict.values())[1]
        self.accuracyThree = list(accuracyDict.values())[2]
        self.accuracyFour = list(accuracyDict.values())[3]
        self.accuracyFive = list(accuracyDict.values())[4]
        self.accuracySix = list(accuracyDict.values())[5]
        self.accuracySeven = list(accuracyDict.values())[6]
        self.accuracyEight = list(accuracyDict.values())[7]

        # House keeping functions.
        self.title("Deep-Q Neural Network Algorithm")
        self.geometry(f"{950}x{720}")
        self.protocol("WM_DELETE_WINDOW", self.DqnCloseWindow)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        #-----------------------------------------------------------------------------------------------#
        #---------------------------------------Left Side of UI-----------------------------------------#
        #-----------------------------------------------------------------------------------------------#
        
        # Right frame area.
        self.FrameOne = customtkinter.CTkFrame(master=self, width=700)
        self.FrameOne.grid(row=0, column=2, columnspan=1, sticky="nswe", padx=20, pady=20)
        self.FrameOne.grid_rowconfigure(0)
        self.FrameOne.grid_rowconfigure(1)
        self.FrameOne.grid_rowconfigure(2)
        self.FrameOne.grid_rowconfigure(3)
        self.FrameOne.grid_rowconfigure(4)
        self.FrameOne.grid_rowconfigure(5)
        self.FrameOne.grid_rowconfigure(6)
        self.FrameOne.grid_rowconfigure(7)
        self.FrameOne.grid_rowconfigure(8)
        self.FrameOne.grid_rowconfigure(9)
        self.FrameOne.grid_rowconfigure(11, weight=1)

        # Labels.
        self.LabelOne = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"\nRecent Predictions\nfor {self.symbol}\n",
            text_font=("Roboto", -18))
        
        self.LabelTwo = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateOne}\n" +
            f"    Prediction:\t              {self.firstDay}\n   Result:" +
            f"                    {self.accuracyOne}!\n",
            text_font=("Roboto", -13))

        self.LabelThree = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateTwo}\n" +
            f"    Prediction:\t              {self.secondDay}\n   Result:" +
            f"                    {self.accuracyTwo}!\n",
            text_font=("Roboto", -13))

        self.LabelFour = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateThree}\n" +
            f"    Prediction:\t              {self.thirdDay}\n   Result:" +
            f"                    {self.accuracyThree}!\n",
            text_font=("Roboto", -13))

        self.LabelFive = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateFour}\n" +
            f"    Prediction:\t              {self.fourthDay}\n   Result:" +
            f"                    {self.accuracyFour}!\n",
            text_font=("Roboto", -13))

        self.LabelSix = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateFive}\n" +
            f"    Prediction:\t              {self.fifthDay}\n   Result:" +
            f"                    {self.accuracyFive}!\n",
            text_font=("Roboto", -13))

        self.LabelSeven = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateSix}\n" +
            f"    Prediction:\t              {self.sixthDay}\n   Result:" +
            f"                    {self.accuracySix}!\n",
            text_font=("Roboto", -13))

        self.LabelEight = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateSeven}\n" +
            f"    Prediction:\t              {self.seventhDay}\n   Result:" +
            f"                    {self.accuracySeven}!\n",
            text_font=("Roboto", -13))

        self.LabelNine = customtkinter.CTkLabel(
            master=self.FrameOne,
            text=f"    Date:\t\t{self.dateEight}\n" +
            f"    Prediction:\t              {self.eigthDay}\n   Result:" +
            f"                    {self.accuracyEight}!\n",
            text_font=("Roboto", -13))

        # Drop-down options.
        self.AppearanceOptions = customtkinter.CTkOptionMenu(
            master=self.FrameOne,
            values=["Light", "Dark"],
            width=75,
            command=self.DqnAppearance)

        self.AppearanceOptions.set("Dark")

        # Buttons.
        self.SeventhButton = customtkinter.CTkButton(
            master=self.FrameOne,
            width=50,
            text="More Predictions",
            command=self.DqnPredictionsButton,
            fg_color=("#a9a9ff", "#192841"),
            border_color=("white", "gray38"),
            hover_color="#152238"
        )

        # Grids.
        self.LabelOne.grid(row=0, column=0, pady=0, padx=15)
        self.LabelTwo.place(anchor="w")
        self.LabelTwo.grid(row=1, column=0, pady=0, padx=15, sticky="w")
        self.LabelThree.place(anchor="w")
        self.LabelThree.grid(row=2, column=0, pady=0, padx=15, sticky="w")
        self.LabelFour.place(anchor="w")
        self.LabelFour.grid(row=3, column=0, pady=0, padx=15, sticky="w")
        self.LabelFive.place(anchor="w")
        self.LabelFive.grid(row=4, column=0, pady=0, padx=15, sticky="w")
        self.LabelSix.place(anchor="w")
        self.LabelSix.grid(row=5, column=0, pady=0, padx=15, sticky="w")
        self.LabelSeven.place(anchor="w")
        self.LabelSeven.grid(row=6, column=0, pady=0, padx=15, sticky="w")
        self.LabelEight.place(anchor="w")
        self.LabelEight.grid(row=7, column=0, pady=0, padx=15, sticky="w")
        self.LabelNine.place(anchor="w")
        self.LabelNine.grid(row=8, column=0, pady=0, padx=15, sticky="w")
        self.SeventhButton.place(relx=0.056, rely=0.93)
        self.AppearanceOptions.place(relx=0.62, rely=0.93)


        #-----------------------------------------------------------------------------------------------#
        #---------------------------------------Left Side of UI-----------------------------------------#
        #-----------------------------------------------------------------------------------------------#
        
        # Frames.
        self.FrameTwo = customtkinter.CTkFrame(master=self, width=200)
        self.FrameTwo.grid(row=0, column=1, columnspan=1, sticky="nswe", padx=20, pady=20)
        self.FrameTwo.rowconfigure((0, 1, 2, 3), weight=1)
        self.FrameTwo.rowconfigure(7, weight=10)
        self.FrameTwo.columnconfigure((0, 1), weight=1)
        self.FrameTwo.columnconfigure(2, weight=0)
        
        self.FirstEmbeddedFrame = customtkinter.CTkFrame(master=self.FrameTwo)
        self.FirstEmbeddedFrame.grid(row=0, column=0, columnspan=2, rowspan=1, pady=20, padx=20, sticky="nsew")
        self.FirstEmbeddedFrame.rowconfigure(4, weight=1)
        self.FirstEmbeddedFrame.columnconfigure(0, weight=1)
        
        self.SecondEmbeddedFrame = customtkinter.CTkFrame(master=self.FrameTwo, height=270)
        self.SecondEmbeddedFrame.grid(row=1, column=0, columnspan=2, rowspan=4, pady=0, padx=20, sticky="nsew")
        self.SecondEmbeddedFrame.rowconfigure(2, weight=1)
        self.SecondEmbeddedFrame.columnconfigure(0, weight=1)
        
        # Labels.
        self.FirstEmbeddedLabel = customtkinter.CTkLabel(
            master=self.FirstEmbeddedFrame,
            text="Deep-Q Neural Network Learner",
            height=100,
            corner_radius=6,
            text_font=("Roboto", -20),
            text_color=("#DB3E39", "#821D1A"),
            justify=tkinter.LEFT)

        self.SecondLabel = customtkinter.CTkLabel(
            master=self.FirstEmbeddedFrame,
            text=
            "\n\n\n\n\n   Description: Deep reinforcment learning " +
            "algorithm that combines our Dyna-QLearner\n   with " +
            "neural networks (NN). Within our DQN agent, we " +
            "create functions that remember\n   the behavior of the " +
            "stock within a given iteration of the Q table and " +
            "stores them in a\n   Deque collection." +
            "\n\n   Technical Indicators: Daily return of "+
            f"{self.symbol} is used as its reward.\n\n" +
            f"    Trading Decision: Based on the behavior of  " +
            f"in the time range {self.trainStart} to\n "+
            f"    {self.trainEnd}, " +
            f"as well as the closing price on {self.lastTradingDay}, our Deep-" +
            "Q Learner says\n     you should:\n\n",
            height=100,
            corner_radius=6,
            text_font=("Roboto", -13),
            fg_color=("white", "gray38"),
            justify=tkinter.LEFT)
        
        self.ThirdLabel = customtkinter.CTkLabel(
            master=self.FirstEmbeddedFrame,
            text=
            "Deep-Q Neural Network Algorithm",
            height=50,
            corner_radius=6,
            fg_color=("white", "gray38"),
            bg_color=("white", "gray38"),
            text_font=("Roboto", -20),
            justify=tkinter.LEFT)

        self.FourthLabel = customtkinter.CTkLabel(
            master=self.FirstEmbeddedFrame,
            text=f"{self.tradingDecision}",
            text_color=self.decisionColor,
            width=20,
            height=10,
            fg_color=("white", "gray38"),
            bg_color=("white", "gray38"),
            text_font=("Roboto", -20))

        # Images.
        symbolCased = self.symbol.lower().capitalize()
        imageThree = Image.open(f"Images/{symbolCased}DqnLearnerVisual.png").resize((570, 260))
        self.ImageThree = ImageTk.PhotoImage(imageThree)

        self.SixthButton = customtkinter.CTkButton(
            master=self.FrameTwo,
            image=self.ImageThree,
            fg_color=("white", "gray38"),
            bg_color=("white", "gray38"),
            hover_color=("white", "gray38"),
            text=""
        )

        # Buttons.
        self.FourthButton = customtkinter.CTkButton(
            master=self.FrameTwo,
            text="SELL",
            border_width=2,
            width=250,
            fg_color=("#DB3E39", "#821D1A"),
            hover_color="#8B0000",
            command=self.DqnSellStock(self.symbol, 1))
            
        self.FifthButton = customtkinter.CTkButton(
            master=self.FrameTwo,
            text="BUY",
            border_width=2,
            width=250,
            fg_color=("#7cb69d", "#006400"),
            hover_color="#002800",
            command=self.DqnBuyStock(self.symbol, 1))

        # Grids.                       
        self.FourthButton.place(relx=0.545, rely=0.93)

        self.FifthButton.place(relx=0.075, rely=0.93)

        self.FirstEmbeddedLabel.grid(column=0, row=0,
            sticky="nwe", padx=15, pady=20)
        
        self.SecondLabel.grid(row=0, column=0, padx=10, pady=20)

        self.ThirdLabel.place(relx=0.095, rely=0.1)

        self.SixthButton.place(relx=0.066, rely=0.5)

        self.FourthLabel.place(relx=0.245, rely=0.761)

    def DqnPredictionsButton(self):
        # PlotAccurateData(accuracyList)
        predictionsApp = customtkinter.CTkToplevel()
        symbolCased = self.symbol.lower().capitalize()
        predictionsApp.title("Here is the Predictions Plot!")
        predictionsApp.geometry("1200x750")
        imageThree = Image.open(f"Images/{symbolCased}DqnPredictionsPlot.png").resize((1200, 750))
        PredictionsImage = ImageTk.PhotoImage(imageThree)
        
        PredictionsButton = customtkinter.CTkButton(
            master=predictionsApp,
            image=PredictionsImage,
            fg_color=("white", "gray38"),
            bg_color=("white", "gray38"),
            hover_color=("white", "gray38"),
            text=""
        )
        PredictionsButton.place(relx=0, rely=0)

        predictionsApp.mainloop()
                 
    def DqnBuyStock(self, symbol, quantity):
        # Uncomment while TWS is opened and ready to trade.
        # ibSession = IbFunctions() 
        # ibSession.BuyOrder(symbol, quantity)
        x=1 # Dummy to fill empty function.
          
    def DqnSellStock(self, symbol, quantity):
        # Uncomment while TWS is opened and ready to trade.
        # ibSession = IbFunctions() 
        # ibSession.SellOrder(symbol, quantity)
        x=1 # Dummy to fill empty function.

    def DqnAppearance(self, mode):
        customtkinter.set_appearance_mode(mode)

    def DqnCloseWindow(self):
        self.destroy()
      

def CreateSymbolSelectionInterface():
    app = customtkinter.CTk()
    app.title("Type In A Stock Symbol!")
    app.geometry("400x250")
    var = customtkinter.StringVar()

    Label = customtkinter.CTkLabel(
        justify=tkinter.CENTER,
        text="Welcome! Please Type Your\nDesired" +
        " Stock Acronym.",
        text_font=("Roboto", -22))

    LoadingMessage = customtkinter.CTkLabel(
        text="As soon as you click Enter, there will " +
        "about a 10 minute\ndelay before the next " +
        "window opens. " +
        "This is because\nwe will be performing all " +
        "necessary computations.")

    Entry = customtkinter.CTkEntry(
        placeholder_text="Enter Stock Symbol",
        textvariable=var,
        height=30,
        width=330)

    def SaveSymbol():
        app.destroy()
        
    Button = customtkinter.CTkButton(
        text="Enter",
        text_font=("Roboto", -13),
        height=35,
        width=330,
        fg_color=("#FF8473", "#B22222"),
        hover_color=("#FF6955", "#460000"),
        command=SaveSymbol)

    Label.place(relx=0.15, rely=0.1)
    LoadingMessage.place(relx=0.11, rely=0.73)
    Button.place(relx=0.089, rely=0.55)
    Entry.place(relx=0.089, rely=0.38)
    Entry.focus()

    app.mainloop()

    symbolStr = str(var.get())
    print("\nTraining for the Strategy Dyna-Q and Deep-Q Learners for {} has begun...".format(symbolStr))
    stlDecision, stlTrades, stlAccuracyDict, stlLastDate = DynaQOutput(symbolStr)
    dqnDecision, dqnTrades, dqnAccuracyDict, dqnLastDate = DqnOutput(symbolStr)
    
    return (symbolStr, stlDecision, stlTrades, stlAccuracyDict, stlLastDate), (dqnDecision, dqnTrades, dqnAccuracyDict, dqnLastDate)


def RunMe():
    (symbolStr, stlDecision, stlTrades, stlAccuracyDict, stlLastDate), (dqnDecision, dqnTrades, dqnAccuracyDict, dqnLastDate)  = CreateSymbolSelectionInterface()

    appTwo = SelectionInterface(symbolStr, stlDecision, stlLastDate, stlTrades, stlAccuracyDict,
                                dqnDecision, dqnLastDate, dqnTrades, dqnAccuracyDict)
    appTwo.mainloop()


if __name__ == "__main__":
    (symbolStr, stlDecision, stlTrades, stlAccuracyDict, stlLastDate), (dqnDecision, dqnTrades, dqnAccuracyDict, dqnLastDate)  = CreateSymbolSelectionInterface()

    appTwo = SelectionInterface(symbolStr, stlDecision, stlLastDate, stlTrades, stlAccuracyDict,
                                dqnDecision, dqnLastDate, dqnTrades, dqnAccuracyDict)
    appTwo.mainloop()
