import numpy as N
import matplotlib
import matplotlib.pyplot as plt
import math


def import_files():
	
	Data = N.genfromtxt("Data_bitcoinity/30dPrice.csv",delimiter = ",")
	
		
	return Data[:,range(1,len(Data[0]))]
	

	
	
Data = import_files()



Timesteps = len(Data)

def BuyBTC(Balance,Price):
	USD = 0.0
	BTC = Balance[0]/Price
	return [USD,BTC]
	
def BuyUSD(Balance,Price):
	USD = Balance[1]*Price
	BTC = 0.0
	return [USD,BTC]

def argmin(Array_):

	Array = []
	for x in Array_:
		if not(math.isnan(x)):
			Array.append(x)

	lowest = 0
	for x in range(len(Array)):
		if Array[lowest] > Array[x]:
			lowest = int(x)
 	
 
	if math.isnan(Array[lowest]):
		print("ALLLLLLAAAARRRMMMMMMMM!!!!!!!!!!!!!!!!!!!!!!!!!!!!")	
 	
	return Array[lowest]

def argmax(Array_):

	Array = []
	for x in Array_:
		if not(math.isnan(x)):
			Array.append(x)
			
	highest = 0
	for x in range(len(Array)):
		if Array[highest] < Array[x]:
			highest = int(x)

	if math.isnan(Array[highest]):
		print("ALLLLLLAAAARRRMMMMMMMM!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			
	return Array[highest]

Balance = [100.0,0.0]

for i in range(Timesteps-1):

	#Evaluation
	if i%100 == 0:
		print("Balance: "+str(Balance[0])+"$\t ROI: "+str(Balance[0]/100)+"%")

		
		
	#Buying BTC
	BuyPrice = argmin(Data[i])
	Balance = BuyBTC(Balance,BuyPrice)
	
	#SellingBTC
	SellPrice = argmax(Data[i+1])
	Balance = BuyUSD(Balance,SellPrice)

	if math.isnan(Balance[0]):
		print("t: "+str(i)+"\tSellprice: "+str(SellPrice)+"\tBuyprice: "+str(BuyPrice))

		
