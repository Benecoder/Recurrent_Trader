import numpy as N
#import csv
import time
import datetime
import warnings
import math

import matplotlib
import matplotlib.pyplot as plt

N.seterr(all='warn')
warnings.filterwarnings('error')
#Cleaning the files

def clean(filename):

	#Removes the Empty spots
	File = []
	with open(filename+".csv","rb") as Inputfile:

		Input_Table = csv.reader(Inputfile)

		for row in Input_Table:
			File.append(row)

	print(len(File))

	with open(filename+"_clean.csv","wb") as Inputfile:

		Writer = csv.writer(Inputfile)
		for Row in range(1,len(File)):
		
			#Converts to unix timestamp
			File[Row][0] = time.mktime(time.strptime(File[Row][0],"%Y-%m-%d %H:%M:%S UTC"))
		
			#Files Wholes
			for Element in range(len(File[Row])):
				if File[Row][Element] == "":
					File[Row][Element] = "0.0"
					
			Writer.writerow(File[Row])
	


			
def Average(filename):
	File = []
	
	Table = N.genfromtxt(filename+"_clean.csv",delimiter = ",")
	for row in Table:
		count = 0
		for i in row[1:]:
			if i != 0.0:
				count += 1	
		File.append([row[0],N.sum(row[1:])])
	
	#Max = N.max(N.abs(File[:,1]))
	#File[:,1] /= Max

	with open(filename+"_clean_new.csv","wb") as Inputfile:
		 Writer = csv.writer(Inputfile)
		 for Line in File:
		 	Writer.writerow(Line)

def readData():
	#Raw Data Read
	Data = []
	Data.append(N.genfromtxt("./Data/Volume_clean.csv",delimiter = ","))
	Data.append(N.genfromtxt("./Data/trades_per_minute_clean.csv",delimiter=","))
	Data.append(N.genfromtxt("./Data/Time_per_block_clean.csv",delimiter=","))
	Data.append(N.genfromtxt("./Data/Price_clean.csv",delimiter=","))
	Data.append(N.genfromtxt("./Data/Number_of_Transactions_clean.csv",delimiter=","))
	Data.append(N.genfromtxt("./Data/Mining_Difficulty_clean.csv",delimiter = ","))
	Data.append(N.genfromtxt("./Data/Market_cap_clean.csv",delimiter = ","))
	Data.append(N.genfromtxt("./Data/Hashrate_clean.csv",delimiter = ","))
	Data.append(N.genfromtxt("./Data/Block_size_clean.csv",delimiter = ","))
	Data.append([])
	Data.append([])
	
	Basp = N.genfromtxt("./Data/Bid_ask_sum_clean.csv",delimiter = ",")
	Basm = N.genfromtxt("./Data/Bid_ask_spread_clean.csv",delimiter=",")

	Len = len(Data[0])

	Baspt = 0
	Basmt = 0
	for t in range(Len):
		if Data[0][t][0] < Basp[Baspt][0]:
			Data[9].append(Basp[Baspt])
		elif Data[0][t][0] >= Basp[Baspt][0]:
			Data[9].append(Basp[Baspt])
			Baspt += 1
			
		if Data[0][t][0] < Basm[Basmt][0]:
			Data[10].append(Basm[Basmt])
		elif Data[0][t][0] >= Basm[Basmt][0]:
			Data[10].append(Basm[Basmt])
			Basmt += 1
			
	Data_ = []
	for i in range(Len):
		dump = []
		for a in Data:
			dump.append(a[i][1])
		Data_.append(dump)

	Data_ = N.array(Data_,dtype = 'float64')

	print(Data_.shape)

	N.savetxt("Big_Data.csv",Data_,delimiter =",")	

def merge():
	Names = ["1coinUSD.csv","btc24USD.csv","btcexUSD.csv","crytrUSD.csv",
			"exmoUSD.csv","hitbtcUSD.csv","intrsngUSD.csv",
			"itbitUSD.csv","justUSD.csv","krakenUSD.csv"]

	#Oldest List  is btc24USD Starting at 02/04/2011 @ 10:02am (UTC) / 1296813733.0
	
	startpoint = 1296813600
	i=0

	Resolution = int((time.time()-startpoint)/1800)

	
	File = []
	for Batch in Names:
	
		Data = N.genfromtxt(Batch,delimiter = ",")
		
		print(Batch+" imported.")
		
		dump = []
	
		startindex = 0
		stopindex = 0

	
		for t in range(Resolution):
		
			t_start = (t*1800)+1296813600
			t_stop = (t*1800)+1296813600+1800
	
	
			startindex = int(stopindex)
			
			try:		
				while startindex < len(Data)-2 and t_start >= Data[startindex][0]:
					startindex += 1
			except Warning:
				print("Startindex: "+str(startindex)+"\tt_start: "+str(t_start))
				
			stopindex = int(startindex)+1
			

			
			try:
				while stopindex < len(Data)and t_stop >= Data[stopindex][0]:
					stopindex += 1
			except IndexError:
				print("stopindex: "+str(stopindex)+"\tlength: "+str(len(Data)))
			
			try:
				dump.append(N.mean(Data[:,1][startindex:stopindex]))
			except Warning:
				dump.append(0.0)

			
			if t%1000 == 0:
				print("Time stamp: "+str(datetime.datetime.fromtimestamp(t_start))+"\t"+str(round(100.0*(float(startindex)/float(len(Data))),4))+"% done.")
	
		
		
		N.savetxt(Batch+".2",dump)
		

	
	
	#File = N.array(File)
	#N.savetxt("Big_Data2.csv",File)
	
def merge_2():

	Names = ["1coinUSD.csv","btc24USD.csv","btcexUSD.csv","crytrUSD.csv",
			"exmoUSD.csv","hitbtcUSD.csv","intrsngUSD.csv",
			"itbitUSD.csv","justUSD.csv","krakenUSD.csv"]

	Data = []
	for file in Names:
		Data.append(N.genfromtxt(file+".2",delimiter = ","))
	
	
	plt.plot(Data[0])
	plt.show()
	"""	
	Price = []
	for i in range(len(Data[0][:115513])):
		dump = 0
		count = 0
		for b in range(len(Data)):
			if Data[b][i] != 0.0:
				dump += Data[b][i]
				count += 1
		try:
			Price.append(dump/count)
		except ZeroDivisionError:
			Price.append(0.0)
			print(i)


	#N.savetxt("Big_Data3.csv",Price)
	
	"""


def Stock_Price():

	Companies = ["F","HMC","NSANY","TM","VLKAY"]

	Data = []
	for name in Companies:
		#Data of one file consistes of :
		#Date,Open,High,Low,Close,Adj Close,Volume
	
		buffer = N.genfromtxt("./Cars/"+name+".csv",delimiter=",")

		Data.append(buffer[:,2]) # High
		Data.append(buffer[:,3]) # Low
		Data.append(buffer[:,6]) # Close

		
	for i in range(len(Data)):
		Data[i] -= N.mean(Data[i])
		Data[i] /= N.max(Data[i])
		
	Data = N.array(Data)
	Data = Data.transpose()
	
	
	N.savetxt("CarData.csv",Data)


		
Stock_Price()