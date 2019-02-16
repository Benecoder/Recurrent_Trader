Quotes = ["aapl","adbe","amzn","celg","csco","gild","goog","googl","intc","isrg","msft","nflx","nvda","qqq"]

import matplotlib.pyplot as plt
import numpy as N
import math


"""
for i in Quotes:
	Names.append("Raw_Data/"+i+".csv")


for i in Names:
	A = N.genfromtxt(i,delimiter = ",")
	print(A)
	Lists.append(A)
	
Data = []
for i in range(len(Lists[0])):
	
	buffer = [Lists[0][i][0]]
	for x in range(len(Lists)):
		for a in Lists[x][i][1:]:
			buffer.append(a)
	Data.append(buffer)
	
N.savetxt("Raw_Data/UnscaledStock.csv",Data,delimiter = ",")


import urllib



for i in Quotes:
	testfile = urllib.urlretrieve("http://www.google.com/finance/historical?q="+i+"&output=csv","Raw_Data/"+i+".csv")

import os

for i in Quotes:
	FILE = open("Raw_Data/"+i+".csv","r")
	Content = FILE.readlines()
	FILE.close()

	os.system("rm Raw_Data/"+i+".csv")
	
	FILE = open("Raw_Data/"+i+".csv","w")
	FILE.writelines(Content[1:])
	FILE.close()
	
import time
import datetime

Months = ["Jan",28,"Feb",31,"Mar",30,
			"Apr",31,"May",30,"Jun",31,
			"Jul",31,"Aug",30,"Sep",31,
			"Oct",30,"Nov",31,"Dec",0]
Marks = {}

days = 0
for i in range(0,len(Months),2):
	days += Months[i+1]
	buffer = {Months[i]:days}
	Marks.update(buffer)



def convert(date):
	divider = 0
	while date[divider] != "-":
		divider += 1
		
	day = float(date[:divider])
	
	month = date[divider+1:divider+4]
	day += float(Marks[month])
	
	return str(day)
	


import numpy as N
Names = []
Lists = []

for i in Quotes:
	Names.append("Raw_Data/"+i+".csv")

for i in Names:
	FILE = open(i,"r")
	A = FILE.readlines()
	FILE.close()
	for x in range(len(A)):
		Line = A[x]
		
		mark = 0
		while Line[mark] != ",":
			mark += 1
		
		date = Line[:mark]
		
		day = convert(date)
		
		A[x] = day + Line[mark:]
		print(A[x])
		
	#FILE = open(i,"w")
	#FILE.writelines(A)
	#FILE.close()
		
"""

Data = N.genfromtxt("Raw_Data/UnscaledStock.csv",delimiter = ",")

Data = list(Data)

for x in range(len(Data)):
	Data[x] = list(Data[x])
	for i in range(len(Data[x])):
		if math.isnan(Data[x][i]):
			Data[x][i] = 50

Data = N.array(Data)

for x in range(len(Data[0])):
	Data[:,x] = Data[:,x]/N.max(Data[:,x])


plt.plot(Data[:,3])
plt.show()

Data = N.flip(Data,0)
 


#N.savetxt("CompleteStock.csv",Data,delimiter = ",")