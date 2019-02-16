import matplotlib
import matplotlib.pyplot as plt
import numpy as N
import tensorflow as tf
import datetime
import random

Data = N.genfromtxt("BitcoinPriceHistory2010_2017.csv",delimiter = ",")
print(len(Data))
print("CSV file read succesfull")

def BuyBTC(Balance,Price):
	_Balance = [0,0]
	_Balance[0] = Balance[0]-100
	_Balance[1] = Balance[1]+(100*Price)
	return _Balance
	
def BuyEUR(Balance,Price):
	_Balance = [0,0]
	_Balance[0] = Balance[0]+100
	_Balance[1] = Balance[1]-(100*Price)
	return _Balance

def value(Balance,Price):
	Eur = Balance[0]
	Btc = Balance[1]/Price
	return round(Eur+Btc,1)
	
def simulate():

	print("Starting Balance" +str(round(Data[0][1],1))+"\t"+str(Balance)+"\t"+str(value(Balance,Data[0][1])))
	Balance = [2500,2500*Data[0][1]]  #Euros and Bitcoins
	x = 0
	while x < 24000:

		if random.random() > 0.2:
			Balance = BuyBTC(Balance,Data[x][1])
			print("Buy Bitcoin\t"+str(round(Data[x][1],1))+"\t"+str(N.round(Balance))+"\t"+str(value(Balance,Data[x][1])))
		else:
			Balance = BuyEUR(Balance,Data[x][1])
			print("Buy Euros\t"+str(round(Data[x][1],1))+"\t"+str(N.round(Balance))+"\t"+str(value(Balance,Data[x][1])))
		x = x + 1000

def moving_Average(List,n):	
	ret = N.cumsum(List)
	ret[n:] = ret[n:]-ret[:-n]
	return ret[n-1:]/n

def clean_Fog(Prices,n1 = 5000,n2=5000,Threshhold = 0):

	#Moving Averages
	Prices = moving_Average(Prices,n1)
	CorrectPrediction = moving_Average((Prices[1:]-Prices[:-1]),n2)
	Prices = Prices[:len(CorrectPrediction)-1]

	
	#Thresholding the Noise
	noise_indicies = N.abs(CorrectPrediction) < Threshhold 
	CorrectPrediction[noise_indicies] = 0
	
	CorrectPrediction = (N.sign(CorrectPrediction)+1)/2
	
	return Prices,CorrectPrediction
	
def one_hot(Array,features):

	Encoded = N.zeros((len(Array),features))
	Encoded[N.arange(len(Array)),Array] = 1
	
	return Encoded

LR = 0.05
MO = 0.001
Len_of_Sample = 10
Num_of_Epochs = 100
Batchsize = 2000
Dimension = [Len_of_Sample,10,2]



#Data
Prices,CorrectPrediction = clean_Fog(Data[:,1],n1=1,n2=50)
OH_CorrectPrediction = one_hot(CorrectPrediction.astype(int),2)
WinningBias = N.sum(OH_CorrectPrediction[:,1])/len(OH_CorrectPrediction)
print("Data cleaned. \t Winning Bias: "+str(round(WinningBias,4)))



#Placeholder
In = tf.placeholder(tf.float32,shape=[1,Len_of_Sample])
Out = tf.placeholder(tf.float32,shape =[1,2])

Weights = []
Biases = []
for Layer in range(1,len(Dimension)):
	Weights.append(tf.Variable(tf.random_uniform([Dimension[Layer-1],Dimension[Layer]],
								minval = -1,maxval = 1)))
	Biases.append(tf.Variable(tf.random_uniform([Dimension[Layer]],
								minval = -1,maxval = 1)))
								
#Model 
z = In
for Layer in range(len(Weights)-1):
	z = tf.tanh(tf.add(tf.matmul(z,Weights[Layer]),Biases[Layer]))
z = tf.nn.softmax(tf.add(tf.matmul(z,Weights[-1]),Biases[-1]))
prediction = z

#Cost
Cost = tf.reduce_sum(tf.pow(prediction-Out,2))
Accuracy = tf.reduce_sum(tf.abs(tf.round(prediction)-Out))

#Optimiser
optimizer = tf.train.MomentumOptimizer(LR,MO).minimize(Cost)

print("Model has been defined.")

#Initialiser
init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)
	
	print("Tensorflow Session initiated")
	
	for Epoch in range(Num_of_Epochs):
	

	
 		print("Epoch: "+str(Epoch))
					
		if Epoch % 1 ==0:
			c = 0
			a = 0
			
			for Sample in range(Len_of_Sample,len(Prices)-11):
				c += sess.run(Cost,feed_dict={
				In:N.reshape(Prices[Sample-Len_of_Sample:Sample],(1,Len_of_Sample)),
				Out:N.reshape(OH_CorrectPrediction[Sample+10],(1,2))})
				
				a += sess.run(Accuracy,feed_dict={
				In:N.reshape(Prices[Sample-Len_of_Sample:Sample],(1,Len_of_Sample)),
				Out:N.reshape(OH_CorrectPrediction[Sample+10],(1,2))})
			
								
			print("Cost: "+str(c)+"\t Accuracy: "+str(100*WinningBias*(1-(a/(Batchsize*2))))+"%")
			
	
		for Sample in range(Len_of_Sample,len(Prices)-11):
			sess.run(optimizer,feed_dict={
					In:N.reshape(Prices[Sample-Len_of_Sample:Sample],(1,Len_of_Sample)),
					Out:N.reshape(OH_CorrectPrediction[Sample+10],(1,2))})
