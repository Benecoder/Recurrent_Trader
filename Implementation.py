import tensorflow as tf 
import matplotlib
import matplotlib.pyplot as plt
import numpy as N

#Raw Data Read
Data =N.genfromtxt("Big_Data2.csv",delimiter =",")


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

	Balance = [2500,2500*Data[0]]  #USD and Bitcoins
	print("Starting Price: " +str(round(Data[0]))+"\tcombinedValue: "+str(value(Balance,Data[0]))+"\tUSD: "+str(Balance[0])+"$\tBTC: "+str(Balance[1]))
	
	Oracle = sess.run(prediction,feed_dict={
	In:N.reshape(Data[:batch_size*num_batches],(num_batches,batch_size,1)),
	Out:N.reshape(OH_CorrectPrediction[forecast:(batch_size*num_batches)+forecast],(num_batches,batch_size,num_features))})

	print(len(Oracle[0]))

	for x in range(len(Oracle[0])):
	
		if N.argmax(Oracle[0][x]) == 0:
			Balance = BuyBTC(Balance,Data[x])
			#print("Buy Bitcoin\t"+str(round(Data[x],1))+"\t"+str(N.round(Balance))+"\t"+str(value(Balance,Data[x])))
		else:
			Balance = BuyEUR(Balance,Data[x])
			#print("Buy Euros\t"+str(round(Data[x],1))+"\t"+str(N.round(Balance))+"\t"+str(value(Balance,Data[1])))
		if x%1000 == 0:
			print("Balance: "+str(N.round(N.array(Balance),2))+"\tValue: "+str(value(Balance,Data[x]))+"$")
			
def EventCounter(Pred):
	sign = int(Pred[0])
	Pred = abs(Pred[1:]-Pred[:-1])
		
	Events = [0]
	for x in range(len(Pred)):
		if Pred[x]:
			Events.append(int(x))
	Events.append(len(Pred)-1)
	return Events,sign


def clean_Fog(Prices):
	
	CorrectPrediction = Prices[forecast:]-Prices[:-forecast]
	
	CorrectPrediction = (N.sign(CorrectPrediction)+1)/2

	return CorrectPrediction	
def one_hot(Array,features):

	Encoded = N.zeros((len(Array),features))
	Encoded[N.arange(len(Array)),Array] = 1
	
	return Encoded

#Raw Data Read
Data =N.genfromtxt("CarData.csv",delimiter =" ")

NumOfElements = len(Data)
print("Dataset has recorded "+str(NumOfElements)+" Timesteps.")
print("CSV file read succesfull")


#Data 
forecast = 10
CorrectPrediction = clean_Fog(Data[:,0])
Data = Data[:len(CorrectPrediction)]

NumOfSources = len(Data[0])

OH_CorrectPrediction = one_hot(CorrectPrediction.astype(int),2)


global_WinningBias = 100*N.sum(OH_CorrectPrediction[:,1])/len(OH_CorrectPrediction)
print("Data cleaned.")

#TrainingPrameters
LR = 0.001
MO = 0.0001
Num_of_Epochs = 1000
batch_size = int(len(Data)/2)
num_features = 2


if len(Data) != len(OH_CorrectPrediction):
	print("\n Length of Data: "+str(len(Data))+"\tLength of Prediction: "+str(len(OH_CorrectPrediction)))
	raise Exception("\n\nYou fucked up and now your In/Out arrays don't match in length. It's a catastrophy!")




#Placeholder
In = tf.placeholder(tf.float32,shape=[1,batch_size,NumOfSources])
Out = tf.placeholder(tf.float32,shape =[1,batch_size,num_features])
								
#Model 
lstm_size_l1 = 100
lstm_l1 = tf.contrib.rnn.BasicLSTMCell(lstm_size_l1)
lstm_out,state_l1 = tf.nn.dynamic_rnn(lstm_l1,In,dtype=tf.float32)
prediction = tf.contrib.layers.fully_connected(lstm_out,num_features,
												activation_fn = tf.nn.softmax)

#Cost
Cost = tf.reduce_sum(tf.pow(prediction-Out,2))
hit_rate = tf.equal(tf.argmax(prediction,axis = 2),tf.argmax(Out,axis = 2))
Accuracy = 100.0*tf.reduce_mean(tf.cast(hit_rate,tf.float32))


print("Model has been defined.")


with tf.Session() as sess:

		saver = tf.train.Saver()
		saver.restore(sess,"LSTM_model.mod")
	
		Performance = [-1]
		
		LenOfTest = len(Data)-batch_size
		
		for i in range(LenOfTest):#len(Data)-batch_size):
			Pred = sess.run(prediction,feed_dict={In:N.reshape(Data[i:i+batch_size],(1,batch_size,NumOfSources)),
			Out:N.reshape(OH_CorrectPrediction[i:i+batch_size],(1,batch_size,num_features))})

			Pred = N.round(Pred[0][:,0],0)
			
			if Pred[-1] == OH_CorrectPrediction[i][1]:
				print(True)
				Performance.append(1)
			else:
				print(False)
				Performance.append(-1)
		Performance.append(1)
		
		print("Accuracy over the ehole Dataset: "+str(N.mean(N.array(Performance))))
		
		Matches,sign = EventCounter(N.array(Performance))
		
		plt.plot(Data[:,0][batch_size:batch_size+LenOfTest])
		for i in range(1,len(Matches)):
			if sign == 1:
				plt.axvspan(Matches[i-1],Matches[i],facecolor = "green",alpha = 0.5)
				sign -= 2
			elif sign == -1:
				plt.axvspan(Matches[i-1],Matches[i],facecolor = "red",alpha =0.5)
				sign += 2
		plt.show()
		
