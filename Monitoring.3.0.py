import matplotlib
import matplotlib.pyplot as plt
import numpy as N
import tensorflow as tf
import datetime
import random


def moving_Average(List,n):	
	ret = N.cumsum(List)
	ret[n:] = ret[n:]-ret[:-n]
	return ret[n-1:]/n

def clean_Fog(Prices,n1 = 5000,n2=5000,Threshhold = 0):

	#Moving Averages
	Prices = moving_Average(Prices,n1)
	Prices -= N.mean(Prices)
	Prices /= N.max(Prices)
	
	CorrectPrediction = moving_Average((Prices[forecast:]-Prices[:-forecast]),n2)
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

	
#Raw Data Read
Data =N.genfromtxt("Big_Data2.csv",delimiter =",")



print("CSV file read succesfull")
print("Dataset has recorded "+str(len(Data))+" Timesteps.")

#Data 
forecast = 10
Prices,CorrectPrediction = clean_Fog(Data,n1=1,n2=50)


OH_CorrectPrediction = one_hot(CorrectPrediction.astype(int),2)
global_WinningBias = 100*N.sum(OH_CorrectPrediction[:,1])/len(OH_CorrectPrediction)
print("Data cleaned. \t Winning Bias: "+str(round(global_WinningBias,4)))


#TrainingPrameters
LR = 0.001
MO = 0.0001
Num_of_Epochs = 1000
num_batches = 2
batch_size = (len(Prices)/num_batches)-(forecast*num_batches)
num_features = 2

#Placeholder
In = tf.placeholder(tf.float32,shape=[num_batches,batch_size,1])
Out = tf.placeholder(tf.float32,shape =[num_batches,batch_size,num_features])
								
#Model 
lstm_size_l1 = 100
lstm_l1 = tf.contrib.rnn.BasicLSTMCell(lstm_size_l1)
lstm_out,state_l1 = tf.nn.dynamic_rnn(lstm_l1,In,dtype=tf.float32)
prediction = tf.contrib.layers.fully_connected(lstm_out,num_features,
												activation_fn = tf.nn.softmax)



#Cost
Cost = tf.reduce_sum(tf.pow(prediction-Out,2))
hit_rate = tf.equal(tf.argmax(prediction),tf.argmax(Out))
Accuracy = 100.0*tf.reduce_mean(tf.cast(hit_rate,tf.float32))


#Optimiser
optimizer = tf.train.AdamOptimizer().minimize(Cost)

print("Model has been defined.")

#Initialiser
init = tf.global_variables_initializer()

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

	sess.run(init)
	
	print("Tensorflow Session initiated")
	
	for Epoch in range(Num_of_Epochs):
 				

		a = sess.run(Accuracy,feed_dict={
		In:N.reshape(Data[:batch_size*num_batches],(num_batches,batch_size,1)),
		Out:N.reshape(OH_CorrectPrediction[forecast:(batch_size*num_batches)+forecast],(num_batches,batch_size,num_features))})
			
		c = sess.run(Cost,feed_dict={
		In:N.reshape(Data[:batch_size*num_batches],(num_batches,batch_size,1)),
		Out:N.reshape(OH_CorrectPrediction[forecast:(batch_size*num_batches)+forecast],(num_batches,batch_size,num_features))})			
								
		print("Epoch: "+str(Epoch)+"\tCost: "+str(c)+"\t Accuracy: "+str(a)+"%\t Market adjusted Accuracy:"+str(a-global_WinningBias))
			
			
		sess.run(optimizer,feed_dict={
			In:N.reshape(Data[:batch_size*num_batches],(num_batches,batch_size,1)),
			Out:N.reshape(OH_CorrectPrediction[forecast:(batch_size*num_batches)+forecast],(num_batches,batch_size,num_features))})


	saver = tf.train.Saver()
	saver.save(sess,"LSTM_model.mod")