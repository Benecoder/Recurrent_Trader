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
forecast = 10#
CorrectPrediction = clean_Fog(Data[:,0])
Data = Data[:len(CorrectPrediction)]
NumOfSources = len(Data[0])

OH_CorrectPrediction = one_hot(CorrectPrediction.astype(int),2)


global_WinningBias = 100*N.sum(OH_CorrectPrediction[:,1])/len(OH_CorrectPrediction)
print("Data cleaned. \t Winning Bias: "+str(round(global_WinningBias,4)))

#TrainingPrameters
LR = 0.001
MO = 0.0001
Num_of_Epochs = 100
batch_size = int(len(Data)/2)
num_features = 2

TestData = Data[-batch_size:]
Data = Data[batch_size:]

TestOH_CorrectPrediction = OH_CorrectPrediction[-batch_size:]
OH_CorrectPrediction = OH_CorrectPrediction[batch_size:]

if len(Data) != len(OH_CorrectPrediction):
	print("\n Length of Data: "+str(len(Data))+"\tLength of Prediction: "+str(len(OH_CorrectPrediction)))
	raise Exception("\n\nYou fucked up and now your In/Out arrays don't match in length. It's a catastrophy!")
elif len(Data) != batch_size:
	print("\n Length of Data: "+str(len(Data))+"\tBatchsize: "+str(batch_size))
	raise Exception("\n\nYou fucked up and now your Data Array does not match up with the specified batch size!")
elif  len(TestData) != batch_size:
	print("\n Length of Data: "+str(len(Data))+"\tBatchsize: "+str(batch_size))
	raise Exception("\n\nYou fucked up and now your TestData Array does not match up with the specified batch size!")
	
if len(TestData) != len(TestOH_CorrectPrediction):
	print("\n Length of TestData: "+str(len(Data))+"\tLength of TestPrdiction: "+str(len(OH_CorrectPrediction)))
	raise Exception("\n\nYou fucked up and now your TEST In/Out arrays don't match in length. It's a catastrophy!")	
	


#Placeholder
In = tf.placeholder(tf.float32,shape=[1,batch_size,NumOfSources])
Out = tf.placeholder(tf.float32,shape =[1,batch_size,num_features])
								
#Model 
lstm_size_l1 = 100
lstm_l1 = tf.contrib.rnn.BasicLSTMCell(lstm_size_l1)
lstm_out,state_l1 = tf.nn.dynamic_rnn(lstm_l1,In,time_major = False,dtype=tf.float32)
prediction = tf.contrib.layers.fully_connected(lstm_out,num_features,
												activation_fn = tf.nn.softmax)



#Cost
Cost = tf.reduce_sum(tf.pow(prediction-Out,2))
hit_rate = tf.equal(tf.argmax(prediction,axis = 2),tf.argmax(Out,axis = 2))
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
 				

		a = sess.run(Accuracy,feed_dict={In:N.reshape(TestData,(1,len(TestData),NumOfSources)),
		Out:N.reshape(TestOH_CorrectPrediction,(1,len(TestData),num_features))})
		
		c = sess.run(Cost,feed_dict={In:N.reshape(TestData,(1,len(TestData),NumOfSources)),
		Out:N.reshape(TestOH_CorrectPrediction,(1,len(TestData),num_features))})

								
		print("Epoch: "+str(Epoch)+"\tCost: "+str(c)+"\t Accuracy: "+str(a)+"%\t Market adjusted Accuracy:"+str(a-global_WinningBias))			
			
		sess.run(optimizer,feed_dict={In:N.reshape(Data,(1,batch_size,NumOfSources)),
		Out:N.reshape(OH_CorrectPrediction,(1,batch_size,num_features))})
		
	Pred = sess.run(prediction,feed_dict={In:N.reshape(TestData,(1,len(TestData),NumOfSources)),
		Out:N.reshape(TestOH_CorrectPrediction,(1,len(TestData),num_features))})

	plt.plot(Pred[0][:,0][500:750])
	plt.plot(TestOH_CorrectPrediction[:,0][500:750])
	plt.show()
	
	saver = tf.train.Saver()
	saver.save(sess,"LSTM_model.mod")