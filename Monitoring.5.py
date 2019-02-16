import matplotlib.pyplot as plt
import numpy as N
import tensorflow as tf

#Parameters
forecast = 10 #days
batch_size = 10
num_of_epochs = 4000


#Data Import
def ImportData():
	Data = N.genfromtxt("CompleteStock.csv",delimiter=",")

	print(str(len(Data))+" Timesteps with "+str(len(Data[0]))+" Features have been imported.")
	
	Labels = (Data[:,6][forecast:]-Data[:,6][:-forecast])

	Labels = (N.sign(Labels)+1.0)/2.0

	return Data,Labels
	
def OneHot(BinList,features=2):

	BinList = BinList.astype(int)

	#Converts Binary List of ones and zeros to One Hot Encoding
	Encoded = N.zeros((len(BinList),features))
	Encoded[N.arange(len(BinList)),BinList] = 1
	
	GlobalWinningBias = 100*N.sum(Encoded[:,1])/len(Encoded)

	print("Global Winning Bias: "+str(round(GlobalWinningBias,3))+"%")
	
	return Encoded
	
#Plotting
def Plotbinary(List):

	Change = List[1:]-List[:-1]
	Change = N.insert(Change,0,List[0])
	
	Events = [0]
	
	for i in range(len(Change)):
		if Change[i] != 0:
			Events.append(i)
	Events.append(len(List)-1)
	
			
	for i in range(len(Events)-1):
		if Change[Events[i]] > 0:
			plt.axvspan(Events[i],Events[i+1],facecolor = "green",alpha = 0.5,edgecolor="none" )
		elif Change[Events[i]] < 0:
			plt.axvspan(Events[i],Events[i+1],facecolor = "red",alpha = 0.5,edgecolor="none" )

Data,Labels = ImportData()
OH_Labels = OneHot(Labels)
NumFeatures = len(Data[0])

#TF Variables
global_stddev = 0.01
In = tf.placeholder(tf.float32,[batch_size,NumFeatures])
Out = tf.placeholder(tf.float32,[batch_size,2])

init_state = tf.Variable(tf.zeros([1,NumFeatures]))

W = tf.Variable(tf.random_normal([NumFeatures,NumFeatures],stddev = global_stddev))
b = tf.Variable(tf.random_normal([NumFeatures],stddev = global_stddev))

W2 = tf.Variable(tf.random_normal([NumFeatures,NumFeatures],stddev = global_stddev))
b2 = tf.Variable(tf.random_normal([NumFeatures],stddev = global_stddev))

W3 = tf.Variable(tf.random_normal([NumFeatures,2],stddev = global_stddev))



#Single RNN Pass
def RNN_step(h_1,X):

	X = tf.expand_dims(X,0)

	t = tf.matmul(X,W)
	t_1 = tf.matmul(h_1,W2)

	t = tf.add(t,b)
	t_1 = tf.add(t_1,b2)
	
	h = tf.add(t,t_1)
	
	o = tf.matmul(h,W3)
	
	o = tf.squeeze(o)
	
	ex = tf.exp(o)
	sm = ex/tf.reduce_sum(ex)
	
	return h,sm

def RNN(X):

	X = tf.unstack(X,axis=0)
		
	h,Sm = RNN_step(init_state,X[0])

	States = [h]
	Outputs = [Sm]

	for Input in X[1:]:
		h,Sm = RNN_step(States[-1],Input)
		
		States.append(h)
		Outputs.append(Sm)
		
	Outputs = tf.stack(Outputs)
	States = tf.stack(States) 
	
	return Outputs,States
	
def Performance(Output,Labels):
	
	Entropy = Labels*tf.log(Output)+(1-Labels)*tf.log(1-Output)	
	Mean_Entropy = tf.reduce_sum(Entropy)/batch_size
	
	Argmax = tf.cast(tf.argmax(Output,axis = 1),tf.float32)
	Equal = tf.cast(tf.equal(Argmax,Labels[:,1]),tf.float32)
	Accuracy = (tf.reduce_sum(Equal)/batch_size)*100.0

	
	return Mean_Entropy,Accuracy
	
Prediction,States = RNN(In)
Loss,Accuracy = Performance(Prediction,Out)

optimizer = tf.train.MomentumOptimizer(0.01,0.01).minimize(Loss)

init = tf.global_variables_initializer()

Loss_overtime = []
Accuracy_overtime = []
	
with tf.Session() as sess:

	sess.run(init)

	for epoch in range(num_of_epochs):	
		if epoch%10 == 0:
			Loss_eval,Accuracy_eval,Prediction_eval = sess.run((Loss,Accuracy,Prediction),feed_dict={In:Data[-batch_size:],Out:OH_Labels[-batch_size:]})
	
			print("\nEpoch: "+str(epoch)+"\tAccuracy: "+str(round(Accuracy_eval,2))+"\tLoss: "+str(round(Loss_eval,2)))
			print("\n\n   Prediction \t\t\t	Label \n")
			for i in range(batch_size-10,batch_size):
				print(str(Prediction_eval[i])+"  \t\t"+str(OH_Labels[i]))
			print("-----------------")
			
			Loss_overtime.append(Loss_eval)
			Accuracy_overtime.append(Accuracy_eval)
	
		part = epoch%3
		sess.run(optimizer,feed_dict={In:Data[part*batch_size:(part+1)*batch_size],Out:OH_Labels[part*batch_size:(part+1)*batch_size]})
	
#plt.plot(Accuracy_overtime)	
plt.plot(Loss_overtime)	
plt.show()
	
	