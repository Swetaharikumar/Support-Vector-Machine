import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix
from cvxopt import solvers


class SVM:
    
    # Parse files into numpy arrays
    def parse_text_file(self,path):
        f = open(path,"r")
        if f.mode == 'r':
            contents = np.loadtxt(f)
            return contents
        
        
    # Calculate mean and standard deviation of training data    
    def mean_sd(self,X_train):
        self.mean = np.mean(X_train,axis = 0)
        self.sd = np.std(X_train,axis=0,ddof=1)
        return self.mean,self.sd
   
    # Preprocess data by subtracting mean and dividing by standard deviation of each feature
    def pre_processing(self,X):
        return np.divide(X - self.mean, self.sd)
    
    
    
    # Train classifier by calculating P,Q,G,H for feeding to cvxopt (optimization library)
    def train_svm(self,train_data, train_label, C):
        size = train_data.shape[0] + train_data.shape[1]+1
        P_temp = np.zeros((size,size))
        
        for i in range(train_data.shape[1]):
            P_temp[i][i] = 1
        P = matrix(P_temp, tc = "d")
        
        Q_temp1 = np.ones((train_data.shape[0],1)) * C
        Q_temp2 = np.zeros((train_data.shape[1]+1,1))
        Q = matrix(np.concatenate((Q_temp2,Q_temp1),axis=0), tc = "d")
       
 
        train_label = np.reshape(train_label,(train_data.shape[0],1))
        G_temp = (train_data * train_label)
        G_temp = np.concatenate((G_temp,train_label),axis=1)
        G_temp = np.concatenate((G_temp,np.identity(train_data.shape[0])),axis =1)
        
        temp1 = np.identity(train_data.shape[0])
        temp2 = np.zeros((train_data.shape[0],train_data.shape[1] +1))
    
        
        temp3 = np.concatenate((temp2,temp1),axis = 1)
        
        G = matrix(-1 * np.concatenate((G_temp,temp3),axis =0))
       
        H_temp = np.ones((train_data.shape[0],1))
        H_temp2 = np.zeros((train_data.shape[0],1))
        
        H = matrix(-1 * np.concatenate((H_temp,H_temp2), axis = 0))
        sol = solvers.qp(P,Q,G,H)
        solution = sol["x"]
        b = np.asarray(solution[60])
        w = np.asarray(solution[:60])
        return w,b
  
        
    # Test data using calculated optimal weights
    # b - bias term
    def test_svm(self,test_data,test_label,w,b):
        pred = (np.matmul(test_data,w) + b)
        test_label = test_label.reshape((test_label.shape[0],1))
        pred = pred * test_label
        correct = pred > 0
        numerator = np.sum(correct,axis=0)
        accuracy = numerator/test_label.shape[0] * 100
        return accuracy

# Please uncomment below code and add appropriate paths
# path_train_data = '/Users/swetaharikumar/Desktop/SupportVectorMachines/train_data.txt'
# path_train_label = '/Users/swetaharikumar/Desktop/SupportVectorMachines/train_label.txt'
# path_test_data = '/Users/swetaharikumar/Desktop/SupportVectorMachines/test_data.txt'
# path_test_label = '/Users/swetaharikumar/Desktop/SupportVectorMachines/test_label.txt'


model = SVM()
X_train = model.parse_text_file(path_train_data)
Y_train = model.parse_text_file(path_train_label)
X_test = model.parse_text_file(path_test_data)
Y_test = model.parse_text_file(path_test_label)

mean,sd = model.mean_sd(X_train) 
X_train_norm = model.pre_processing(X_train)
X_test_norm = model.pre_processing(X_test)


C = []
for i in range(-6,7):
    C.append(4**i)


# Perform 5-fold cross validation on the training set and calculate time taken for training and accuracy
import time
accuracy_C = []
time_elapsed = []
for i in range(len(C)):
    start_time = time.time()
    accuracy_1 = []
    w,b = model.train_svm(X_train_norm[200:1000][:],Y_train[200:1000],C[i])
    accuracy_1.append(model.test_svm(X_train_norm[0:200][:],Y_train[0:200],w,b))
    
    X = np.concatenate((X_train_norm[0:200][:],X_train_norm[400:1000][:]),axis = 0)
    Y = np.concatenate((Y_train[0:200],Y_train[400:1000]),axis =0)
    w,b = model.train_svm(X,Y,C[i])
    accuracy_1.append(model.test_svm(X_train_norm[200:400][:],Y_train[200:400],w,b))
    
    X = np.concatenate((X_train_norm[0:400][:],X_train_norm[600:1000][:]),axis = 0)
    Y = np.concatenate((Y_train[0:400],Y_train[600:1000]),axis =0)
    w,b = model.train_svm(X,Y,C[i])
    accuracy_1.append(model.test_svm(X_train_norm[400:600][:],Y_train[400:600],w,b))
    
    X = np.concatenate((X_train_norm[0:600][:],X_train_norm[800:1000][:]),axis = 0)
    Y = np.concatenate((Y_train[0:600],Y_train[800:1000]),axis =0)
    w,b = model.train_svm(X,Y,C[i])
    accuracy_1.append(model.test_svm(X_train_norm[600:800][:],Y_train[600:800],w,b))
    
    X = X_train_norm[0:800][:]
    Y = Y_train[0:800]
    w,b = model.train_svm(X,Y,C[i])
    accuracy_1.append(model.test_svm(X_train_norm[800:1000][:],Y_train[800:1000],w,b))
    time_elapsed.append((time.time()-start_time)/5)
    
    
    
    accuracy_C.append(np.mean(np.asarray(accuracy_1)))



for i in range(len(C)):
    print("The cross validation accuracy for C = " + str(C[i]) + " is " + str(accuracy_C[i]) + " and training time is " + str(time_elapsed[i]))

w,b = model.train_svm(X_train_norm,Y_train,C[3])
accuracy_test = model.test_svm(X_test_norm,Y_test,w,b)

print("The accuracy for optimal C on the test set is " + str(accuracy_test))

