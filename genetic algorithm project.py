#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1 - Import the dataset from the .csv file provide
dataset = pd.read_csv('pro1_dataset.csv')
dataset = dataset.iloc[:,0:14]


# 2 - Choose N=10 (see STRUCTURE on slide #2)
N=10

# 3 – Only select the first 5 columns as input, and the very 
# last column as output (target). You can eliminate the rest of the columns
dataset_x = dataset.iloc[:, :-9].values
dataset_y = dataset.iloc[:, 13].values


# 4 - Choose 25% of the dataset (random) as testing and the rest 75% 
# as training samples. Leave the testing dataset on the side for the time being
from sklearn.cross_validation import train_test_split
dataset_x_train, dataset_x_test, dataset_y_train, dataset_y_test = train_test_split(dataset_x, dataset_y, test_size = 0.25)

# 5 – Normalise the training dataset with values between 0 and 1
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
normalized_dataset_x_train = min_max_scaler.fit_transform(dataset_x_train)
normalized_dataset_y_train = (dataset_y_train-min(dataset_y_train))/(max(dataset_y_train)-min(dataset_y_train))


# 6 – Calculate the number of parameters (weights) you need to tune 
# in the STRUCTURE (refer to slide #2). You need to tune PxN parameters (weights).
# NOTE : For each weight (parameter) you need 10 bits for binarized 
# version of it. Therefore, your long ‘chromosomes’ are having length 10xPxN bits (genes).

Npop=500
whole_pop_mat = np.zeros((Npop,5*10))
fitness_value=np.zeros(Npop)
for r in range (Npop):
    yhat= np.zeros(len(normalized_dataset_x_train))
    #7 – Create randomly around Npop=500 (should be large number, 
    # so feel free to have more even) initial population of parameters(solutions).
    W=np.zeros((5, N))
    pop_mat=[]
    for j in range (5):
        random_population_of_possible_weights=np.random.random(10)
        random_population_of_possible_weights=random_population_of_possible_weights*2-1
        W[j,:]=random_population_of_possible_weights
        if j>0:
            # creating each chromosome
            pop_mat=np.append(pop_mat,random_population_of_possible_weights)
        else:
            pop_mat = random_population_of_possible_weights
        
    for i in range (len(normalized_dataset_x_train)):
        X_mat=np.matmul(normalized_dataset_x_train[i,:],W)
        summ=0.0
        # applying eq1
        for k in range (N):
            summ=summ+(1/(1+np.exp(-X_mat[k])))
        yhat[i] = summ
    
    
    total=0
    for k in range (len(normalized_dataset_y_train)):
        total = total + (yhat[k]-normalized_dataset_y_train[k])*(yhat[k]-normalized_dataset_y_train[k])
    # 8 – Calculate the fitness_values via Eq(2) for each solution.
    fitness_value[r] = (1 - total/len(normalized_dataset_y_train))*100
    
    if r>1:
        if max(fitness_value[0:r-1])<fitness_value[r]:
            W_max=W
            X_mat_max=X_mat
            # Via indexx variable, I determine best chromosome
            indexx=r
    else:
        W_max=W
        X_mat_max=X_mat
        indexx=r
    whole_pop_mat[r,] = pop_mat


# 10 - Binarize the parent and all other population according to the following procedure :
#I ) For each single parameter (weight) - it should be a figure between -1 and 1 , normalise the weights to numbers between 0 and 1
#II ) Multiply the normalised figures to 1000. Your figures are now float numbers less than 1000.
#III ) Round the numbers to closest integer. Now you have integer numbers less than 1000.
#IV ) Get the base-2 (binary) 10-bit conversion of the weights.
#NOTE : Make sure for each binary weight you have fix 10 bits allocated.

normalized_whole_pop_mat = min_max_scaler.fit_transform(whole_pop_mat)
normalized_whole_pop_mat = normalized_whole_pop_mat*1000
normalized_whole_pop_mat = np.floor(normalized_whole_pop_mat)
normalized_whole_pop_mat = normalized_whole_pop_mat.astype(int)
# binarizing and creating binarized matrix
binarized_whole_pop_mat = np.zeros((Npop,5*N*10))
for i in range (Npop):
    row_mat = np.zeros(10)
    # 11 – Concatenate all 10-bit weights along each other and make a ‘chromosome’ .
    #NOTE : Please remember the order you align the weights in the chromosome , because later you need to de-segment the chromosome and put each weight in its own place in the STRUCTURE (refer to slide #2), to produce 𝑦ො .
    for j in range (5*N):
        binarized = bin(normalized_whole_pop_mat[i,j])[2:].zfill(10)
        if j<1:
            row_mat = list(binarized)
        else:
            row_mat = row_mat+ list(binarized)
    binarized_whole_pop_mat[i,]=row_mat

# 9 – Select the solution with highest fitness_values (fittest). This is parent 
# now (or you can call it sire).
parent = binarized_whole_pop_mat[indexx,:]

# 12– Do the Cross-Over of the parent, with each single member of Npop and create two offsprings from each. Now your population is increased by 2xNpop
offspring = np.zeros((Npop,5*N*10))
for i in range (500):
    c_point = np.random.random_integers(2, 499)
    offspring[i] = np.concatenate((parent[0:c_point], binarized_whole_pop_mat[i, c_point:]))

# 13– Do the mutation for each newly born chromosome.
for i in range (500):
    t_f = np.random.random_integers(0,499,25)
    for j in range (25):
        if offspring[i,t_f[j]]== 0:
            offspring[i,t_f[j]]= 1
        else:
            offspring[i,t_f[j]]= 0

# combining both matrices which are binarized population matrix and offspring matrix
pop_offsprings= np.concatenate((binarized_whole_pop_mat,offspring))

# 14– Do the de-binarization of the chromosomes according to following procedure :
#I ) De-segment each chromosome to its 10-bits components.
#II ) Make a binary to decimal conversion of each single 10-bit weight.
#III ) Divide them by 1000
#IV ) De-normalise weights to values between -1 and 1
decimal_x = np.zeros((1000,50))
for i in range (len(pop_offsprings)):
    for j in range (50):
        binary_x = pop_offsprings[i,j*10:j*10+10]
        binary_x= np.array2string(binary_x.astype(int))
        binary_x= binary_x.replace(" ", "")
        #print(''.join(binary_x))
        decimal_x[i,j]= int(binary_x[1]+binary_x[2]+binary_x[3]+binary_x[4]+binary_x[5]+binary_x[6]+binary_x[7]+binary_x[8]+binary_x[9]+binary_x[10],2)

decimal_x= decimal_x/1000
# Denormalization
# I keep denormalized values of both population and offsprings in decimal_x variable
# decimal_x is the most important variable in my coding schema, because iterations 
# continues by changing decimal_x with better population matrices
decimal_x = decimal_x*2-1


# 18 – Iterate from step 12 to 17. Each time you do steps 12 -17, one iteration is elapsed. You iterate until the highest fitness_value reaches to a plateau (like a ‘while’ loop). 
keep_fitness_values=np.zeros(30)
for aa in range (30):   
    fitness_value2=np.zeros(Npop*2)
    whole_pop_mat2 = np.zeros((Npop*2,5*10))
    for s in range (1000): 
        yhat2= np.zeros(len(normalized_dataset_x_train))
        pop_mat2=[]
        W2=np.zeros((5, N))
        for j in range (5):
            W2[j,:]=decimal_x[s,j*10:(j+1)*10]
            if j>0:
                pop_mat2=np.append(pop_mat2,decimal_x[s,j*10:(j+1)*10])
            else:
                pop_mat2 = decimal_x[s,j*10:(j+1)*10]
            
        for i in range (len(normalized_dataset_x_train)):
            X_mat2=np.matmul(normalized_dataset_x_train[i,:],W2)
            summ=0.0
            for k in range (N):
                summ=summ+(1/(1+np.exp(-X_mat2[k])))
                yhat2[i] = summ
    
        
        total=0
        for k in range (len(normalized_dataset_y_train)):
            total = total + (yhat2[k]-normalized_dataset_y_train[k])*(yhat2[k]-normalized_dataset_y_train[k])
        # 15 – Calculate the fitness_value for all population from Eq(2).
        fitness_value2[s] = (1 - total/len(normalized_dataset_y_train))*100
      
        if s>1:
            if max(fitness_value2[0:s-1])<fitness_value2[s]:
                W_max2=W2
                X_mat_max2=X_mat2
                indexx2=s
        else:
            W_max2=W2
            X_mat_max2=X_mat2
            indexx2=s
        whole_pop_mat2[s,] = pop_mat2
    
    # 16 – Eliminate the lowest fitness_value chromosomes. Now the population is reduced back from 2xNpop to Npop again.    
    sort_index_fitness_value2 = np.argsort(fitness_value2)
    whole_pop_mat2_first500=whole_pop_mat2[sort_index_fitness_value2[500:1000]]
    keep_fitness_values[aa] = max(fitness_value2)
    


    normalized_whole_pop_mat2 = min_max_scaler.fit_transform(whole_pop_mat2_first500)
    normalized_whole_pop_mat2 = normalized_whole_pop_mat2*1000
    normalized_whole_pop_mat2 = np.floor(normalized_whole_pop_mat2)
    normalized_whole_pop_mat2 = normalized_whole_pop_mat2.astype(int)
    
    binarized_whole_pop_mat2 = np.zeros((Npop,5*N*10))
    for i in range (Npop):
        row_mat2 = np.zeros(10)
        for j in range (5*N):
            binarized2 = bin(normalized_whole_pop_mat2[i,j])[2:].zfill(10)
            if j<1:
                row_mat2 = list(binarized2)
            else:
                row_mat2 = row_mat2+ list(binarized2)
        binarized_whole_pop_mat2[i,]=row_mat2
    
    # 17 – Save the chromosome with highest fitness_value as the parent. If by any chance the highest fitness_value was less than previous iteration, keep previous iteration parent as current parent.
    #parent2 = binarized_whole_pop_mat2[indexx2,:]
    parent2 = binarized_whole_pop_mat2[499,:]
    
    offspring2 = np.zeros((Npop,5*N*10))
    for i in range (500):
        c_point2 = np.random.random_integers(2, 499)
        offspring2[i] = np.concatenate((parent2[0:c_point2], binarized_whole_pop_mat2[i, c_point2:]))
    
    for i in range (500):
        t_f2 = np.random.random_integers(0,499,25)
        for j in range (25):
            if offspring2[i,t_f2[j]]== 0:
                offspring2[i,t_f2[j]]= 1
            else:
                offspring2[i,t_f2[j]]= 0
    
    pop_offsprings2= np.concatenate((binarized_whole_pop_mat2,offspring2))
    
    decimal_x = np.zeros((1000,50))
    for i in range (len(pop_offsprings2)):
        for j in range (50):
            binary_x2 = pop_offsprings2[i,j*10:j*10+10]
            binary_x2= np.array2string(binary_x2.astype(int))
            binary_x2= binary_x2.replace(" ", "")
            decimal_x[i,j]= int(binary_x2[1]+binary_x2[2]+binary_x2[3]+binary_x2[4]+binary_x2[5]+binary_x2[6]+binary_x2[7]+binary_x2[8]+binary_x2[9]+binary_x2[10],2)
    
    decimal_x= decimal_x/1000
    # Denormalization
    decimal_x = decimal_x*2-1
    
    
# Scatter plot highest fitness values for each iteration (15%)    
ite_number= range(0, 30)
plt.scatter(ite_number, keep_fitness_values, c="red", alpha=0.4)
plt.title('Highest Fitness Values of Each Iteration')
plt.xlabel('Iterations')
plt.ylabel('Highest fitness values')
plt.show()



# Find out overall error for testing dataset (10%)
normalized_dataset_x_test = min_max_scaler.fit_transform(dataset_x_test)
normalized_dataset_y_test = (dataset_y_test-min(dataset_y_test))/(max(dataset_y_test)-min(dataset_y_test))

fitness_value_test=np.zeros(Npop*2)
for r in range (Npop*2):
    yhat_test= np.zeros(len(normalized_dataset_x_test))
    #7 – Create randomly around Npop=500 (should be large number, 
    # so feel free to have more even) initial population of parameters(solutions).
    W_test=np.zeros((5, N))
    pop_mat_test=[]
    for j in range (5):
        #random_population_of_possible_weights_test=np.random.random(10)
        #random_population_of_possible_weights_test=random_population_of_possible_weights_test*2-1
        W_test[j,:]=decimal_x[s,j*10:(j+1)*10]
        if j>0:
            pop_mat_test=np.append(pop_mat_test,decimal_x[s,j*10:(j+1)*10])
        else:
            pop_mat_test = decimal_x[s,j*10:(j+1)*10]

    for i in range (len(normalized_dataset_x_test)):
        X_mat_test=np.matmul(normalized_dataset_x_test[i,:],W_test)
        summ=0.0
        for k in range (N):
            summ=summ+(1/(1+np.exp(-X_mat_test[k])))
        yhat_test[i] = summ
    
overall_error = 0
for o in range (len(normalized_dataset_x_test)):
    overall_error= overall_error+(yhat_test[o] - normalized_dataset_y_test[o])*(yhat_test[o] - normalized_dataset_y_test[o])
overall_error = overall_error/len(normalized_dataset_y_test)
print("Overall error is: "+ "{0:.2f}%".format(overall_error))


# Scatter plot in 3D, the first and second output and estimated output yhat, together with real output y with different colors
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
# Data for three-dimensional scattered points
zdata_yhat = yhat_test
xdata = normalized_dataset_x_test[:,0]
ydata = normalized_dataset_x_test[:,1]
ax.scatter3D(xdata, ydata, zdata_yhat, c=zdata_yhat, cmap='Reds');   
zdata_real = normalized_dataset_y_test
ax.scatter3D(xdata, ydata, zdata_real, c=zdata_real, cmap='Blues'); 
plt.title('Scatter Plot in 3D together with yhat and real output y')
plt.xlabel('First Input')
plt.ylabel('Second Input')
plt.show()