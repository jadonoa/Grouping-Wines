import numpy as np
from numpy import genfromtxt
import random
import copy

class matrix:

    #initialise a 2D matrix as a class attribute
    array_2d = np.empty([2,2])

    def __init__(self, filename = None):
        #load desired data upon object creation
        if filename == None:
            self.array_2d = np.ones([2,2])
        else:
            self.array_2d = self.load_from_csv(filename)

    def load_from_csv(self, filename):
        #load the data using a delimiter to separate values by a comma
        data = genfromtxt(filename, delimiter=',')
        return data

    def standardise(self):
        columnLength = len(self.array_2d)
        rowLength = len(self.array_2d[0])

        #for each column, find the maximum and minumum values
        for i in range(0, rowLength):
            sum = 0
            maxColumnValue = max(self.array_2d[:,i])
            minColumnValue = min(self.array_2d[:,i])

            #sum the all the values in each column and calculate mean
            for j in range(0, columnLength):
                sum += self.array_2d[j][i]

            meanColumn = sum/columnLength

            #apply the standarisation formula for values in a column
            for j in range(0, columnLength):
                self.array_2d[j][i] = (self.array_2d[j][i] - meanColumn) / (maxColumnValue - minColumnValue)

    def get_distance(self, other_matrix, weights, beta):
        numRowsW, numColsW = weights.array_2d.shape
        numRowsM, numColsM = self.array_2d.shape
        numRowsOM, numColsOM = other_matrix.array_2d.shape
        #returnMatrix will have the same number of rows as other_matrix
        returnMatrix = matrix()
        returnMatrix.array_2d = np.empty([numRowsOM, 1])

        #the weights matrix and matrix which called the method must have one row
        if (numRowsW != 1 or numRowsM != 1):
            print("Non-Compatible number of rows")
            return

        #for each row in other_matrix, apply the weighted euclidean distance formula
        for i in range(0, numRowsOM):
            sum = 0
            for j in range(0, numColsOM):
                sum += (weights.array_2d[0][j]**beta) * (self.array_2d[0][j] - other_matrix.array_2d[i][j])**2

            returnMatrix.array_2d[i][0] = sum

        return returnMatrix

    def get_count_frequency(self):
        numRowsM, numColsM = self.array_2d.shape
        dict = {}

        #matrix that called this method must have one column
        if (numColsM != 1):
            print("Non-Compatible number of columns")
            return

        #for each row
        for i in range(0, numRowsM):
            #if the value in the row is not already in the dictionary
            if self.array_2d[i][0] not in dict:
                #count the number of occurences of that value and store in dictionary
                column_occurrences = np.count_nonzero(self.array_2d == self.array_2d[i][0], axis = 0)
                dict[self.array_2d[i][0]] = column_occurrences[0]

        return dict

def get_initial_weights(m):
    #initialise a matrix with one row and m columns
    weights = matrix()
    weights.array_2d = np.empty([1, m])
    sum = 0

    #for each value in the matrix, generate and store a random number
    for i in range(0, m):
        #generate a random number between 0 and 1
        randomNum = random.uniform(0,1)
        weights.array_2d[0][i] = randomNum
        sum += randomNum

    #divide each value in the matrix by the sum of those values and store again
    #the sum of these new values are still random but now sum to 1
    for j in range(0, m):
        weights.array_2d[0][j] = (weights.array_2d[0][j] / sum)

    return weights

def get_centroids(data_matrix, S, K):
    numRowsM, numColsM = data_matrix.array_2d.shape
    centroids = matrix()
    centroids.array_2d = np.empty([K, numRowsM])

    #complete the update for each row K in centroids
    for i in range(0, K):
        sum = 0
        #find the indicies of rows which are assigned to the value K
        result = np.where(S.array_2d == i)
        rowsofK = result[0]

        #if the current value of K has more than one row assigned to it
        #update the k row in centroids
        if rowsofK.size > 0:
            for j in range(0, numColsM):
                for index in rowsofK:
                    sum += data_matrix.array_2d[index][j]
                mean = (sum / rowsofK.size)
                centroids.array_2d[i][j] = mean

    return centroids

def get_groups(data_matrix, K, beta):
    numRowsM, numColsM = data_matrix.array_2d.shape
    #check if K and beta are postive integers
    if not K > 1 or not beta > 0:
        print("Values of K must be more than 1, K and beta must be positive integers")
        return

    #check if K is less than n-1
    if not K < numRowsM:
        print("Value of K must be less than total number of rows in the data")
        return

    data_matrix.standardise()
    weights = get_initial_weights(numColsM)

    #initialise centroids and S matricies
    centroids = matrix()
    centroids.array_2d = np.empty([2,2])
    prevS = matrix()
    prevS.array_2d = np.ones([numRowsM,1])
    S = matrix()
    S.array_2d = np.zeros([numRowsM,1])
    #select random incicies from 0 to the number of rows
    random_indices = np.random.choice(numRowsM, size=K, replace=False)
    #store the K rows for these indicies in centroids
    centroids.array_2d = data_matrix.array_2d[random_indices, :]

    #continue updating the values in S and weights until S does not change
    while(not np.array_equal(prevS.array_2d, S.array_2d)):
        prevS.array_2d = S.array_2d
        #for each row in matrixRow
        for i in range(0, numRowsM):
            #copy a matrix row into another matrix object so we have only one row
            matrixRow = copy.deepcopy(data_matrix)
            matrixRow.array_2d = data_matrix.array_2d[i]
            matrixRow.array_2d = np.reshape(matrixRow.array_2d, (1, 13))
            #get the distances between a matrix row and centroids
            returnMatrix = matrixRow.get_distance(centroids, weights, beta)
            #the nearest centroid to the matrix row is the smallest value in returnMatrix
            nearestCentroid = np.argmin(returnMatrix.array_2d, axis=0)
            #store the index of this centroid in S
            S.array_2d[i] = nearestCentroid[0]

        centroids = get_centroids(data_matrix, S, K)
        weights = get_new_weights(data_matrix, centroids, S)

    return S


def get_new_weights(data_matrix, centroids, S):
    numRowsM, numColsM = data_matrix.array_2d.shape
    numRowsK, numColsK = centroids.array_2d.shape
    new_weights = matrix()
    new_weights.array_2d = np.empty([1, numColsM])
    deltaJs = np.empty([1, numColsM])
    beta = 2

    #calculate the dispersion of column j in data matrix
    for j in range(0, numColsM):
        sumk = 0
        #calculate outer sum
        for k in range(0, numRowsK):
            sumi = 0
            #calculate inner sum
            for i in range(0, numRowsM):
                #if statement to determine the value of u
                if S.array_2d[i] == k:
                    u = 1
                else:
                    u = 0
                #use formula to calculate sum
                sumi += (u * (data_matrix.array_2d[i][j] - centroids.array_2d[k][j])**2)
            sumk += sumi
        deltaJs[0][j] = sumk

    #for each column j, calculate its new weight
    for j in range(0, numColsM):
        if deltaJs[0][j] == 0:
            new_weights.array_2d[0][j] = 0
        else:
            #calculate the denominator to get new weight of column j
            sum = 0
            for i in range(0, numColsM):
                sum += ((deltaJs[0][j] / deltaJs[0][i])**(1/beta-1))
            new_weights.array_2d[0][j] = 1 / sum

    return new_weights

def run_test():
    m = matrix("Data.csv")
    for k in range(2,5):
        for beta in range(11,25):
            S = get_groups(m, k, beta/10)
            print(str(k)+"-"+str(beta)+"="+str(S.get_count_frequency()))

run_test()
