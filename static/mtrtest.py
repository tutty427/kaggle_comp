from numpy import *
import operator
from os import listdir


# mat1 = mat([[0,0,1,1,0],[0,0,1,1,0],[0,0,1,1,0],[0,0,1,1,0],[0,0,1,1,0]]);

mat1 = tile([0,3,2,3,0], (5,1))
diffmat = mat1**2
sqDistances = diffmat.sum(axis=1)
print(sqDistances)
distances = sqDistances**0.5
print(distances)
sortedDistIndicies = distances.argsort()
print(sortedDistIndicies)
#mat2 = mat([[1,0,1,0,1],[0,1,0,1,0],[1,0,0,0,1],[1,1,1,1,1],[1,0,0,1,1]]);

# diffMat = mat1 - mat2
# print diffMat
# print transpose(diffMat)
# sqDiffMat = diffMat * transpose(diffMat)
# print sqDiffMat
# # disMat = sqDiffMat.sum(axis=1);
# # print disMat
# # distances = disMat**0.5
# # print distances
# trainingMat = zeros((5,5))
#
# mat3 = tile([1,2,3,4,5], (5,1))
# returnVect = zeros((1,5))
# returnVect[0,0] = 1
# returnVect[0,1] = 0
# returnVect[0,2] = 1
# returnVect[0,3] = 0
# returnVect[0,4] = 1
# trainingMat[0] = returnVect
#
#
# returnVect1 = zeros((1,5))
# returnVect1[0,0] = 0
# returnVect1[0,1] = 0
# returnVect1[0,2] = 1
# returnVect1[0,3] = 1
# returnVect1[0,4] = 0
# trainingMat[1] = returnVect1
#
# returnVect2 = zeros((1,5))
# returnVect2[0,0] = 1
# returnVect2[0,1] = 0
# returnVect2[0,2] = 0
# returnVect2[0,3] = 0
# returnVect2[0,4] = 1
# trainingMat[2] = returnVect2
#
# returnVect3 = zeros((1,5))
# returnVect3[0,0] = 1
# returnVect3[0,1] = 0
# returnVect3[0,2] = 0
# returnVect3[0,3] = 1
# returnVect3[0,4] = 1
# trainingMat[3] = returnVect3
#
#
# returnVect4 = zeros((1,5))
# returnVect4[0,0] = 1
# returnVect4[0,1] = 1
# returnVect4[0,2] = 1
# returnVect4[0,3] = 1
# returnVect4[0,4] = 1
# trainingMat[4] = returnVect4
#
# mat2 =  trainingMat
#
# diffMat = mat1 - mat2
# print diffMat
#
# sqdiffMat = diffMat ** 2
# print sqdiffMat
#
# dis = sqdiffMat.sum(axis=1)
# d = dis**0.5
# sortedDistIndicies = d.argsort()
# print sortedDistIndicies[0]
#
# arr1 = array([0,0,1,1,0]);
# arr2 = array([1,0,1,0,1]);
# diffArray = (arr1 - arr2)**2
# print diffArray