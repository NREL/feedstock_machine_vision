# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:44:11 2019

@author: cgudaval
"""

import numpy as np
def generalizeAlarm(actual, predicted, leftThresh = float("inf"), rightThresh = float("inf")):
    result = np.zeros(actual.shape)
    
    for i in range(len(result)):
        if (predicted[i] == 1) and (actual[i] == 1) and (result[i] != 1):
            result[i] = 1
            
            #Fill left ones
            left_i = i-1
            leftStop = False
            time = 0
            while (leftStop == False and left_i > 0):
                if actual[left_i] == 1 and time < leftThresh:
                    result[left_i] = 1
                    time += 1
                else:
                    leftStop = True
                    
                left_i = left_i - 1
                
            #Fill ones in right side
            right_i = i+1
            rightStop = False
            time = 0
            while (rightStop == False and right_i < len(result)):
                if actual[right_i] == 1 and time < rightThresh:
                    result[right_i] = 1
                    time += 1
                else:
                    rightStop = True
                    
                right_i = right_i + 1
        elif result[i] != 1:
            result[i] = predicted[i]
    
    return result

#actual =    np.array([0,0,1,1,1,1,1,1,0,0,1,1,1,1,0,0,1,1])
#predicted = np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1])
#
#
#result = generalizeAlarm(actual, predicted, 2)
#print(actual)
#print(predicted)
#print(result.astype(int))