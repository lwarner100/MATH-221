import numpy as np
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import random
import matplotlib
from matplotlib import pyplot as plt

class Model:
    
    def __init__(self, x, y, degree=2, exp=False, predictive=False):
        self.x = x
        self.y = y
        self.exp = exp
        self.xrange = np.linspace(min(self.x),1.5*max(self.x),100) if predictive\
                      else np.linspace(min(self.x),max(self.x),100)
        self.degree = degree
        self.ln = np.lib.scimath.log
        self.logx, self.logy = self.filterZeroes(self.x,self.y,logy=True)
        self.xvec = self.createVec(self.x) 
        self.yvec = self.createVec(self.y) if not exp else\
                    self.createVec(self.logy)
        self.matrix = self.createMatrix(self.x,self.degree) if not exp else\
                      self.createMatrix(self.logx,1)
        self.coeff = self.coefficientApprx(self.matrix) if not exp else\
                     [np.e**float(i) for i in self.coefficientApprx(self.matrix)]
        self.strfunc = self.stringFunc(self.coeff)
        self.latexfunc = self.latexFunc(self.coeff)
        self.ypred = self.yPrediction()
        self.ypred_x = self.yPrediction_onX()

        
        

    def createMatrix(self,x_values,degree_=2):
        rows = [[xval**i for i in range(degree_+1)] for xval in x_values]
        return matrix(rows)


    def createVec(self,*args):
        if isinstance(args[0],(list,tuple,np.ndarray)):
            argument = [[args[0][i]] for i in range(len(args[0]))]
            vec = matrix(argument)
        else:
            argument = [[i] for i in args]
            vec = matrix(argument)
        return vec

    def unique(self,arr):
        used = []
        result = []
        for i in arr:
            if i not in used:
                used.append(i)
                result.append(i)
        result.sort()
        return result

    def filterZeroes(self,x,y,no_negatives=True,logy=False):
        new_x = []
        new_y = []
        if no_negatives:
            for i in zip(x,y):
                if i[1] > 0 and i[0] != 0:
                    new_x.append(i[0])
                    new_y.append(i[1])
        else:
            for i in zip(x,y):
                if 0 not in i:
                    new_x.append(i[0])
                    new_y.append(i[1])
        new_y = self.ln(new_y) if logy else new_y
        return new_x, new_y

    def func(self,value):
        coefficients = [float(i) for i in self.coeff]
        if not self.exp:
            func_ = lambda x: sum([x**i*coefficients[i] for i in range(len(coefficients))])
        else:
            func_ = lambda x: coefficients[0] * coefficients[1]**x
        return func_(value)


    def coefficientApprx(self,A_matrix):
        A_t = np.transpose(A_matrix)
        A_t_A = A_t*A_matrix
        invA_t_A = (A_t_A)**-1
        z = invA_t_A*A_t*self.yvec
        return z

    def stringFunc(self,z):
        if not self.exp:
            string = f'{1*np.round(float(z[0]),4)}'
            for i in range(1,len(z)):
                string += f' + {np.round(float(z[i]),4)}*x^{i}'
        else:
            string = f'{np.round(self.coeff[0],3)} * {np.round(self.coeff[1],3)}^x'
        return string
    
    def latexFunc(self,c):
        y = 'y'
        z = [float(i) for i in c]
        if not self.exp:
            string = rf'$\hat{y} = {np.round(z[0],2)} $ '
            if len(z) > 1:
                if z[1] > 0:
                    string += rf'$+$ ${np.round(z[1],2)}x$ '
                elif z[1] < 0:
                    string += rf'$-$ ${np.round(abs(z[1]),2)}x$ '

            for i in range(2,len(z)):
                if z[i] >= 0:
                    string += rf'$+$ ${np.round(z[i],2)}x^{i}$ '
                elif z[i] < 0:
                    string += rf'$-$ ${np.round(abs(z[i]),2)}x^{i}$ '
        else:
            string = rf'$\hat{y} = $ ${np.round(self.coeff[0],2)}$ $\cdot$ ${np.round(self.coeff[1],2)}^x $'
        return string

    def yPrediction_onX(self):
        y_apprx = [self.func(x) for x in self.x]
        return y_apprx

    def yPrediction(self):
        y_apprx = [self.func(x) for x in self.xrange]
        return y_apprx

    def validation_R(self,x,y):
        y_apprx = [self.func(xval) for xval in x]
        residuals = [i-j for i,j in zip(y_apprx,y)]
        norm_residuals = np.sqrt(sum([i**2 for i in residuals]))
        print(norm_residuals)
        return norm_residuals

    def norm_residuals(self):
        residualVec = [(i-j) for i,j in zip(self.y,self.ypred_x)]
        norm = np.sqrt(sum([i**2 for i in residualVec]))
        return norm

    def summary(self):
        ss_total = sum([(i-np.mean(self.y))**2 for i in self.y])
        ss_res = sum([(i-j)**2 for i,j in zip(self.y,self.ypred_x)])
        r_squared = 1 - (ss_res/ss_total)
        print(f'''
        R^2 = {np.round(r_squared,5)}
        Regression equation = {self.strfunc}
        SSR = {np.round(ss_res,5)}
        Norm of Residuals (SSR^0.5) = {np.round(self.norm_residuals(),5)}
        ''')

    def plot(self):
        self.summary()
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        plt.title(self.latexFunc(self.coeff),fontweight='bold')
        plt.plot(self.x,self.y,'o',alpha=0.25,color='blue')
        plt.plot(self.xrange,self.ypred,color='red')
        plt.show()
            



if __name__ == '__main__':
    xset = np.linspace(0,1,100)
    yset = [i+np.random.random() for i in xset]
    model =  Model(xset,yset)
    model.plot()
    

