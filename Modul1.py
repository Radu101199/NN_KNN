#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:15:28 2023

@author: paunradu
"""

import cv2
from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as la
import time
from statistics import mode
import pandas as pd
import os

caleBD = '/Users/paunradu/Desktop/att_faces'
nrPers = 40
nrPozeAntr = 6
nrPozeTest = 4
nrPixeli = 112*92
nrPozePers = 10

def configurareA(caleBD):
    A = np.zeros((nrPixeli, nrPozeAntr*nrPers))
    for i in range(1, nrPers+1):
        caleFolderPers = caleBD + '/s' + str(i) + '/'
        for j in range(1, nrPozeAntr + 1):
            calePoza = caleFolderPers + str(j) + '.pgm'
            poza = cv2.imread(calePoza, 0)
            poza = np.array(poza)
            pozaVect = np.reshape(poza, (nrPixeli, ))
            A[:, (i-1)*nrPozeAntr +j-1] = pozaVect
    return A

def NN(A, poza_test, norma):
    z = np.zeros(len(A[0]))
    for i in range(0, len(A[0])):
        if(norma=='cos'):
            z[i] = 1-np.dot(A[:,i],poza_test)/np.linalg.norm(A[:,i])*np.linalg.norm(poza_test)
        elif(norma=='inf'):
            z[i] = np.linalg.norm(poza_test-A[:, i], np.inf)
        elif(norma == '2'):
            z[i] = np.linalg.norm(poza_test - A[:, i])
        elif(norma == '1'):
            z[i] = np.linalg.norm(poza_test - A[:, i], 1)
    i0 = np.argmin(z)
    return i0

def KNN(A, poza_test, norma, k):
    z = np.zeros(len(A[0])) #tablou unidimensional cu numarul de coloane
    for i in range(len(A[0])):
        if(norma=='cos'):
            z[i] = 1-np.dot(A[:,i],poza_test)/(la.norm(A[:,i])*la.norm(poza_test))
        elif(norma=='inf'):
            z[i] = la.norm(poza_test-A[:, i], np.inf)
        elif(norma == '2'):
            z[i] = la.norm(poza_test - A[:, i])
        elif(norma == '1'):
            z[i] = la.norm(poza_test - A[:, i], 1)
    index = np.argsort(z)
    index_k = index[:k]
    classes_k = (index_k // nrPozeAntr) + 1 #clasele de provenienta
    
    if k == 1:
        return index[0] // nrPozeAntr + 1
    else:
        return mode(classes_k) #clasa majoritara
        
def statistici(A):
    valK = [1, 3, 5, 7, 9]
    norme = ['1', '2', 'inf', 'cos']
    matriceStatisticaRR = np.zeros((len(valK), len(norme)))
    matriceStatisticaTMI = np.zeros((len(valK), len(norme)))
    for k in range(len(valK)):
        for n in range(len(norme)):
            t=0
            nrRecunoasteriCorecte = 0
            for i in range(1, nrPers + 1):
                caleFolderPers = caleBD + '/s' + str(i) + '/'
                for j in range(nrPozeAntr + 1, nrPozePers + 1):
                    calePoza = caleFolderPers + str(j) + '.pgm'
                    poza = cv2.imread(calePoza, 0)
                    poza = np.array(poza)
                    pozaVect = np.reshape(poza, (nrPixeli, ))
                    t0 = time.perf_counter()
                    persoanaPrezisa = KNN(A, pozaVect, norme[n], valK[k])
                    t1 = time.perf_counter()
                    t = t+(t1-t0)
                    if persoanaPrezisa == i:
                        nrRecunoasteriCorecte = nrRecunoasteriCorecte + 1
                        
            rataRec = nrRecunoasteriCorecte/(nrPozeTest*nrPers)
            #print(f'Rada de recunoastere: {rataRec:.8f}')
            matriceStatisticaRR[k][n] = rataRec
            tmi=t/(nrPozeTest*nrPers)
            #print(f'Timp mediu de interogare: {tmi:.8f}')
            matriceStatisticaTMI[k][n] = tmi 
    RR = pd.DataFrame(matriceStatisticaRR, columns=norme, index=valK)
    TMI = pd.DataFrame(matriceStatisticaTMI, columns=norme, index=valK)
    csv_RR = 'ORL_8_kNN_RR.csv'
    csv_TMI = 'ORL_8_kNN_TMI.csv'
    RR.to_csv(csv_RR)
    TMI.to_csv(csv_TMI)
    
def plotStatistici(csv_statistici):
    df = pd.read_csv(csv_statistici, index_col=0)
    numefisier = os.path.splitext(os.path.basename(csv_statistici))[0]
    ultimul_cuvant = numefisier.split('_')[-1]
    
    for valoare_linie in df.index:
        linie_data = df.loc[valoare_linie].values
        
        plt.figure(figsize=(10, 6))
        
        plt.xticks(range(len(df.columns)), df.columns)
        plt.yticks([0], [valoare_linie])
        
        plt.scatter(range(len(df.columns)), linie_data, c='red', marker='o')
        for j in range(1, len(linie_data)):
           plt.plot([j - 1, j], [linie_data[j - 1], linie_data[j]], 'k-')

        plt.title(f"K={valoare_linie} ({ultimul_cuvant})")
        plt.xlabel("Norme")
        plt.show()
        
    
    
Matr_antrenare = configurareA(caleBD)
statistici(Matr_antrenare)
plotStatistici('ORL_8_kNN_RR.csv')
plotStatistici('ORL_8_kNN_TMI.csv')