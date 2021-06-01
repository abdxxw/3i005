"""
@author: 3804891
"""

import math
import utils as ut
import numpy as np
import pandas as pd 
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt


##################################################################################
###########################	   Question1	  #################################
##################################################################################


def getPrior(df):
    """
    :param df: Dataframe des données. Doit avoir une colonne "target" (0 ou 1).
    :type df: pandas dataframe
    :return: Dictionnaire contennant la moyenne et les extremités de l'intervalle
             de confiance. Clés 'estimation', 'min5pourcent', 'max5pourcent'.
    :rtype: Dictionnaire
    """
	out =  dict()
	m = df["target"].mean()
	temp = 1.96 * math.sqrt((m * (1 - m))/df.shape[0])
	out['estimation'] = m
	out['min5pourcent'] = m - temp
	out['max5pourcent'] = m + temp
	
	return out

	
##################################################################################
###########################	   Question2	  #################################
##################################################################################
	
	
class APrioriClassifier(ut.AbstractClassifier):

	def __init__(self):
		pass

	def estimClass(self, att):
	 """
        estimer la classe
        Pour ce Classifier, la classe vaut toujours 1.
        
        :param att: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
		return 1

	def statsOnDF(self, df):
	    """
        à partir d'un dataframe, calcule les stats pour la classification.
       
        :param df:  le dataframe 
        :return: un dictionnaire des VP,FP,VN,FN,précision et rappel
        
        VP : vrai positif, VN : vrai negatif, FP : faux positif, FN : faux negatif.
        """
		out = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0 , 'Precision': 0 , 'Rappel': 0 }
		
		for i in range(df.shape[0]):
			dic = ut.getNthDict(df, i)
			predict = self.estimClass(dic)
			if dic['target'] == 1 and predict == 1:
				out['VP'] += 1
			elif dic['target'] == 0 and predict == 0:
				out['VN'] += 1
			elif dic['target'] == 0 and predict == 1:
				out['FP'] += 1
			else:
				out['FN'] += 1

		out['Precision'] = out['VP']*1.0/(out['VP']+out['FP'])
		out['Rappel'] = out['VP']*1.0/(out['VP']+out['FN'])
		
		return out
		
	
##################################################################################
###########################	   Question3	  #################################
##################################################################################		
		
def getUnique(listofElements):
	"""
    Retourne valeurs unique d'une liste.
    """
	uniqueList = []
	
	for elem in listofElements:
		if elem not in uniqueList:
			uniqueList.append(elem)
		 
	return uniqueList
	
	
def P2D_l(df, attr):
    """
    Calcul de la probabilité conditionnelle P(attribut | target).
    
    :param df: dataframe avec les données.
    :param attr: attribut à utiliser.
    :return: dictionnaire de dictionnaire des probabilté conditionnelle.
    :rtype: Dictionnaire de Dictionnaire.
    """
	out = dict()
	keys = getUnique(df[attr].values)
	for t in df['target'].unique():
		out[t] = dict.fromkeys(keys, 0)
		
	tuple = df.groupby(["target", attr]).groups
	for t, v in tuple:
		out[t][v] = len(tuple[(t, v)])
	
	nb = df.groupby('target')[attr].count()
  
	for t in out.keys():
		for v in out[t].keys():
			out[t][v] /= nb[t]
		
	return out	
	
def P2D_p(df, attr):
    """
    Calcul de la probabilité conditionnelle P(target | attribut).
    
    :param df: dataframe avec les données.
    :param attr: attribut à utiliser.
    :return: dictionnaire de dictionnaire des probabilté conditionnelle.
    :rtype: Dictionnaire de Dictionnaire.
    """

	out = dict()
	keys = getUnique(df[attr].values)
	for v in df[attr].unique():
		out[v] = dict.fromkeys(keys, 0)
		
	tuple = df.groupby([attr, "target"]).groups
	for v, t in tuple:
		out[v][t] = len(tuple[(v, t)])
	
	nb = df.groupby(attr)['target'].count()
  
	for v in out.keys():
		for t in out[v].keys():
			out[v][t] /= nb[v]
		
	return out
	
	
	
class ML2DClassifier(APrioriClassifier):

	def __init__(self, df, attr):    
    """
        Initialise le classifieur. Crée un dictionnaire avec les probabilités conditionnelles P(attribut | target).
        
        :param df: dataframe.
        :param attr: attribut a utiliser.
        """
		self.df = df
		self.attr = attr
		self.proba = P2D_l(self.df,self.attr)

	def estimClass(self, attrs):
	        """
        estimer la classe
        Pour ce Classifier, la classe est estimeé par le maximum de vraisemblance.   
        
        :param attrs: le  dictionnaire.
        :return: la classe 0 ou 1 estimée
        """
		if(self.proba[0][attrs[self.attr]] >= self.proba[1][attrs[self.attr]]):
			return 0
		else:
			return 1
			
			
class MAP2DClassifier(APrioriClassifier):

	def __init__(self, df, attr):
	    """
        Initialise le classifieur. Crée un dictionnaire avec les probabilités conditionnelles P(targer | attribut).
        
        :param df: dataframe.
        :param attr: attribut a utiliser.
        """
		self.df = df
		self.attr = attr
		self.proba = P2D_p(df,self.attr)

	def estimClass(self, attrs):
		        """
        estimer la classe
        Pour ce Classifier, la classe est estimeé par le maximum de posteriori.   
        
        :param attrs: le  dictionnaire.
        :return: la classe 0 ou 1 estimée
        """
		if(self.proba[attrs[self.attr]][0] >= self.proba[attrs[self.attr]][1]):
			return 0
		else:
			return 1
			
			
			
	
##################################################################################
###########################	   Question4	  #################################
##################################################################################	


def nbParams(df, liste = None):
    """
    Affiche la taille mémoire de tables P(target|attr1,..,attrk) en supposant qu'un float
    est représenté sur 8 octets..
	
	:param df: Dataframe. 
    :param liste: liste des colonnes prises en considération pour le calcul.
	"""
	if liste is None:
		liste = list(df.columns)
	taille = 8
	for c in liste:
		taille *= np.unique(df[c].values).size
	out = str(len(liste)) + " variable(s) : " + str(taille) + " octets"
	if taille >= 1024:
		out = out + " = " + convert(taille)
	print (out) 
	
	
	
def convert(taille):
    """
    Transforme un entier a un String (representation en nombre d’octets, ko, mo, go et to). 
    :param taille: le nombre à être transformé.
    """
	units = ["o", "Ko", "Mo", "Go", "To"]
	out = ""
	for u in units:
		if taille == 0:
			break  
		if u == "To":
			out = " {:d}".format(taille) + u + out
		else:
			out = " {:d}".format(taille % 1024) + u + out
			taille //= 1024
	
	if out == "":
		out = " 0o"
	return out
	
	
def nbParamsIndep(df):
    """
    Affiche la taille mémoire pour représenter les tables de probabilité, en supposant l'indépendance des variables et qu'un
    float est représenté sur 8 octets.
    
    :param df: Dataframe.  
    """
	taille = 0
	liste = list(df.columns) 
	
	for c in liste:
		taille += (np.unique(df[c].values).size * 8)
	
	out = str(len(liste)) + " variable(s) : " + str(taille) + " octets"
	if taille >= 1024:
		out = out + " = " + convert(taille)
	print (out)	


##################################################################################
###########################	   Question5	  #################################
##################################################################################	
	
	
def drawNaiveBayes(df, col):
    """
    dessiner un graphe orienté.
    
    :param df: Dataframe.  
    :param col: le nom de la racine.
    """
	liste = list(df.columns.values)
	liste.remove(col)
	out = ""
	for e in liste:
		out = out + col + "->" + e + ";"
	return ut.drawGraph(out)  
	
	
	
	
def nbParamsNaiveBayes(df, p, liste = None):
    """
    Affiche la taille mémoire de tables P(target), P(attr1|target),.., P(attrk|target),
    en supposant qu'un float est représenté sur 8 octets.
    
    :param df: Dataframe. 
    :param p: le nom de la racine.
    :param liste: liste des colonnes prises en considération. 
    """
	taille = np.unique(df[p].values).size * 8
	
	if liste is None:
		liste = list(df.columns) 
		
	if liste != []:  
		liste.remove(p)
	
	for c in liste:
		taille += (np.unique(df[p].values).size * np.unique(df[c].values).size) * 8
	
	out = str(len(liste)) + " variable(s) : " + str(taille) + " octets"
	if taille >= 1024:
		out = out + " = " + convert(taille)
	print (out)
	
	
	
class MLNaiveBayesClassifier(APrioriClassifier):

	def __init__(self, df):
        """
        Initialise le classifieur. Crée un dictionnarie qui associe à chaque attribut un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target).
        
        :param df: dataframe.
        """
		self.df = df
		self.proba = dict()
		liste = list(df.columns.values)
		liste.remove("target")
		for attr in liste:
			self.proba[attr] = P2D_l(df, attr)
	
	def estimClass(self, attrs):
        """
        estimer la classe
        Pour ce Classifier, la classe est estimeé par le maximum de vraisemblance.   
        
        :param attrs: le  dictionnaire.
        :return: la classe 0 ou 1 estimée
        """
		res = self.estimProbas(attrs)
		if res[0] >= res[1]:
			return 0
		return 1
		
	def estimProbas(self, attrs):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).
        
        :param attrs: le dictionnaire nom-valeur des attributs
        """  
		out1 = 1
		out2 = 1
		for p in self.proba:
			temp = self.proba[p]
			if attrs[p] in temp[0]:
				out1 *= temp[0][attrs[p]]
				out2 *= temp[1][attrs[p]]
			else:
				return {0: 0.0, 1: 0.0}
		return {0: out1, 1: out2}
	
class MAPNaiveBayesClassifier(APrioriClassifier):

	def __init__(self, df):
        """
        Initialise le classifieur. Crée un dictionnarie qui associe à chaque attribut un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target).
        
        :param df: dataframe.
        """
		self.df=df
		self.moy = self.df["target"].mean()
		self.proba = dict()
		liste = list(df.columns.values)
		liste.remove("target")
		for attr in liste:
			self.proba[attr] = P2D_l(df, attr)
	
	def estimClass(self, attrs):
        """
        estimer la classe
        Pour ce Classifier, la classe est estimeé par le maximum à posteriori.   
        
        :param attrs: le  dictionnaire.
        :return: la classe 0 ou 1 estimée
        """
		res = self.estimProbas(attrs)
		if res[0] >= res[1]:
			return 0
		return 1
		

	def estimProbas(self, attrs):
         """
        Calcule la probabilité à posteriori par naïve Bayes : P(attr1, ..., attrk | target).
        
        :param attrs: le dictionnaire nom-valeur des attributs
        """  
		out2 = self.moy
		out1 = 1-self.moy
		
		for p in self.proba:
			temp = self.proba[p]
			if attrs[p] in temp[0]:
				out1 *= temp[0][attrs[p]]
				out2 *= temp[1][attrs[p]]
			else:
				return {0: 0.0, 1: 0.0}

		return {0: (out1 / (out1 + out2)), 1: (out2 / (out1 + out2))}	

	


##################################################################################
###########################	   Question6	  #################################
##################################################################################	

	
def isIndepFromTarget(df,attr,x):
    """
    Vérifie si attr est indépendant de target au seuil de x%.
    
    :param df: dataframe.
    :param attr: le nom d'une colonne du dataframe df.
    :param x: seuil de confiance.
    """
	mat = pd.crosstab(df[attr],df.target).values
	return chi2_contingency(mat)[1] > x
	
	
	
	
class ReducedMLNaiveBayesClassifier(APrioriClassifier):
	def __init__(self, df, seuil):
	     """
        Initialise le classifieur. Crée un dictionnarie qui associe à chaque attribut un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target) où attribut et target
        ne sont pas indépendants.
        
        :param df: dataframe.
        :param seuil: seuil de confiance.
        """
		self.df = df
		self.proba = dict()
		liste = list(df.columns.values)
		liste.remove("target")
		for attr in liste:
			if not isIndepFromTarget(df, attr, seuil):
				self.proba[attr] = P2D_l(df, attr)
			

	def estimClass(self, attrs):
	    """
        estimer la classe
        Pour ce Classifier, la classe est estimeé par le maximum de vraisemblance.   
        
        :param attrs: le  dictionnaire.
        :return: la classe 0 ou 1 estimée
        """
		res = self.estimProbas(attrs)
		if res[0] >= res[1]:
			return 0
		return 1


	def estimProbas(self,attrs):
	    """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).
        
        :param attrs: le dictionnaire nom-valeur des attributs
        """  
		out1 = 1
		out2 = 1
		for p in self.proba:
			temp = self.proba[p]
			if attrs[p] in temp[0]:
				out1 *= temp[0][attrs[p]]
				out2 *= temp[1][attrs[p]]
			else:
				return {0: 0.0, 1: 0.0}
		return {0: out1, 1: out2}

	def draw(self):
		out = ""
		for i in self.proba.keys():
			out += "target"+"->"+i+";"
		return ut.drawGraph(out)



class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
	def __init__(self, df, seuil):
	     """
        Initialise le classifieur. Crée un dictionnarie qui associe à chaque attribut un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target) où attribut et target
        ne sont pas indépendants.
        
        :param df: dataframe.
        :param seuil: seuil de confiance.
        """
		self.df=df
		self.moy = self.df["target"].mean()
		self.proba = dict()
		liste = list(df.columns.values)
		liste.remove("target")
		for attr in liste:
			if not isIndepFromTarget(df, attr, seuil):
				self.proba[attr] = P2D_l(df, attr)



	def estimClass(self, attrs):
	     """
        estimer la classe
        Pour ce Classifier, la classe est estimeé par le maximum à posteriori.   
        
        :param attrs: le  dictionnaire.
        :return: la classe 0 ou 1 estimée
        """
		res = self.estimProbas(attrs)
		if res[0] >= res[1]:
			return 0
		return 1


	def estimProbas(self, attrs):
          """
        Calcule la probabilité à posteriori par naïve Bayes : P(attr1, ..., attrk | target).
        
        :param attrs: le dictionnaire nom-valeur des attributs
        """  
		out2 = self.moy
		out1 = 1-self.moy
		
		for p in self.proba:
			temp = self.proba[p]
			if attrs[p] in temp[0]:
				out1 *= temp[0][attrs[p]]
				out2 *= temp[1][attrs[p]]
			else:
				return {0: 0.0, 1: 0.0}

		return {0: (out1 / (out1 + out2)), 1: (out2 / (out1 + out2))} 

	def draw(self):
		out = ""
		for i in self.proba.keys():
			out += "target"+"->"+i+";"
		return ut.drawGraph(out)

		

##################################################################################
############################	   Question7	  ################################
##################################################################################			
		
def mapClassifiers(dic, df):
    """
    Représente graphiquement les classifiers à partir d'un dictionnaire dic de 
    {nom:instance de classifier}. 
    
    :param dic: dictionnaire {nom:instance de classifier}
    :param df: dataframe..
    """
	precision = []
	rappel = []
	name = []
	
	for i, n in dic.items():
         dico_stats = n.statsOnDF(df)
         name.append(i)
         precision.append(dico_stats["Precision"])
         rappel.append(dico_stats["Rappel"])
    
	plt.scatter(precision, rappel, color="r", marker="x")
	for i, txt in enumerate(name):
		plt.annotate(txt, (precision[i], rappel[i]))
	
	plt.show()
	
			
