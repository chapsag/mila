# -*- coding: utf-8 -*-

#####
# Pierre-Emmanuel Goffi 18110928
###

import numpy as np
import matplotlib.pyplot as plt


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, polynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None

        
    def fctNoyau(self, x, x_prime):
        if self.noyau == 'rbf':
            return np.exp(-np.linalg.norm(x - x_prime, ord=2, axis = 1) / (2 * self.sigma_square))
        elif self.noyau == 'lineaire':
            return np.dot(x, np.transpose(x_prime))
        elif self.noyau == 'polynomial':
            inner = np.dot(x, np.transpose(x_prime)) + self.c
            return np.sign(inner) * (np.abs(inner)) ** self.M
        elif self.noyau == 'sigmoidal':
            return np.tanh(self.b * np.dot(x, np.transpose(x_prime)) + self.d)
        print ('Fonction inconnue')
        return np.zeros(len(x))
    
    def assignParam(self, val, val2):
        if self.noyau == 'rbf':
            self.sigma_square = val
        elif self.noyau == 'polynomial':
            self.M = val
            self.c = val2
        elif self.noyau == 'sigmoidal':
            self.b = val
            self.d = val2
    
    def getParam(self):
        if self.noyau == 'rbf':
            return self.sigma_square, 0
        elif self.noyau == 'polynomial':
            return self.M, self.c
        elif self.noyau == 'sigmoidal':
            return self.b, self.d
        return 0, 0

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        K = self.fctNoyau(x_train, x_train)
        self.a = np.dot(np.linalg.inv(K + self.lamb * np.identity(len(K))), np.transpose(t_train))
        self.x_train = x_train
        
    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        y =  np.dot(np.transpose(self.a), self.fctNoyau(self.x_train, x))
        return 1 if y > 0.5 else 0 

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
       
        return np.sum((t - prediction)**2)

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k>=1 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        M = np.arange(2, 6, 0.5)
        B = np.arange(0.00001, 0.01, 0.001)
        D = B
        C = np.arange(0, 5, 0.1)
        lamb = np.arange(0.000000001, 2, 0.1)
        sigma = lamb
        
        param = range(1)
        param2 = range(1)
        if self.noyau == 'polynomial':
            param = M
            param2 = C
        elif self.noyau == 'rbf':
            param = sigma
        elif self.noyau == 'sigmoidal':
            param = B
            param2 = D
        minErreur = float('inf')
        bestLamb = 0
        bestParam = 0
        bestParam2 = 0
        for i in param:
            for j in param2:
                for l in lamb:
                   for k in range(10):
                       self.lamb = l
                       self.assignParam(i, j)
                    
                       nb = int(0.8 * len(t_tab))
                       indexes = np.array(range(0, len(t_tab)))
                       np.random.shuffle(indexes)
                       indexTrain = indexes[0 : nb]
                       indexTest = indexes[nb + 1 :]
                    
                       x_train = x_tab[indexTrain]
                       t_train = t_tab[indexTrain]
                       x_valid = x_tab[indexTest]
                       t_valid = t_tab[indexTest]
                    
                       self.entrainement(x_train, t_train)
                       predictions = np.array([self.prediction(x) for x in x_valid])
                       erreur = self.erreur(t_valid, predictions)
                       if erreur < minErreur:
                           bestParam, bestParam2 = self.getParam()
                           bestLamb = self.lamb
                           minErreur = erreur
        self.lamb = bestLamb
        self.assignParam(bestParam, bestParam2)
        self.entrainement(x_tab, t_tab)          
        

    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()
