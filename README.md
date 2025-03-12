# Reconnaissance de chiffres manuscrits, methode LeCun

## Sommaire

- [Introduction](#introduction)
- [Importation des données](#importation-des-données)
- [Transformation des données](#transformation-des-données)
- [Modèle](#modèle)
- [Entraînement du modèle](#entraînement-du-modèle)
- [Vérification](#vérification)
- [Vérification avec le test set](#vérification-avec-le-test-set)
- [Conclusion](#conclusion)


## Introduction

Dans ce projet, l'objectif est de développer un modèle de reconnaissance de chiffres manuscrits, en suivant les instructions et l'architecture décrites dans le papier de recherche "Handwritten Digit Recognition with a Back-Propagation Network" de Yann LeCun et al sorti en 1989. L'idée est de reconstruire le modèle proposé dans ce papier en utilisant PyTorch et des techniques de deep learning et en s'inspirant uniquement de la description du modèle.

Ce projet permet d’explorer les concepts fondamentaux du deep learning, tels que la manipulation des tenseurs, la construction de réseaux de neurones et l’optimisation des hyperparamètres pour améliorer les prédictions.


## Importation des données 

Le jeu de données a été importé depuis : www.di.ens.fr/~lelarge/MNIST.tar.gz

Il est composé de deux parties : un training set et un test set. Nous allons d'abord importer le training set sans transformer les données afin de voir à quoi ressemble le dataset, puis nous allons créer un dataset personnalisé pour le prétraitement.

Le jeu de données est composé de 60 000 images de 28 pixels par 28 pixels, accompagné de 60 000 labels correspondants.

![Capture d’écran 2025-02-10 190018](https://github.com/user-attachments/assets/1bf786fd-a784-4387-a316-b6b4fb5d9cc8)

## Transformation des données 

Dans l'article de recherche, les images sont initialement de taille 16×16, puis redimensionnées en 28×28 afin d'éviter des erreurs ou des problèmes liés aux bordures. Cependant, les images du dataset MNIST sont déjà de taille 28×28. Nous choisissons donc de ne pas les transformer.

Par ailleurs, il est nécessaire de normaliser les images. La normalisation consiste à ramener les valeurs des pixels dans un intervalle restreint, ici entre -1 et 1. Cette étape est essentielle, car des valeurs trop élevées peuvent rendre l’entraînement du modèle instable. Sans normalisation, les mises à jour des poids des neurones seraient trop irrégulières, les grandes valeurs ayant un impact disproportionné sur l'apprentissage par rapport aux plus petites.

Il faut également effectuer un One-Hot Encoding pour la variable à expliquer **y**. Le One-Hot Encoding est une méthode de représentation des labels sous forme de vecteurs binaires.
Nous effectuons cela car le One-Hot Encoding permet au modèle de calculer la perte plus précisément en comparant directement les probabilités prédites avec les labels.


## Modèle 

L'architecture du modèle, qui est un CNN, suit celle décrite dans le papier de recherche. Il est composé de deux couches de convolution et de deux couches de pooling, utilisant un average pooling. La fonction d'activation n'est pas explicitement précisée dans l'article, mais nous utilisons Tanh plutôt que ReLU, car cette dernière n'était pas encore couramment utilisée dans les années 1990.


## Entraînement du modèle 

### Fonction coût

La fonction coût utilisée pour notre modèle est la MSE (Mean Squared Error). Ce n'est pas la fonction la plus optimale car elle pénalise trop les petites erreurs. La Cross Entropy serait plus adaptée mais c'est la MSE qui est utilisé dans le papier de LeCun donc nous l'utilisons.

### Optimiseur 

L'optimiseur n'étant pas précisé dans l'article de recherche, nous utilisons la Descente de Gradient Stochastique. En effet, des optimiseurs plus avancés comme Adam n'étaient pas encore disponibles à l'époque.

### Entraînement

![Capture d’écran 2025-03-12 145611](https://github.com/user-attachments/assets/b066ae77-90d3-406a-93c0-54054b44558f)

Nous constatons qu'au fur et à mesure de l'entraînement, les erreurs du modèle diminuent. Au début, l'erreur était de 0.03 pour atteindre 0.01 à la fin.

## Vérification 

Voici le résultat obtenu pour les 20 premières images du train set :

![Capture d’écran 2025-02-10 191213](https://github.com/user-attachments/assets/ae58b198-1cf2-4bbe-ae5d-3c3e48a8f07b)


## Vérification avec le test set

Il faut maintenant vérifier si le modèle souffre de surapprentissage ou de sous-apprentissage des données. C'est là qu'intervient le test set, car le modèle n'a jamais vu ces images.

Voici le résultat obtenu pour les 20 premières images du test set :

![Capture d’écran 2025-02-10 191435](https://github.com/user-attachments/assets/fc61e9de-8098-44ca-9e60-7d11bd73d046)

Le modèle est très bon car il n'y pas de surapprentissage.

## Conclusion 

En conclusion, notre modèle montre de très bonnes performances dans la reconnaissance des chiffres manuscrits. En effet, il est capable de prédire avec précision, sans surapprentissage, comme l'indique l'évaluation sur le test set. Les erreurs du modèle diminuent significativement au fil des epochs, ce qui témoigne de l'efficacité de l'apprentissage. Ainsi, nous pouvons affirmer que le modèle est bien entraîné et qu'il généralise correctement sur de nouvelles données.

Notre objectif, qui était de reproduire l’architecture décrite dans le papier de recherche de Yann LeCun sur la reconnaissance des chiffres manuscrits, confirme la pertinence de cette approche lorsqu'on examine les résultats obtenus et démontre l’efficacité de l’apprentissage par rétropropagation dans un réseau de neurones convolutionnel.
