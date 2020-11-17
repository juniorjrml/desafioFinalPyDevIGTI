from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from matplotlib import pyplot as plt
import pandas as pd

from copy import copy
import joblib


# normalizando
def normalizando(entradas):
  normaliza = MinMaxScaler()  # objeto para a normalização
  entradas_normalizadas = normaliza.fit_transform(entradas)
  return entradas_normalizadas

def testando(modelo, X, y):
  predicao = modelo.predict(X)
  avaliacao(y,predicao)
  return modelo.score(X,y)

# Avaliando o modelo
def avaliacao(y_test, y_pred):
  matriz_confusao = confusion_matrix(y_test, y_pred)
  fig, ax = plot_confusion_matrix(conf_mat=matriz_confusao)
  plt.show()

# separando o dataset em treino e teste e normalizando
def trata_dataset(X, y):
  # Normalizando \/
  entradas_normalizadas = normalizando(X)

  # Dividindo o dataset para treinar e testar
  X_train, x_test, y_train, y_test = train_test_split(entradas_normalizadas, y, test_size=0.30,random_state=42)

  return X_train, x_test, y_train, y_test

# Seleciona o modelo se o score do modelo for melhor
def modelo_melhor_score(modelo, X, y, melhor_modelo):
  if melhor_modelo == None or modelo.score(X, y) > melhor_modelo.score(X, y):
    return modelo
  else:
    return melhor_modelo

# treina diversos modelos de uma mesma classe para salvar o melhor resultado
def treina_melhor_modelo(modelo_cls, X, y, ciclo = 10):
  melhor_modelo = None
  for i in range(ciclo):
    modelo = copy(modelo_cls)
    X_train, x_test, y_train, y_test = trata_dataset(X, y)
    modelo.fit(X_train, y_train)
    melhor_modelo = modelo_melhor_score(modelo, X, y, melhor_modelo)
  return melhor_modelo

# Escolhe o modelo em que o score e maior em uma lista de modelos
def escolhe_modelo_maior_score(modelos, X, y):
  melhor_modelo = None
  for modelo in modelos:
    melhor_modelo = modelo_melhor_score(modelo, X, y, melhor_modelo)
  return melhor_modelo

dataset = pd.read_csv("pima-indians-diabetes.csv", header=None)
X = dataset.drop([8], axis=1) # features
y = dataset[8] # Alvo

clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,10), random_state=1)
clf_mlp = treina_melhor_modelo(clf_mlp, X, y, ciclo=100)

clf_arvore = DecisionTreeClassifier(random_state=1)
clf_arvore = treina_melhor_modelo(clf_arvore, X, y, ciclo=100)

clf_KNN = KNeighborsClassifier(n_neighbors=5)
clf_KNN = treina_melhor_modelo(clf_KNN, X, y, ciclo=100)

nome_modelos = ["Rede MLP Classificador", "KNN Classificador", "Arvore Classificador"]
modelos = [clf_mlp, clf_KNN, clf_arvore]


X_train, x_test, y_train, y_test = trata_dataset(X, y)
melhor_modelo = escolhe_modelo_maior_score(modelos, x_test, y_test)
print(melhor_modelo.score(X, y))
arquivo_saida = "melhor_modelo.sav"
joblib.dump(melhor_modelo, arquivo_saida)

print(melhor_modelo.score(x_test,y_test))
print(melhor_modelo.__str__())
