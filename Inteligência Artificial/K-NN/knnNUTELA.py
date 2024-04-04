import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo as fucirepo
from sklearn.datasets import load_iris

# Carregar conjunto de dados Iris
iris = load_iris()
# Converter os dados em um DataFrame do Pandas para facilitar a manipulação
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
 #Adicionar os rótulos (classes) ao DataFrame
df['target'] = iris.target
# Plotar o gráfico de dispersão
sns.pairplot(df, hue='target', markers=["o", "s", "D"])
plt.show()
# Carregar conjunto de dados Iris
iris = load_iris()
# Obter características (features) e rótulos (targets)
X = iris.data
y = iris.target
# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Criar e treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train.ravel())  # Usar ravel() para garantir que os rótulos tenham a forma correta
# Avaliar o modelo
accuracy = knn.score(X_test, y_test)
print("taxa de acerto:", accuracy)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# Fazer previsões
y_pred = knn.predict(X_test)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
# Fazer previsões
y_pred = knn.predict(X_test)
# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(cm)
# Relatório de Classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
# Plotar a matriz de confusão
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Matriz de confusão')
plt.show()
