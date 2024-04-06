import math
from collections import defaultdict
import matplotlib.pyplot as plt

# Definindo a classe para representar os atributos e a classe de cada exemplo
class Targets:
    def __init__(self, a, b, c, d, classe):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.classe = classe

    # Métodos para acessar os atributos e a classe de cada exemplo
    def get_classe(self):
        return self.classe

    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_c(self):
        return self.c

    def get_d(self):
        return self.d
    
# Função para calcular a distância euclidiana entre dois exemplos
def distancia_euclidiana(a, b):
    return math.sqrt((a.get_a() - b.get_a()) ** 2 +
                     (a.get_b() - b.get_b()) ** 2 +
                     (a.get_c() - b.get_c()) ** 2 +
                     (a.get_d() - b.get_d()) ** 2)

# Função para classificar um novo exemplo com base nos K vizinhos mais próximos
def classificar_amostra(targetss, novo_exemplo, K):
    # Dicionário para contar as classes dos K vizinhos mais próximos
    cont_classes = defaultdict(int)

    # Calcula a distância euclidiana do novo exemplo para cada exemplo do conjunto de treinamento
    dist_targetss = [(distancia_euclidiana(target, novo_exemplo), i) for i, target in enumerate(targetss)]
    dist_targetss.sort()
    # Contabiliza as classes dos K vizinhos mais próximos
    for dist, idx in dist_targetss[:K]:
        classe = targetss[idx].get_classe()
        cont_classes[classe] += 1

    # Decide a qual classe pertence o novo exemplo baseado na contagem das classes dos vizinhos
    classe_classificacao = max(cont_classes, key=cont_classes.get)

    return classe_classificacao

# Dados de treinamento
data = """5.1 3.5 1.4 0.2 Iris-setosa
7.0 3.2 4.7 1.4 Iris-versicolor
6.3 3.3 6.0 2.5 Iris-virginica
4.9 3.0 1.4 0.2 Iris-setosa
5.7 2.8 4.1 1.3 Iris-versicolor
5.8 2.7 5.1 1.9 Iris-virginica
4.7 3.2 1.3 0.2 Iris-setosa
5.1 2.5 3.0 1.1 Iris-versicolor
7.1 3.0 5.9 2.1 Iris-virginica
4.6 3.1 1.5 0.2 Iris-setosa
5.8 2.6 4.0 1.2 Iris-versicolor
6.3 2.9 5.6 1.8 Iris-virginica
5.0 3.6 1.4 0.2 Iris-setosa
6.1 3.0 4.6 1.4 Iris-versicolor
6.5 3.0 5.8 2.2 Iris-virginica
5.4 3.9 1.7 0.4 Iris-setosa
5.5 2.6 4.4 1.2 Iris-versicolor
7.6 3.0 6.6 2.1 Iris-virginica
4.6 3.4 1.4 0.3 Iris-setosa
5.5 2.5 4.0 1.3 Iris-versicolor
4.9 2.5 4.5 1.7 Iris-virginica
5.0 3.4 1.5 0.2 Iris-setosa
5.6 3.0 4.1 1.3 Iris-versicolor
7.3 2.9 6.3 1.8 Iris-virginica
5.1 3.5 1.4 0.3 Iris-setosa
6.3 2.3 4.4 1.3 Iris-versicolor
6.7 2.5 5.8 1.8 Iris-virginica
5.7 3.8 1.7 0.3 Iris-setosa
6.7 3.1 4.7 1.5 Iris-versicolor
7.2 3.6 6.1 2.5 Iris-virginica
5.1 3.8 1.5 0.3 Iris-setosa
6.0 3.4 4.5 1.6 Iris-versicolor
6.5 3.2 5.1 2.0 Iris-virginica
5.4 3.4 1.7 0.2 Iris-setosa
5.4 3.0 4.5 1.5 Iris-versicolor
6.4 2.7 5.3 1.9 Iris-virginica
5.1 3.7 1.5 0.4 Iris-setosa
6.0 2.7 5.1 1.6 Iris-versicolor
6.8 3.0 5.5 2.1 Iris-virginica
4.6 3.6 1.0 0.2 Iris-setosa
5.8 2.7 3.9 1.2 Iris-versicolor
6.0 2.2 5.0 1.5 Iris-virginica
5.1 3.3 1.7 0.5 Iris-setosa
5.5 2.4 3.7 1.0 Iris-versicolor
6.9 3.2 5.7 2.3 Iris-virginica
4.8 3.4 1.9 0.2 Iris-setosa
5.5 2.4 3.8 1.1 Iris-versicolor
5.6 2.8 4.9 2.0 Iris-virginica
5.0 3.0 1.6 0.2 Iris-setosa
5.7 2.6 3.5 1.0 Iris-versicolor
7.7 2.8 6.7 2.0 Iris-virginica
5.0 3.4 1.6 0.4 Iris-setosa
6.0 2.9 4.5 1.5 Iris-versicolor
6.3 2.7 4.9 1.8 Iris-virginica
5.2 3.5 1.5 0.2 Iris-setosa
6.7 3.0 5.0 1.7 Iris-versicolor
6.7 3.3 5.7 2.1 Iris-virginica
5.2 3.4 1.4 0.2 Iris-setosa
5.6 3.0 4.5 1.5 Iris-versicolor
7.2 3.2 6.0 1.8 Iris-virginica
4.7 3.2 1.6 0.2 Iris-setosa
6.7 3.1 4.4 1.4 Iris-versicolor
6.2 2.8 4.8 1.8 Iris-virginica
4.8 3.1 1.6 0.2 Iris-setosa
5.6 2.9 3.6 1.3 Iris-versicolor
6.1 2.6 5.6 1.4 Iris-virginica
5.4 3.4 1.5 0.4 Iris-setosa
6.1 2.9 4.7 1.4 Iris-versicolor
7.7 3.0 6.1 2.3 Iris-virginica
4.4 3.0 1.3 0.2 Iris-setosa
6.0 2.2 4.0 1.0 Iris-versicolor
6.3 3.4 5.6 2.4 Iris-virginica
5.1 3.4 1.5 0.2 Iris-setosa
5.9 3.0 4.2 1.5 Iris-versicolor
6.4 3.1 5.5 1.8 Iris-virginica
5.0 3.5 1.3 0.3 Iris-setosa
5.0 2.0 3.5 1.0 Iris-versicolor
6.0 3.0 4.8 1.8 Iris-virginica
4.5 2.3 1.3 0.3 Iris-setosa
5.2 2.7 3.9 1.4 Iris-versicolor
6.9 3.1 5.4 2.1 Iris-virginica
4.4 3.2 1.3 0.2 Iris-setosa
6.6 2.9 4.6 1.3 Iris-versicolor
6.7 3.1 5.6 2.4 Iris-virginica
5.0 3.5 1.6 0.6 Iris-setosa
4.9 2.4 3.3 1.0 Iris-versicolor
6.8 3.2 5.9 2.3 Iris-virginica
5.1 3.8 1.9 0.4 Iris-setosa
6.3 3.3 4.7 1.6 Iris-versicolor
6.7 3.3 5.7 2.5 Iris-virginica
4.8 3.0 1.4 0.3 Iris-setosa
5.7 2.8 4.5 1.3 Iris-versicolor
6.7 3.0 5.2 2.3 Iris-virginica
5.1 3.8 1.6 0.2 Iris-setosa
6.5 2.8 4.6 1.5 Iris-versicolor
6.3 2.5 5.0 1.9 Iris-virginica
4.6 3.2 1.4 0.2 Iris-setosa
5.5 2.3 4.0 1.3 Iris-versicolor
6.5 3.0 5.2 2.0 Iris-virginica
5.3 3.7 1.5 0.2 Iris-setosa
6.9 3.1 4.9 1.5 Iris-versicolor
6.2 3.4 5.4 2.3 Iris-virginica
5.0 3.3 1.4 0.2 Iris-setosa
6.4 3.2 4.5 1.5 Iris-versicolor
5.9 3.0 5.1 1.8 Iris-virginica
5.2 4.1 1.5 0.1 Iris-setosa
6.1 2.8 4.0 1.3 Iris-versicolor
6.1 3.0 4.9 1.8 Iris-virginica
5.5 4.2 1.4 0.2 Iris-setosa
5.6 2.5 3.9 1.1 Iris-versicolor
7.2 3.0 5.8 1.6 Iris-virginica
4.9 3.1 1.5 0.1 Iris-setosa
6.2 2.2 4.5 1.5 Iris-versicolor
7.4 2.8 6.1 1.9 Iris-virginica
5.0 3.2 1.2 0.2 Iris-setosa
5.8 2.7 4.1 1.0 Iris-versicolor
7.9 3.8 6.4 2.0 Iris-virginica
5.5 3.5 1.3 0.2 Iris-setosa
6.2 2.9 4.3 1.3 Iris-versicolor
6.4 2.8 5.6 2.2 Iris-virginica
4.9 3.1 1.5 0.1 Iris-setosa
5.7 2.9 4.2 1.3 Iris-versicolor
6.3 2.8 5.1 1.5 Iris-virginica
4.4 2.9 1.4 0.2 Iris-setosa
5.7 3.0 4.2 1.2 Iris-versicolor
5.7 2.5 5.0 2.0 Iris-virginica
4.9 3.1 1.5 0.1 Iris-setosa
5.6 2.7 4.2 1.3 Iris-versicolor
5.8 2.8 5.1 2.4 Iris-virginica
5.4 3.7 1.5 0.2 Iris-setosa
5.7 2.8 4.5 1.3 Iris-versicolor
6.4 3.2 5.3 2.3 Iris-virginica
4.8 3.4 1.6 0.2 Iris-setosa
6.3 2.5 4.9 1.5 Iris-versicolor
6.5 3.0 5.8 2.2 Iris-virginica
4.8 3.0 1.4 0.1 Iris-setosa
5.6 2.8 4.9 2.0 Iris-versicolor
7.7 2.6 6.9 2.3 Iris-virginica
5.0 3.4 1.6 0.4 Iris-setosa
6.0 2.2 4.0 1.0 Iris-versicolor
6.9 3.2 5.7 2.3 Iris-virginica
5.8 4.0 1.2 0.2 Iris-setosa
6.1 3.0 4.6 1.4 Iris-versicolor
5.1 3.8 1.5 0.3 Iris-setosa
6.3 2.3 4.4 1.3 Iris-versicolor
7.2 3.6 6.1 2.5 Iris-virginica
6.0 3.0 4.8 1.8 Iris-versicolor
6.8 3.0 5.5 2.1 Iris-virginica"""
# Separando os dados em linhas e convertendo para objetos da classe Targets
data_lines = data.split('\n')
targetss = []

for line in data_lines:
    a, b, c, d, classe = line.split()
    targetss.append(Targets(float(a), float(b), float(c), float(d), classe))

K = 3

acertos = 0
tam_testes = 150 - len(targetss)

# Entrada de dados e classificação
entrada_dados = []
classificacoes_erradas = []
while True:
    try:
        entrada = input("Entre com os valores de a, b, c, d e a classe (separados por espaço), ou pressione Enter para encerrar: ")
        if entrada.strip() == "":
            break
        a, b, c, d, classe = entrada.split()
        novo_exemplo = Targets(float(a), float(b), float(c), float(d), classe)
        entrada_dados.append(novo_exemplo)
    except ValueError:
        print("Entrada inválida. Certifique-se de inserir valores numéricos para a, b, c, d.")
        continue

# Classifica os exemplos inseridos pelo usuário e verifica a acurácia
for novo_exemplo in entrada_dados:
    classe_obtida = classificar_amostra(targetss, novo_exemplo, K)
    print("Classe esperada:", novo_exemplo.get_classe())
    print("Classe obtida:", classe_obtida, "\n")
    if novo_exemplo.get_classe() == classe_obtida:
        acertos += 1
    else:
        classificacoes_erradas.append(novo_exemplo)

print(acertos, "acertos de um total de", len(entrada_dados), "testes.")
print("Acurácia: {:.2f}%".format(acertos / len(entrada_dados) * 100))


# Separando os atributos por classe para plotagem
setosa = [attr for attr in targetss if attr.get_classe() == 'Iris-setosa']
versicolor = [attr for attr in targetss if attr.get_classe() == 'Iris-versicolor']
virginica = [attr for attr in targetss if attr.get_classe() == 'Iris-virginica']

# Plotagem
fig, ax = plt.subplots()
ax.scatter([attr.get_a() for attr in setosa], [attr.get_b() for attr in setosa], color='r', label='Iris-setosa (treinamento)')
ax.scatter([attr.get_a() for attr in versicolor], [attr.get_b() for attr in versicolor], color='g', label='Iris-versicolor (treinamento)')
ax.scatter([attr.get_a() for attr in virginica], [attr.get_b() for attr in virginica], color='b', label='Iris-virginica (treinamento)')
ax.scatter([attr.get_a() for attr in entrada_dados], [attr.get_b() for attr in entrada_dados], color='black', label='Entradas do usuário')

# Destaca as classificações erradas
for attr in classificacoes_erradas:
    ax.scatter(attr.get_a(), attr.get_b(), color='k', marker='x', s=100)

plt.xlabel('a')
plt.ylabel('b')
plt.legend(loc='best')
plt.show()