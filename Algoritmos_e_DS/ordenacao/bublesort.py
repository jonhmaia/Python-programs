def bubble_sort(arr):
    n = len(arr)
    # Percorre todos os elementos do vetor
    for i in range(n-1):
        # Percorre os elementos do vetor de 0 até n-i-1
        for j in range(0, n-i-1):
            # Troca se o elemento encontrado for maior que o próximo
            if arr[j] > arr[j+1]:
                # Troca os elementos
                arr[j], arr[j+1] = arr[j+1], arr[j]
# Entrada de dados o split serve para separar os elementos do vetor
arr = input("Entre com o vetor: ").split()
# Converte os elementos do vetor para inteiros
arr = [int(num) for num in arr]
# Chama a função bubble_sort
bubble_sort(arr)
# Imprime o vetor organizado
print("Vetor organizado:", arr)
