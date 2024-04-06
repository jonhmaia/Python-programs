def selection_sort(arr):
    # Tamanho do vetor
    n = len(arr)
    # Percorre todos os elementos do vetor
    for i in range(n):
        # Encontra o menor elemento no vetor
        min_idx = i
        # Percorre os elementos do vetor de i+1 até n
        for j in range(i+1, n):
            # Troca se o elemento encontrado for menor que o próximo
            if arr[j] < arr[min_idx]:
                # Atualiza o índice do menor elemento
                min_idx = j
                # Troca os elementos
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

vetor = input("Entre com o vetor: ")
numbers = list(map(int, vetor.split()))


vetor_ordenado = selection_sort(numbers)

# Print the sorted list
print("Vetor ordenado:", vetor_ordenado)