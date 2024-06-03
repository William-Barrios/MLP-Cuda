import matplotlib.pyplot as plt

# Leer datos del archivo
iterations = []
accuracies = []

with open('accuracy_data.txt', 'r') as file:
    for line in file:
        iteration, accuracy = line.split()
        iterations.append(int(iteration))
        accuracies.append(float(accuracy))

# Graficar los datos
plt.figure(figsize=(10, 6))
plt.plot(iterations, accuracies, marker='o')
plt.title('Precisión del Modelo Durante el Entrenamiento')
plt.xlabel('Iteración')
plt.ylabel('Precisión (%)')
plt.grid(True)
plt.show()