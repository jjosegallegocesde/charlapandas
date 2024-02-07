import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos del archivo CSV en un DataFrame de Pandas
titanic_data = pd.read_csv('./titanic.csv')

# Calcular la probabilidad general de supervivencia
prob_supervivencia_general = titanic_data['Survived'].mean()

# Calcular la probabilidad de supervivencia por clase de pasajero
prob_supervivencia_por_clase = titanic_data.groupby('Pclass')['Survived'].mean()

# Calcular la probabilidad de supervivencia por género
prob_supervivencia_por_genero = titanic_data.groupby('Sex')['Survived'].mean()

# Categorizar la edad en dos grupos: Niños (< 18 años) y Adultos (>= 18 años)
titanic_data['AgeGroup'] = pd.cut(titanic_data['Age'], bins=[0, 18, 80], labels=['Niño', 'Adulto'])

# Calcular la probabilidad de supervivencia por grupo de edad
prob_supervivencia_por_edad = titanic_data.groupby('AgeGroup')['Survived'].mean()

# Gráfico para la Probabilidad General de Supervivencia
plt.figure(figsize=(6, 4))
plt.bar(['Supervivencia'], [prob_supervivencia_general], color='blue')
plt.ylabel('Probabilidad')
plt.title('Probabilidad General de Supervivencia')
plt.show()

# Gráfico para la Probabilidad de Supervivencia por Clase
plt.figure(figsize=(6, 4))
prob_supervivencia_por_clase.plot(kind='bar', color='green')
plt.ylabel('Probabilidad')
plt.title('Probabilidad de Supervivencia por Clase')
plt.show()

# Gráfico para la Probabilidad de Supervivencia por Género
plt.figure(figsize=(6, 4))
prob_supervivencia_por_genero.plot(kind='bar', color='red')
plt.ylabel('Probabilidad')
plt.title('Probabilidad de Supervivencia por Género')
plt.show()

# Gráfico para la Probabilidad de Supervivencia por Grupo de Edad
plt.figure(figsize=(6, 4))
prob_supervivencia_por_edad.plot(kind='bar', color='purple')
plt.ylabel('Probabilidad')
plt.title('Probabilidad de Supervivencia por Grupo de Edad')
plt.show()
