import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import *
# %matplotlib inline 


sns.set(style="ticks")
iris = load_wine()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
data1.head()
# Размер датасета - 8143 строк, 7 колонок
data1.shape
total_count = data1.shape[0]
print('Всего строк: {}'.format(total_count))

# Список колонок
print(data1.columns)

# Список колонок с типами данных
data1.dtypes

# Проверим наличие пустых значений
# Цикл по колонкам датасета
for col in data1.columns:
    # Количество пустых значений - все значения заполнены
    temp_null_count = data1[data1[col].isnull()].shape[0]
    print('{} - {}'.format(col, temp_null_count))

# Основные статистические характеристки набора данных
data1.describe()

#Определим уникальные значения для целевого признака
data1['target'].unique()

fig, ax = plt.subplots(figsize=(20,20)) 
sns.scatterplot(ax=ax, x='alcohol', y='color_intensity', data=data1)

fig, ax = plt.subplots(figsize=(20,20)) 
sns.scatterplot(ax=ax, x='alcohol', y='color_intensity', data=data1, hue='target')

fig, ax = plt.subplots(figsize=(20,20)) 
sns.distplot(data1['alcohol'])

sns.jointplot(x='alcohol', y='malic_acid', data=data1)

sns.jointplot(x='alcohol', y='malic_acid', data=data1, kind="hex")

sns.jointplot(x='alcohol', y='malic_acid', data=data1, kind="kde")

# sns.pairplot(data1)

# sns.pairplot(data1, hue="alcohol")

sns.boxplot(x=data1['alcohol'])

# По вертикали
sns.boxplot(y=data1['alcohol'])

# Распределение параметра Humidity сгруппированные по Occupancy.
sns.boxplot(x='malic_acid', y='alcohol', data=data1)

sns.violinplot(x=data1['color_intensity'])


fig, ax = plt.subplots(2, 1, figsize=(30,30))
sns.violinplot(ax=ax[0], x=data1['alcohol'])
sns.distplot(data1['alcohol'], ax=ax[1])

# Распределение параметра Humidity сгруппированные по Occupancy.
fig = plt.subplots(1,1,figsize=(30,30)) 
sns.violinplot(x='target', y='color_intensity', data=data1)

data1.corr()

data1.corr(method='pearson')

data1.corr(method='kendall')

data1.corr(method='spearman')

fig = plt.subplots(1,1,figsize=(30,30)) 
sns.heatmap(data1.corr())

# Вывод значений в ячейках
fig = plt.subplots(1,1,figsize=(30,30)) 
sns.heatmap(data1.corr(), annot=True, fmt='.3f')

# Изменение цветовой гаммы
sns.heatmap(data1.corr(), cmap='YlGnBu', annot=True, fmt='.3f')

# Треугольный вариант матрицы
mask = np.zeros_like(data1.corr(), dtype=np.bool)
# чтобы оставить нижнюю часть матрицы
# mask[np.triu_indices_from(mask)] = True
# чтобы оставить верхнюю часть матрицы
mask[np.tril_indices_from(mask)] = True
sns.heatmap(data1.corr(), mask=mask, annot=True, fmt='.3f')

fig, ax = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(30,30))
sns.heatmap(data1.corr(method='pearson'), ax=ax[0], annot=True, fmt='.1f')
sns.heatmap(data1.corr(method='kendall'), ax=ax[1], annot=True, fmt='.1f')
sns.heatmap(data1.corr(method='spearman'), ax=ax[2], annot=True, fmt='.1f')
fig.suptitle('Корреляционные матрицы, построенные различными методами')
ax[0].title.set_text('Pearson')
ax[1].title.set_text('Kendall')
ax[2].title.set_text('Spearman')
