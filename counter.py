# Перебирает список х (элементы выборки);
# Находит элементы, которые принадлежат интервалам;
# Подсчитывает эти элементы и выводит количество;
# Считаются остальные значения для таблицы частот.


# Импортируем модули
import pandas as pd
import numpy as np

df = pd.read_excel("lab2/data_x.xlsx")
# Вывод по столбцам из экселя
x = df["Выборка"].tolist()

n = len(x)  # объем выборки
#print(n)
h = np.round((max(x) - min(x)) / 9, 3)
# Отсортировать и вывести все элементы выборки
# x.sort()
# j = 0
# for i in x:
#     j += 1
#     print(f"{j} >>> {i}")

m1, m2, m3, m4, m5, m6, m7, m8, m9 = [], [], [], [], [], [], [], [], []  # Пустные списки

for i in x:
    if min(x) <= i <= min(x) + h:
        m1.insert(0, i)
    elif min(x) + h <= i <= min(x) + 2 * h:
        m2.insert(0, i)
    elif min(x) + 2 * h <= i <= min(x) + 3 * h:
        m3.insert(0, i)
    elif min(x) + 3 * h <= i <= min(x) + 4 * h:
        m4.insert(0, i)
    elif min(x) + 4 * h <= i <= min(x) + 5 * h:
        m5.insert(0, i)
    elif min(x) + 5 * h <= i <= min(x) + 6 * h:
        m6.insert(0, i)
    elif min(x) + 6 * h <= i <= min(x) + 7 * h:
        m7.insert(0, i)
    elif min(x) + 7 * h <= i <= min(x) + 8 * h:
        m8.insert(0, i)
    elif min(x) + 8 * h <= i <= min(x) + 9 * h:
        m9.insert(0, i)

m1, m2, m3, m4, m5, m6, m7, m8, m9 = len(m1), len(m2), len(m3), len(m4), len(m5), len(m6), len(m7), len(m8), len(m9)

# Частота попадания
p1, p2, p3, p4, p5, p6, p7, p8, p9 = m1 / n, m2 / n, m3 / n, m4 / n, m5 / n, m6 / n, m7 / n, m8 / n, m9 / n

# Плотность частоты
f1, f2, f3, f4, f5, f6, f7, f8, f9 = round(p1 / h, 3), round(p2 / h, 3), round(p3 / h, 3), round(p4 / h, 3), round(
    p5 / h, 3), round(p6 / h, 3), round(p7 / h, 3), round(p8 / h, 3), round(p9 / h, 3)

# Середина интервалов
x1, x2, x3, x4, x5, x6, x7, x8, x9 = round((2 * min(x) + h) / 2, 3), round((2 * min(x) + 3 * h) / 2, 3), round(
    (2 * min(x) + 5 * h) / 2, 3), round(
    (2 * min(x) + 7 * h) / 2, 3), round((2 * min(x) + 9 * h) / 2, 3), round((2 * min(x) + 11 * h) / 2, 3), round(
    (2 * min(x) + 13 * h) / 2, 3), round((2 * min(x) + 15 * h) / 2, 3), round((2 * min(x) + 17 * h) / 2, 3)
