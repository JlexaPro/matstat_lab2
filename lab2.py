# Статическая обработка большой выборки
# Все задания приведены в отчете


# Импортируем модули
import pandas as pd
import numpy as np
import scipy.stats
from counter import *
from matplotlib import pyplot as plt

# Загрузим данные из .xlsx файла
df = pd.read_excel("lab2/data_x.xlsx", "Лист1")

# Сброс ограничений на вывод датафрейма
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Увеличение области отображения вывода
desired_width = 200
pd.set_option('display.width', desired_width)

# Вывод по столбцам из экселя
x = df["Выборка"].tolist()

n = len(x)  # объем выборки
# Находим длину интервала
# h = np.round((max(x) - min(x)) / (1 + 3.322 * np.log10(n)), 3)
h = np.round((max(x) - min(x)) / 10, 3)


# print(n)
# print(x)

# 1 задание
def table(x, n, h):
    print(f"Минимальный элемент выборки: {min(x)}, максимальный элемент выборки: {max(x)}")
    print(f'h = {h} \n')
    # Создаем DataFrame
    data = [[f"({min(x)}; {min(x) + h})", f"({min(x) + h}; {min(x) + 2 * h})",
             f"({min(x) + 2 * h}; {np.round(min(x) + 3 * h, 3)})",
             f"({np.round(min(x) + 3 * h, 3)}; {min(x) + 4 * h})", f"({min(x) + 4 * h}; {min(x) + 5 * h})",
             f"({min(x) + 5 * h}; {np.round(min(x) + 6 * h, 3)})",
             f"({np.round(min(x) + 6 * h, 3)}; {min(x) + 7 * h})", f"({min(x) + 7 * h}; {min(x) + 8 * h})",
             f"({min(x) + 8 * h}; {np.round(min(x) + 9 * h, 2)})",
             f"({min(x) + 9 * h}; {np.round(min(x) + 10 * h, 2)})"],
            [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10],
            [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10],
            [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10],
            [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10],
            [
                p1, p1 + p2, p1 + p2 + p3, p1 + p2 + p3 + p4, p1 + p2 + p3 + p4 + p5, p1 + p2 + p3 + p4 + p5 + p6,
                    p1 + p2 + p3 + p4 + p5 + p6 + p7, p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8,
                    p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9, p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10]]
    df = pd.DataFrame(data, index=['Разряд', u'm\u2097', u'p\u2097', u'f\u2097', u'x\u2097', u'p\u2097*'],
                      columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(df)


def func_emp():
    # Эмпирическая функции распределения
    print("2. Построить эмпирическую функцию распределения F*(x).")
    df1 = pd.read_excel("lab2/data_x.xlsx", "Лист2")
    x = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
    y1 = df1["Накопленная частота"].tolist()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot()

    ax.set_title("Постороение эмпирической функции распределения")
    ax.set_xlabel("Экспериментальные данные")
    ax.set_ylabel("Накопленная частота")

    x.sort()
    ax.step(x, y1, label="F(x)")

    ax.grid()
    ax.legend()
    plt.show()
    # print(df.head(20))


def histogram():
    # Гистограмма относительных частот
    pass


# Оценка математического ожидания
def mathematical_expectation():
    print("\n2. Оценка математического ожидания")
    print("Объем выборки: " + str(len(x)) +
          "\nМаксимум: " + str("%.3f" % max(x)) +
          "\nМинимум: " + str("%.3f" % min(x)) +
          "\nРазмах выборки: " + str("%.2f" % ((max(x)) - min(x))))

    # Математическое ожидание - среднее значение случайной величины
    global mean
    mean = sum(x) / len(x)
    print("Математическое ожидание: " + str("%.3f" % mean))


# Несмещенная оценка дисперсии и среднеквадратичное отклонение
def standard_deviation():
    print("\n3. Найти несмещенную оценку дисперсии и среднеквадратичного отклонения.")
    # Несмещенная оценка дисперсии
    s2 = 1 / (len(x) - 1) * sum((i - mean) ** 2 for i in x)

    # Среднеквадратическое отклонение
    global s2n
    s2n = s2 ** (1 / 2)
    print("Несмещенная оценка дисперсии: " + str("%.3f" % s2) +
          "\nСреднеквадратическое отклонение: " + str("%.3f" % s2n))

    # Оценка коэффициента вариации
    k = s2n / mean
    print(f"Оценка коэффициента вариации: {'%.3f' % k}")


# Доверительный интервал
def confidence_interval():
    print(
        "\n4. Найти доверительный интервал для математического ожидания и дисперсии при значениях доверительной"
        "вероятности β = 0. 9 и β = 0. 95.")
    confidence1 = 0.9
    confidence2 = 0.95

    # Доверительный интервал для математического ожидания
    print("\nДоверительный интервал для математического ожидания")
    a = 1.0 * np.array(x)
    m, se = np.mean(a), scipy.stats.sem(x)
    h1 = se * scipy.stats.t.ppf((1 + confidence1) / 2., n - 1)
    h2 = se * scipy.stats.t.ppf((1 + confidence2) / 2., n - 1)

    print("\tПри β = 0. 9, получаем: \t\t\t\t\tПри β = 0. 95, получаем:"
          "\n\tt-распределение: ", "%.3f" % scipy.stats.t.ppf((1 + confidence1) / 2., n - 1),
          "\t\t\t\t\tt-распределение: ", "%.3f" % scipy.stats.t.ppf((1 + confidence2) / 2., n - 1),
          "\n\tЛевая граница: ", str("%.3f" % (m - h1)),
          "\t\t\t\t\t\tЛевая граница: ", str("%.3f" % (m - h2)),
          "\n\tПравая граница: ", str("%.3f" % (m + h1)),
          "\t\t\t\t\t\tПравая граница: ", str("%.3f" % (m + h2)))

    # Доверительный интервал для дисперсии
    print("\nДоверительный интервал для дисперсии")
    se1 = np.std(a) * np.std(a) * n / (n - 1)
    print("\tПри β = 0. 9, получаем: \t\t\t\t\tПри β = 0. 95, получаем:"
          "\n\tЛевая граница: ",
          str("%.3f" % ((n - 1) * se1 / scipy.stats.chi2.ppf((2 - (1 - confidence1)) / 2., n - 1))), "(",
          str("%.3f" % np.sqrt((n - 1) * se1 / scipy.stats.chi2.ppf((2 - (1 - confidence1)) / 2., n - 1))), ")",
          "\t\t\t\tЛевая граница: ",
          str("%.3f" % ((n - 1) * se1 / scipy.stats.chi2.ppf((2 - (1 - confidence2)) / 2., n - 1))), "(",
          str("%.3f" % np.sqrt((n - 1) * se1 / scipy.stats.chi2.ppf((2 - (1 - confidence2)) / 2., n - 1))), ")",
          "\n\tПравая граница: ", str("%.3f" % ((n - 1) * se1 / scipy.stats.chi2.ppf((1 - confidence1) / 2., n - 1))),
          "(", str("%.3f" % np.sqrt((n - 1) * se1 / scipy.stats.chi2.ppf((1 - confidence1) / 2., n - 1))), ")",
          "\t\t\t\tПравая граница: ", str(
            "%.3f" % ((n - 1) * se1 / scipy.stats.chi2.ppf((1 - confidence2) / 2., n - 1))), "(",
          str("%.3f" % np.sqrt((n - 1) * se1 / scipy.stats.chi2.ppf((1 - confidence2) / 2., n - 1))), ")")


def main():
    table(x, n, h)
    func_emp()
    mathematical_expectation()
    standard_deviation()
    confidence_interval()


if __name__ == '__main__':
    main()
