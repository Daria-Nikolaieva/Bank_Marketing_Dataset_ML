import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def dist_box(dataset, column):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      plt.figure(figsize=(16,6))

      plt.subplot(1,2,1)
      sns.distplot(dataset[column], color = 'purple')
      pltname = 'Графік розподілу для ' + column
      plt.ticklabel_format(style='plain', axis='x')
      plt.title(pltname)

      plt.subplot(1,2,2)
      red_diamond = dict(markerfacecolor='r', marker='D')
      sns.boxplot(y = column, data = dataset, flierprops = red_diamond)
      pltname = 'Боксплот для ' + column
      plt.title(pltname)

      plt.show()

def bi_cat_countplot(df, column, hue_column):
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14,6)

    pltname = f'Нормалізований розподіл значень за категорією: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=axes[0], title=pltname)

    # анотація значень в барплоті
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%')

    pltname = f'Кількість даних за категорією: {column}'
    counts = df.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=axes[1], title=pltname)

    for container in ax.containers:
      ax.bar_label(container)


def uni_cat_target_compare(df, column):
    bi_cat_countplot(df, column, hue_column='y' )


def bi_countplot_target(df0, df1, column, hue_column):
  pltname = 'Клієнт зі складнощами щодо платності'
  print(pltname.upper())
  bi_cat_countplot(df1, column, hue_column)
  plt.show()

  pltname = 'Клієнти зі своєчасними платежами'
  print(pltname.upper())
  bi_cat_countplot(df0, column, hue_column)
  plt.show()


def plot_macro_features(df, features, target='y'):
    """
    Визуализація розподілу макроэкономічних ознак по цільовій змінній y
    """
    n = len(features)
    plt.figure(figsize=(15, 3*n))

    for i, col in enumerate(features, 1):
        plt.subplot(n, 1, i)
        for value, color in zip(df[target].unique(), ['blue','orange']):
            subset = df[df[target] == value][col]
            sns.kdeplot(subset, fill=True, alpha=0.4, label=value, color=color)
        plt.title(f"Розподіл {col} в залежності від {target}")
        plt.xlabel(col)
        plt.ylabel("Плотність")
        plt.legend()

    plt.tight_layout()
    plt.show()


def freqline_by_target_count(df, col, target='y', bins=30):
    """
    Линія по числу объектів (частота), розділених по target.
    """
    plt.figure(figsize=(8,5))
    
    for value, color in zip(df[target].unique(), ['blue','orange']):
        subset = df[df[target]==value][col]
        counts, edges = np.histogram(subset, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        plt.plot(centers, counts, color=color, label=value)
        plt.fill_between(centers, counts, alpha=0.3, color=color)
    
    plt.title(f'Розподіл {col}, в залежності від {target}')
    plt.xlabel(col)
    plt.ylabel('Кількість')
    plt.legend(title=target)
    plt.show()


def freqline_by_target(df, col, target='y', bins=30):
    """
    Будує линію с заповненням по частотам числової ознаки col, розділеної по target.
    """
    plt.figure(figsize=(8,5))
    
    for value, color in zip(df[target].unique(), ['blue','orange']):
        subset = df[df[target]==value][col]
        counts, edges = np.histogram(subset, bins=bins)
        # Преобразуем в плотность относительно количества объектов
        counts = counts / counts.sum()
        centers = (edges[:-1] + edges[1:]) / 2
        plt.plot(centers, counts, color=color, label=value)
        plt.fill_between(centers, counts, alpha=0.3, color=color)
    
    plt.title(f'Розподіл {col} в залежності від {target} (частота)')
    plt.xlabel(col)
    plt.ylabel('Частота')
    plt.legend(title=target)
    plt.show()