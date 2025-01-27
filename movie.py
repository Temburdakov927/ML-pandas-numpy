# -*- coding: utf-8 -*-
"""

# Машинное обучение в задачах анализа текста

**Цели работы:**
1. Решить задачу классификации фильмов и сериалов по жанру, возрастным ограничениям на обучающей выборке
2. Проверить качество правильного предсказания модели на тестовой выборке

**Задачи работы:**
Так как перед нами стоит задача классификации, необходимо присвоить каждому объекту метки: возрастное ограничение, жанр, является кинолента фильмом или сериалом. Для этого необходимо:
1. Предобработать данные
2. Разделить исходный набор данных на тестовую и обучающую выборки
3. Построить модель логистической регрессии
4. Обучить модель на данных обучающей выборки
5. Оценить качество модели с помощью тестовой выборки
6. Сделать выводы

**Ожидаемый результат:**
1. Модель, которая может с высокой вероятностью определить жанр и возрастные ограничения фильма или сериала

**Описание используемого датасета:**
В процессе работы будет использован датасет, состоящий из сериалов и фильмов стримингового сервиса Netflix. Для каждого элемента (фильма или сериала) присутствуют различные показатели, но в рамках данной курсовой работы наиболее важными являются:
1. Тип (фильм или сериал)
2. Краткое описание шоу, данное самим стриминговым сервисом
3. Краткое описание сюжета кинокартины
4. Возрастные ограничения
5. Присутствует ли в киноленте контент для взрослых
6. Жанр



Импортируем все необходимые библиотеки
"""

import pandas as pd #обработка данных
import numpy as np #работа с векторами,матрицами,массивами


import matplotlib.pyplot as plt #Визуализация данных(диаграммы)
from wordcloud import WordCloud #Облако слов(тоже визуализация)

import nltk #Обработка естественного языка
from sklearn.feature_extraction.text import TfidfVectorizer #Перевод слов в числовую форму понятную для компьютера

from sklearn.model_selection import train_test_split #разбиение данных на тренировочные и тестовые
from sklearn.linear_model import LinearRegression,Ridge,Lasso #Линейная регрессия
from collections import Counter #Счётчик(нужен для подсчёта колличества)
from nltk.corpus import wordnet #Проводит Лемматизацию, возвращает упрощённые слова, провдя морфологический разбор
from sklearn.svm import SVC #Метод опорных векторов(не стал использовать ,можно и удалить)
from sklearn.neighbors import KNeighborsClassifier #K ближайших соседей(не стал использовать ,можно и удалить)
from sklearn.feature_selection import SelectKBest, chi2 #Отбор фич (feature selection) — важная составляющая машинного обучения
from sklearn.ensemble import RandomForestClassifier #ещё одна модель обучения(не стал использовать ,можно и удалить)
from sklearn.pipeline import Pipeline #Последовательные стадии преобразования данных, предшествующие их загрузке в Модель
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score, classification_report,explained_variance_score,mean_absolute_error,r2_score #Оценка эффективности обучения модели
from sklearn.naive_bayes import GaussianNB #Импорт наивного баёерского классификатора
from sklearn import tree #для классификации и регрессии
from sklearn.tree import DecisionTreeClassifier#метод дерева
from sklearn.ensemble import GradientBoostingClassifier# градиентный бустинг
from sklearn.ensemble import AdaBoostClassifier#адаптивный бустинг
nltk.download('punkt')# делит текст на список предложений
nltk.download('wordnet')# проводит лемматизацию
nltk.download('stopwords')# поддерживает удаление стоп-слов
nltk.download('averaged_perceptron_tagger')#Определяет часть речи у токена
from time import time #замер времени
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

"""Оставляем только нужные для обучения модели столбцы"""

df = pd.read_csv('netflix_list.csv')
df = df[["type", "summary", "plot", "certificate", "isAdult", "genres"]]
df.info()

"""Исключаем нанов и дубликатов, смотрим информацию, чтобы свериться, что во всех столбцах одинаковое количество строк"""

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.info()

"""После удаление Nan,как мы видим выше счётчик остался без изменения(Он всё ещё заканчивается на 7006-ом элементе)

Обнуляем счётчик и проверяем
"""

df.reset_index(drop=True, inplace=True)
df

"""Проверяем какие значения имеет "контент для взрослых", как видим все фильмы и сериалы не содержат взрослого контента"""

df['isAdult'].value_counts()

"""Теперь смотрим какие возрастные ограничения есть в таблице"""

df['certificate'].value_counts()

"""Преобразуем все возрастные ограничения разных стран в цифровые ограничения"""

df = df.replace(['18+','16+','13+','15+','12+','7+',' 18','All','G','NC-17','PG-13','R','PG','U','A','UA'],['18','16','13','15','12','7','18','0+','0+','18','13','17','13','0+','14','12'])
df['certificate'].value_counts()

"""Удаляем строки без рейтинга и запрещённые, так как эти данные нам не понадобятся"""

list2 = ['Not Rated','(Banned)','Unrated']
df = df[df.certificate.isin(list2) == False]

"""Ещё раз обнуляем счётчик"""

df.reset_index(drop=True, inplace=True)

"""Соединяем обе ячейки, так как и там, и  там находится нужная нам информация по фильмам и сериалам"""

df['full_text'] = df['summary'] + ' ' + df['plot']

"""Делаем копию, которая содержит только текст, смотрим первые 5 текстов"""

df_text = df['full_text'].copy()
df_text.head()

"""Смотрим колличество уникальных слов во всех предложениях"""

tokenized_raw = [nltk.tokenize.word_tokenize(x) for x in list(df_text)]
len(set([y for x in tokenized_raw for y in x]))

"""Смотрим на диаграмме какие возрастные группы преобладают"""

plt.figure(figsize=(12,6))
plt.pie(df['certificate'].value_counts(), labels=['18','16','13','17','14','0+','12','7','15'], autopct='%0.2f')
plt.show()

"""Создаём массив стоп-слов"""

stop = nltk.corpus.stopwords.words('english')
stop.append(':')
stop.append(',')
stop.append('.')
stop.append('?')
stop.append('!')
stop.append('(')
stop.append(')')
stop.append('-')
stop.append('[')
stop.append(']')
stop.append('the')
stop.append('...')
stop.append('&')
stop.append('..')
stop.append(';')
stop.append('n\'t')
stop.append('’')
stop.append('a')
stop.append('wa')
stop.append('\'s')
stop.append("''")
stop.append('``')
stop.append('2')
stop.append('4')
stop.append('#')
stop.append('*')

"""# Токенизация, исключение стоп слов и схожих слов.
функция process обрабатывает текст(токенизация,лемматизация по возращённому значению из функции get_wordnet_pos) возвращает обработанную строчку датасета
функция get_wordnet_pos определяет часть речи слова и возвращает это слова в виде существительного
функция joins принимает значение от функции process и создаёт массив из разрозненных строк
"""

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
lemmatize = nltk.WordNetLemmatizer()
def process(description):
  string = []
  for i in description:
    words = [nltk.word_tokenize(i, language = "english")]
    word1 = [word.lower() for x in words for word in x if word.lower() not in stop]
    wordss = [lemmatize.lemmatize(x, get_wordnet_pos(x)) for x in word1 ]
    string.append(wordss)
  return string

def joins(description):
  df_stream =[]
  for i in description:
    string = [' '.join(i)]
    df_stream.append(string)
  return df_stream
steem = joins(process(df_text))

"""Добавляем в наш датасет новую столбец с обработанным текстом"""

df['clean_text'] = pd.DataFrame(steem)

"""Смотрим, что получилось"""

df

"""Проверяем на сколько уменьшилось колличество уникальных слов после обработки текста"""

text_corpus = []
text_abc = [word.split(' ') for word in df['clean_text'].tolist()]
for mail in text_abc:
  for word in mail:
    text_corpus.append(word)

len(set(text_corpus))

"""Генерируем облако слов, чем крупнее слово ,тем чаще оно встречается"""

y =' '.join(text_corpus)
wordcloud_y = WordCloud(width=1600, height=800).generate(y)

plt.figure( figsize=(20,10), facecolor='k')
# добавляем туда облако слов
plt.imshow(wordcloud_y)
# выключаем оси и подписи
plt.axis("off")
# убираем рамку вокруг
plt.tight_layout(pad=0)
# выводим картинку на экран
plt.show()

"""У меня всё отображается корректно, попробуйте использовать Colab


Удаляем непонятный жанр
"""

df = df[df.genres != '\\N']

"""Ещё раз удаляем повторы и пустые значения"""

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

"""Некоторые жанры состоят из нескольких слов, мы пробегаемся по каждой строке датасета и разделяем подобные перечесления на отдельные слова.

Например,
['Боевик,Приключения,Триллер'] --> ['Боевик','Приключения','Триллер']
Так же подсчитываем колличество уникальных жанров(в сложных жанрах считаем,что ключевой жанр является первым,чтобы не было слишком много классов)
"""

text_corpus2 = []
text_abc2 = [word.split(',') for word in df['genres'].tolist()]
for sentence in text_abc2:
  for word in sentence:
    text_corpus2.append(word)

len(set(text_corpus2))

"""Добавляем эти жанры в список"""

spisok = [i for i in set(text_corpus2)]

"""В сложных жанрах считаем,что ключевой жанр является первым,чтобы не было слишком много классов, удаляем остальные классы, оставляем только первые в кажой строчке датасета"""

text_abc3 = [[sentence[0]] for sentence in text_abc2]
df['genres'] = pd.DataFrame(text_abc3)

"""Создаём словарь, нумеруем каждый жанр от 0 до 26, чтобы обучать модель, меняем название жанра на число в датасете"""

res = dict(zip(spisok, [i for i in range(len(spisok))]))
df.replace(res,inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

"""Берём за X очищенный текст и по нему будем обучать модель определять y(жанр фильма)"""

tfidf = TfidfVectorizer()

X = df['clean_text']
y = df['genres']

df['clean_text'].shape

"""Разбиваем данные на обучающую и тест выборку."""

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

"""Обучение логистической регрессии

Этот код не менял, вам нужно его просто запустить,я его не запускал
"""

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([
                ('vect', CountVectorizer(analyzer='char', ngram_range =(2,10))),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=3,C=1e5, solver='saga',
                                           multi_class='multinomial',
                                           max_iter=1000,
                                           random_state=42)),
])
start = time()
logreg.fit(X_train, y_train)
train_time = time() - start
start = time()
y_pred = logreg.predict(X_test)
predict_time = time()-start
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
print(classification_report(y_test, y_pred))
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)

"""Используется CountVectorizer на ngram с analyzer = char. Именно такой подход дал наилучшую точность на маленьких текстах. В зависимости от данных значение ngram можно корректировать. Целевая метрика – F1.

# Изменяем возрастные ограничение на числа от 0 до 8, чтобы обучить модель определять по тексту возрастное ограничение фильма/сериала
"""

df = df.replace(['18','16','13','17','14','0+','12','7','15'],['0','1','2','3','4','5','6','7','8'])

"""Проделываем те же шаги, что и с жанрами"""

X2 = df['clean_text']
y2 = df['certificate']

X_train2,X_test2,y_train2,y_test2 = train_test_split(X2,y2,test_size=0.25,random_state=42)

logreg = Pipeline([
                ('vect', CountVectorizer(analyzer='char', ngram_range =(3,15))),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=3,C=1e5, solver='saga',
                                           multi_class='multinomial',
                                           max_iter=1000,
                                           random_state=42)),
])

start = time()
logreg.fit(X_train2, y_train2)
train_time = time() - start
start = time()
y_pred2 = logreg.predict(X_test2)
predict_time = time()-start
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
print(classification_report(y_test2, y_pred2))
print(f"F1 Score: {f1_score(y_test2, y_pred2, average='weighted')}")
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)

"""Список используемой литературы:
https://www.helenkapatsa.ru/paiplain/
https://habr.com/ru/companies/ods/articles/323890/
https://habr.com/ru/articles/514818/
https://habr.com/ru/articles/264915/

инициализирует объект класса TfidfVectorizer().Это нам нужно в дальнейшем, чтобы переводить слова из очищенного текста в цифры от 0 до 1, чтобы компьютер нас понимал,точно так же мы до этого заменили жанры числами
"""

tfidf = TfidfVectorizer()

x = tfidf.fit_transform(df['clean_text'])
y = df['genres']
x_train3,x_test3,y_train3,y_test3 = train_test_split(x,y,test_size=0.25,random_state=42)

"""Обучаем несколько моделей KNN(k ближайших соседей), смотрим какое значение параметра k самое оптимальное"""

accuracy = []
num_neigh = []

for ii in range(10,200):
    knc = KNeighborsClassifier(n_neighbors=ii)
    knc.fit(x_train3,y_train3)
    accuracy.append(knc.score(x_test3,y_test3))
    num_neigh.append(ii)

print(accuracy)
plt.scatter(num_neigh,accuracy)
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.show();

"""Сужаем диапазон подбора значений от 50 до 75, так как в этом диапазоне самый высокие показатель"""

accuracy = []
num_neigh = []

for ii in range(50,75):
    knc = KNeighborsClassifier(n_neighbors=ii)
    knc.fit(x_train3,y_train3)

    accuracy.append(knc.score(x_test3,y_test3))
    num_neigh.append(ii)

print(accuracy)
plt.scatter(num_neigh,accuracy)
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.show();

"""Искомое значение 67"""

knc = KNeighborsClassifier(67)
start = time()
knc = knc.fit(x_train3,y_train3)
train_time = time() - start
start = time()
y_pred3 = knc.predict(x_test3)
predict_time = time()-start
print(classification_report(y_test3, y_pred3))
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)

"""Обучаем модель метод опорных векторов"""

svc = SVC()
start = time()
svc.fit(x_train3,y_train3)
train_time = time() - start
start = time()
y_pred3 = svc.predict(x_test3)
predict_time = time()-start
print(classification_report(y_test3, y_pred3))
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)

"""На основе [этой](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769) статьи делаем поправки и улучшаем точность модели"""

svc = SVC(kernel='sigmoid', C=100, gamma=0.01)
start = time()
svc.fit(x_train3,y_train3)
train_time = time() - start
start = time()
y_pred3 = svc.predict(x_test3)
predict_time = time()-start
print(classification_report(y_test3, y_pred3))
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)

"""обучения наивного байерского классификатора"""

x = tfidf.fit_transform(df['clean_text']).toarray()
y = df['genres']
x_train4,x_test4,y_train4,y_test4 = train_test_split(x,y,test_size=0.25,random_state=42)

gaussianNB = GaussianNB()
start = time()
gaussianNB.fit(x_train4,y_train4)
train_time = time() - start
start = time()
y_pred4 = gaussianNB.predict(x_test4)
predict_time = time()-start
print(classification_report(y_test4, y_pred4))
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)

"""Теперь можно попробовать рассмотреть некоторые ансамблевые методы машинного обучения, такие как адаптивный бустинг и градиентный бустинг. Что мы знаем про ансамблевые методы? В ансамблевых методах несколько моделей обучаются для решения одной и той же проблемы и объединяются для получения более эффективных результатов. Основная идея заключается в том, что при правильном сочетании моделей можно получить более точную и надежную модель.
Обучаем адаптивный бустинг
"""

modelClf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.33, random_state = 42)
start = time()
modelClf.fit(X_train, y_train)
train_time = time() - start
start = time()
y_pred4 = modelClf.predict(X_valid)
predict_time = time()-start
print(classification_report(y_valid, y_pred4))
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)

"""Градиентный бустинг, также как и адаптивный, обучает слабые алгоритмы последовательно, исправляя ошибки предыдущих. Принципиальное отличие этих алгоритмов заключается в способах изменения весов. Адаптивный бустинг использует итеративный метод оптимизации, в то время как градиентный оптимизирует веса с помощью градиентного спуска.
Обучаем  градиентный бустинг
"""

modelClf = GradientBoostingClassifier(max_depth=2, n_estimators=150,random_state=12, learning_rate=1)
start = time()
modelClf.fit(x_train4, y_train4)
train_time = time() - start
start = time()
y_pred4 = modelClf.predict(X_valid)
predict_time = time()-start
print(classification_report(y_valid, y_pred4))
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)

"""Обучаем новую модель по гиперпараметрам,задаём отрезки значений в них,чтобы найти лучшие параметры,которые покажут наибольшее значение на выходе"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
from scipy.stats import randint

rs_space={'max_depth':list(np.arange(10, 100, step=10)) + [None],
              'n_estimators':np.arange(10, 500, step=50),
              'max_features':randint(1,7),
              'criterion':['gini','entropy'],
              'min_samples_leaf':randint(1,4),
              'min_samples_split':np.arange(2, 10, step=2)
}

from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(rf, rs_space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=3)
start = time()
model_random = rf_random.fit(x,y)
train_time = time() - start

"""Выводим наибольшую оценку и те гиперпараметры,которые её дали"""

print('Best hyperparameters are: '+str(model_random.best_params_))
print('Best score is: '+str(model_random.best_score_))
print("\tTraining time: %0.3fs" % train_time)

"""Смотрим наибольшую оценку у нескольких моделей(Knn ,Ridge лидируют)"""

regressors = [
    KNeighborsRegressor(),
    GradientBoostingRegressor(),
    KNeighborsRegressor(),
    ExtraTreesRegressor(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    LinearRegression(),
    Lasso(),
    Ridge()
]

"""тренируем модели,смотрим время обучения каждой и их точность, в целом я поменял подход ,так что код выше с моделями обучения можно убрать(те модели,которые были до поправок)"""

head = 10
for model in regressors[:head]:
    start = time()
    model.fit(x_train3, y_train3)
    train_time = time() - start
    start = time()
    y_pred = model.predict(x_test3)
    predict_time = time()-start
    print(model)
    print("\tTraining time: %0.3fs" % train_time)
    print("\tPrediction time: %0.3fs" % predict_time)
    print("\tExplained variance:", explained_variance_score(y_test, y_pred))
    print("\tMean absolute error:", mean_absolute_error(y_test, y_pred))
    print("\tR2 score:", r2_score(y_test, y_pred))
    print()

"""Импортируем библиотеки для Grid Search для  модели Ridge()"""

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import RepeatedKFold

"""Инициализируем объект ,настраиваем парметры во второй строке"""

model = Ridge()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

"""задаём отрезок гиперпараметров,чтобы подобрать лучший из них"""

param_grid = {"alpha": 10.0 ** np.arange(-5, 6),
    'solver':['svd', 'cholesky', 'lsqr', 'sag'],
    'fit_intercept':[True, False]
}

"""Обучаем модель по гиперпараметрам"""

search = GridSearchCV(model, param_grid, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
start = time()
result = search.fit(x, y)
train_time = time() - start

"""Выводим наибольшую оценку и те гиперпараметры,которые её дали"""

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
print("\tTraining time: %0.3fs" % train_time)

"""Отрицательное значение говорит о том,что бывают ситуации, когда коэффициент принимает отрицательные значения (обычно небольшие). Это произошло, потому что ошибка модели среднего стала меньше ошибки модели с переменной. В этом случае оказывается, что добавление в модель с константой некоторой переменной только ухудшает её (т.е. регрессионная модель с переменной работает хуже, чем предсказание с помощью простой средней), поэтому обучим ещё одну модель KNeighborsClassifier()

Обучаем модель и сравниваем результаты в промежутке от 1 до 310
"""

knn = KNeighborsClassifier()
k_range = list(range(1, 310))
param_grid = dict(n_neighbors=k_range)

# defining parameter range
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)

# fitting the model for grid search
start = time()
grid_search=grid.fit(x_train3, y_train3)
train_time = time() - start

"""Наиболее высокий результат дал гиперпараметр K со значением 119"""

print(grid_search.best_params_)
print("\tTraining time: %0.3fs" % train_time)

"""Выводим точность модели"""

accuracy = grid_search.best_score_ *100
print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy) )

"""Таким образом,наилучшей моделью для данной задачи является KNeighborsClassifier()"""
