import pymorphy2
import re
import pandas as pd

# ссылка на исходный датасет https://www.kaggle.com/yutkin/corpus-of-russian-news-articles-from-lenta

df = pd.read_csv('lenta-ru-news.csv')
df = df.drop(['url', 'topic', 'title'], axis=1)  # удаляем лишние колонки
df = df.dropna().reset_index()


tags = df.groupby('tags').size() > 500  # отбиравем только те тэги, которые встречаются более 500 раз
tags = pd.DataFrame(tags)
tags = tags.rename(columns={0: '0'})
tags = tags.to_dict()
lst = []
new_tags = tags['0']

for i in new_tags:  # формируем новую колонку с тэгами
    if new_tags[i] is False:
        lst.append(i)

df = df.drop(df[df.tags == 'Все'].index)  # в датасете есть много тэгов общей категории. Удаляем их тоже
df = df[~df["tags"].isin(lst)].reset_index()
print(df['tags'])
df = df.sample(frac=1).reset_index(drop=True)

ma = pymorphy2.MorphAnalyzer()  # анализатор для преобразования слов к нормальной форме


def clean_text(text):
    """
    Функция для предобработки текста
    """
    text = text.lower()
    text = re.sub('\n', '', text)
    text = " ".join(ma.parse(word)[0].normal_form for word in text.split())
    text = ' '.join(word for word in text.split() if len(word) > 3)  # удаляем слова меньше 3 симвлов (предлоги и проч.)
    return text


categories = {}
for key, value in enumerate(df['tags'].unique()):  # создаём словарь с тэгами
    categories[value] = key + 1


df['tag_code'] = df['tags'].map(categories)
df['new_text'] = df.apply(lambda x: clean_text(x['text']), axis=1)

df.to_csv('lenta_class.csv', encoding='utf-8')
