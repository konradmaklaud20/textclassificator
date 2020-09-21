import numpy as np
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split


df = pd.read_csv('lenta_class.csv')  # обработанный файл

# определяем гиперпараметры
num_words = 50000
maxlen = 1075
embedding_dim = 100
epochs = 5
batch_size = 32

tokenizer = Tokenizer(num_words=num_words, lower=True)
tokenizer.fit_on_texts(df['new_text'].values)
word_index = tokenizer.word_index


X = tokenizer.texts_to_sequences(df['new_text'].values)
X = pad_sequences(X, maxlen=maxlen)

Y = df['tag_code'].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=2020)

model = Sequential()
model.add(Embedding(num_words, embedding_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(150, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(66, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_model = model.fit(X_train, Y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        )

model.save('lenta_class_model.h5')

accuracy = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.2f}  Accuracy: {:0.2f}'.format(accuracy[0], accuracy[1]))

# ------------------------------------------------------------------------------------ #

new_model = keras.models.load_model('lenta_class_model.h5')

new_input_text = ['']  # подставить текст
sequence = tokenizer.texts_to_sequences(new_input_text)
padded = pad_sequences(sequence, maxlen=maxlen)
pred = new_model.predict(padded)
labels = sorted(['Политика', 'Регионы',
                 'Происшествия', 'ТВ и радио', 'Музыка',
                 'Следствие и суд', 'Общество', 'Украина', 'Вирусные ролики', 'Футбол',
                 'Преступность', 'Госэкономика', 'Оружие', 'Криминал', 'Бизнес', 'Интернет',
                 'Экономика', 'Гаджеты', 'Кино', 'Конфликты', 'Люди', 'События', 'Внешний вид',
                 'Хоккей', 'Квартира', 'Рынки', 'Мир', 'Театр', 'Деньги', 'Летние виды',
                 'Белоруссия', 'Звери', 'Зимние виды', 'Явления', 'Город', 'Полиция и спецслужбы',
                 'Наука', 'Игры', 'Бокс и ММА', 'Искусство', 'Стиль', 'Пресса', 'Космос',
                 'Coцсети', 'Инструменты', 'Еда', 'Техника', 'Офис', 'История', 'Деловой климат',
                 'Мировой бизнес', 'Средняя Азия', 'Дача', 'Россия', 'Молдавия', 'Закавказье',
                 'Мнения', 'Достижения', 'Движение', 'Книги', 'Прибалтика', 'Москва', 'Жизнь',
                 'Часы', 'Софт', 'Мемы'])

print(pred, labels[np.argmax(pred)])  # предсказывает, к кому классу принадлежит введённый текст

