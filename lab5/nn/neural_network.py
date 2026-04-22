import time

import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential, model_from_json


# Создаем новую модель НС и обучаем
def create_and_train_model(in_data_train, out_data_train):
    model = Sequential()  # Модель НС - сеть прямого распространения
    model.add(Input(shape=(in_data_train.shape[1],)))
    model.add(Dense(5, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))

    # Компилируем модель и устанавливаем параметры оптимизатора весов (алгоритма обучения)
    model.compile(loss='mean_absolute_error', optimizer=tf.optimizers.RMSprop(learning_rate=0.005))

    start_time = time.time()
    history = model.fit(in_data_train, out_data_train, epochs=1000, batch_size=32)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Input size: ", in_data_train.shape[1])
    model.summary()

    return model


# Сохраняем структуру модели в файл model.json, а весовые коэффициенты в weights.h5
def save_model(model):
    json_file = 'new_model.json'
    model_json = model.to_json()

    with open(json_file, 'w') as f:
        f.write(model_json)

    model.save_weights('new_model.weights.h5')


# Загружаем модель из файла json_file, а веса из weights.h5
def load_model(json_file, weights_file):
    with open(json_file, 'r') as f:
        loaded_model = model_from_json(f.read())

    loaded_model.load_weights(weights_file)

    return loaded_model


if __name__ == '__main__':
    dataset = pd.read_csv("taxi_trip_pricing.csv", sep=',', header='infer', names=None,
                          encoding="utf-8").dropna()
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(dataset.shape)
    # print(dataset.head(5))

    in_data = dataset.iloc[:10000, :10]
    out_data = dataset.iloc[:10000, 10:11].values

    # Нормализация данных - линейная нормализация
    norm = MinMaxScaler()  # Нормализатор для входных данных
    norm_out = MinMaxScaler()  # Нормализатор для выходных данных
    out_data = norm_out.fit_transform(out_data)  # Нормализуем выходные значения

    # Названия колонок для one-hot кодирования
    one_hot_cols = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
    for col_name in one_hot_cols:
        one_hot = pd.get_dummies(in_data[col_name])
        in_data = in_data.drop(col_name, axis=1)
        in_data = in_data.join(one_hot)

    # Трансформируем колонки с True и False значениями в 1 и 0 соответственно
    bin_cols = []
    for col_name in bin_cols:
        in_data[col_name] = in_data[col_name].astype(int)

    # Выводим первые 5 строк нормализованных данных для проверки
    print(in_data.head(5))

    # Делаем MinMax-нормализацию входных значений
    in_data = norm.fit_transform(in_data.values)
    print("Размерность входных данных: ", in_data.shape)

    in_data_train, in_data_test, out_data_train, out_data_test = train_test_split(in_data, out_data, test_size=0.1)

    # Создаем новую модель НС и обучаем на тренировочной выборке
    model = create_and_train_model(in_data_train, out_data_train)

    # model = load_model('model.json', 'weights.h5')

    out_pred = model.predict(in_data_test)

    # Полученные (out_pred) и соответствующие им эталонные значения (out_data_test) помещаем в стандартные списки
    predicted = [out[0] for out in out_pred]
    test = [out[0] for out in out_data_test]

    # Вычисляем относительную ошибку
    approx_err = mean_absolute_percentage_error(predicted, test)
    print('Approximation error:', approx_err * 100)

    # Раскомментируйте строку ниже, если хотите вывести выходные значения сети,
    # приведенные к РЕАЛЬНЫМ величинам (выполняется операция обратная нормализации)
    # print(norm_out.inverse_transform(out_data_test))

    print("Save model to file ?:")
    q = input()
    if q.lower() == 'y':
        save_model(model)
