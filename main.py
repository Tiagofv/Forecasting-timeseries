import keras
import pandas as pd
import sklearn.model_selection

def normalize_datetime(df: pd.DataFrame):
    # Extract features from the timestamp
    df['hour'] = df['Datetime'].dt.hour
    df['day_of_week'] = df['Datetime'].dt.dayofweek
    df['month'] = df['Datetime'].dt.month
    df['day_of_year'] = df['Datetime'].dt.dayofyear
    df['day_of_month'] = df['Datetime'].dt.day

    # Normalize these features to be in the range [0, 1]
    df['hour'] = df['hour'] / 23
    df['day_of_week'] = df['day_of_week'] / 6
    df['month'] = df['month'] / 11
    df['day_of_year'] = df['day_of_year'] / 365
    df['day_of_month'] = df['day_of_month'] / 31
    df.drop(columns=['Datetime'], inplace=True)
    return df

def normalize_dataset(df: pd.DataFrame, train_mean: float, train_std: float):
    df['DEOK_MW'] = (df['DEOK_MW'] - train_mean) / train_std
    return df
def load_dataset(path: str):
    # Load the dataset
    dataset = pd.read_csv(path, dtype={'DEOK_MW': float, 'Datetime': str})
    dataset['Datetime'] = pd.to_datetime(dataset['Datetime'])
    # split train and test
    train, test = sklearn.model_selection.train_test_split(dataset, test_size=0.2)
    # create validation dataset
    train, val = sklearn.model_selection.train_test_split(train, test_size=0.2)
    mean = train['DEOK_MW'].mean()
    std = train['DEOK_MW'].std()
    train = normalize_dataset(train, mean, std)
    test = normalize_dataset(test, mean, std)
    val = normalize_dataset(val, mean, std)
    return train, test, val

class NaiveImplementation:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.group_by_weekday_hour()

    def group_by_weekday_hour(self):
        self.dataset['day_of_week'] = self.dataset['Datetime'].dt.dayofweek
        self.dataset['hour'] = self.dataset['Datetime'].dt.hour
        self.dataset = self.dataset.groupby(['day_of_week', 'hour']).mean()
        return self.dataset

    def predict(self, datetime, ) -> float:
        weekday = datetime.weekday()
        hour = datetime.hour
        return self.dataset.loc[(weekday, hour)]['DEOK_MW']


train, test, val  = load_dataset('dataset/DEOK_hourly.csv')
print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')
naive = NaiveImplementation(train)

# obtain MAE from naive implementation on test set
mae_naive = 0
for index, row in test.iterrows():
    prediction = naive.predict(row['Datetime'])
    mae_naive += abs(row['DEOK_MW'] - prediction)

print(f'MAE Naive {mae_naive:.2f}')

train_normalized = normalize_datetime(train)
train = keras.utils.timeseries_dataset_from_array(train_normalized, train['DEOK_MW'], sequence_length=24, sequence_stride=1, batch_size=32)
val_normalized = normalize_datetime(val)
val = keras.utils.timeseries_dataset_from_array(val_normalized, val['DEOK_MW'], sequence_length=24, sequence_stride=1, batch_size=32)
test_normalized = normalize_datetime(test)
test = keras.utils.timeseries_dataset_from_array(test_normalized, test['DEOK_MW'], sequence_length=24, sequence_stride=1, batch_size=32)

# RNN approach
model = keras.Sequential([
    keras.layers.LSTM(32, input_shape=(24, 6)),
    keras.layers.Dense(1)
])


# print(model.summary())
model.compile(loss='mae', optimizer='adam')
history = model.fit(train, epochs=3, validation_data=val)

# predict on test and calc mae
mae = 0
for batch in test:
    inputs, targets = batch
    prediction = model.predict(inputs)
    for i in range(len(targets)):
        mae += abs(targets[i] - prediction[i][0])
print(20 * '-')
print(f'MAE RNN {mae:.2f}')
print(f'MAE Naive {mae_naive:.2f}')
print(20 * '-')

