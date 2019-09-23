import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer   # Imputer take care of missing values

# Import files project
import test_set

DOWLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

'''Computing the median of each attribute and store in housing_numbers'''

def replace_missing_value(housing):
    imputer = Imputer(strategy="median")
    housing_numbers = housing.drop("ocean_proximity", axis=1)   # removing text from dataset
    imputer.fit(housing_numbers)
    X = imputer.transform(housing_numbers)
    train_set = pd.DataFrame(X, columns=housing_numbers.columns)
    return train_set


if __name__ == "__main__":
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)

    # median = housing["total_bedrooms"].median()
    # housing["total_bedrooms"].fillna(median, inplace=True)
    # print(housing["total_bedrooms"])

    ''' Mapa de calor baseado no preço de casas na California '''
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing["population"]/100, label="population", figsize=(10, 7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

    ''' Finidng the correlation in other vars related about mean price '''
    correlation = housing.corr()
    # print(correlation["median_house_value"])

    ''' Dataset info '''
    # housing.info()
    # housing.hist(bins=50, figsize=(20, 15))

    ''' Habilitando os graficos '''
    # plt.show()

    # housing_with_id = housing.reset_index()
    # train_test, test_set = test_set.split_test_set_by_id(housing_with_id, 0.2, "index")

    train_test, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    # print("Train set: ", len(train_test), ". Test set: ", len(test_set))

    # Cleaning Training set
    '''Separate the predictors and the labels since we don't necessarily want
        to apply the same transformations to the predictors and the target values'''

    strat_train_set = train_test
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    # print("housing_labels: ", housing_labels)
    # print("Housing: ", housing)
  
    # Data Cleaning
    ''' Let's remove missing values in the data set '''
    housing_train_set = replace_missing_value(housing)
    print(housing_train_set)
