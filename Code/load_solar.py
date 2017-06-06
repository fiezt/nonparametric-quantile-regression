import os
import pandas as pd
import numpy as np
import multiprocessing
from natsort import natsorted
import functools


def load_data(solar_dir):
    """Load the initial data into dataframes.
    
    :param solar_dir: Directory to the solar data.
    :return benchmarks, predictors, train, solutions: List ordered by the task 
    number of dataframes containing the benchmark results, the predictor 
    variables, the training labels, and the solution respectively.
    """
    
    benchmarks = {}
    predictors = {}
    train = {}
    solutions = {}
    
    # Walk through directories to get the different files.
    for root, dirs, files in os.walk(solar_dir):
        for fname in files:
            if '.csv' in fname:
                if 'benchmark' in fname:
                    benchmarks[fname.split('.')[0]] = pd.read_csv(os.path.join(root, fname))
                elif 'predictors' in fname:
                    predictors[fname.split('.')[0]] = pd.read_csv(os.path.join(root, fname))
                elif 'train' in fname:
                    train[fname.split('.')[0]] = pd.read_csv(os.path.join(root, fname))
                elif 'Solution' in fname:
                    solutions[fname.split('.')[0]] = pd.read_csv(os.path.join(root, fname))   

    # Number of sets provided.
    N = len(train)
    
    # Testing data.
    benchmarks = natsorted(benchmarks.items(), key=lambda x: x[0])
    benchmarks = [item[1] for item in benchmarks]

    # Given features.
    predictors = natsorted(predictors.items(), key=lambda x: x[0])
    predictors = [item[1] for item in predictors]

    # Training labels.
    train = natsorted(train.items(), key=lambda x: x[0])
    train = [item[1] for item in train]

    # Solution to the test set.
    solutions = natsorted(solutions.items(), key=lambda x: x[0])
    solutions = [item[1] for item in solutions]

    for i in range(N):
        predictors[i]['TIMESTAMP'] =  pd.to_datetime(predictors[i]['TIMESTAMP'], format='%Y%m%d %H:%M')
        benchmarks[i]['TIMESTAMP'] =  pd.to_datetime(benchmarks[i]['TIMESTAMP'], format='%Y%m%d %H:%M')
        train[i]['TIMESTAMP'] =  pd.to_datetime(train[i]['TIMESTAMP'], format='%Y%m%d %H:%M')
        
    return benchmarks, predictors, train, solutions


def featurize_data(predictors, train, solar_dir, i):
    """Create the featurized data from the predictors and add training labels.
    
    :param predictors: List of dataframes containing the predictor information 
    for each task.
    :param train: List of dataframes containing the training label for each task.
    :param solar_dir: Directory to save the featurized solar data to.
    :param i: Task number to create the featurized data for.
    
    :return: Write the featurized data to files.
    """
    
    cols = ['ZONEID', 'TIMESTAMP', 'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6', 
            'VAR7', 'VAR8', 'VAR9', 'VAR10', 'VAR11', 'VAR12']
    
    if 'POWER' in predictors[i].columns:
        del predictors[i]['POWER']
    else:
        pass
    
    predictors[i].columns = cols

    # Wind speed magnitude.
    predictors[i]['VAR13'] = np.sqrt(predictors[i]['VAR6']**2 + predictors[i]['VAR7']**2)    

    # Total cloud cover x surface solar radiation.
    predictors[i]['VAR14'] = predictors[i]['VAR5'] * predictors[i]['VAR9']

    # Total cloud cover x relative humidity.
    predictors[i]['VAR15'] = predictors[i]['VAR5'] * predictors[i]['VAR4']

    # Freezing temperature flag. Note that freezing in kelvin is 273.15.
    predictors[i]['VAR16'] = predictors[i]['VAR8'] < 273.15 
    predictors[i]['VAR16'] = predictors[i]['VAR16'].astype(int)

    # Precipitation flag. Total precipitation > 0.
    predictors[i]['VAR17'] = predictors[i]['VAR12'] > 0 
    predictors[i]['VAR17'] = predictors[i]['VAR17'].astype(int)

    # Snow flag. Freezing flag times precipitation flag.
    predictors[i]['VAR18'] = predictors[i].apply(lambda x: x['VAR16'] * x['VAR17'], axis=1)

    # Differential surface pressure.
    predictors[i]['VAR19'] = predictors[i]['VAR3'] - predictors[i]['VAR3'].shift(1)

    # Differential total cloud cover.
    predictors[i]['VAR20'] = predictors[i]['VAR5'] - predictors[i]['VAR5'].shift(1)

    # Converting the temperature in kelvin to celsius.
    predictors[i]['CELSIUS_TEMP'] = predictors[i]['VAR8'] - 273.15

    # Wind chill index. 
    predictors[i]['VAR_21'] = (10*np.sqrt(predictors[i]['VAR13']) - predictors[i]['VAR13'] + 10.5) \
                              * (33 - predictors[i]['CELSIUS_TEMP'])

    # Solar model temperature.
    predictors[i]['VAR_22'] = predictors[i]['CELSIUS_TEMP'] + np.exp(-3.473 - 0.0594*predictors[i]['VAR13'])

    # Maximum solar power output for each day and each zone.
    predictors[i]['VAR23'] = pd.Series(np.nan, index=predictors[i].index)    
    predictors[i]['VAR24'] = pd.Series(np.nan, index=predictors[i].index)    
    predictors[i]['VAR25'] = pd.Series(np.nan, index=predictors[i].index)    

    predictors[i]['DATE'] = predictors[i]['TIMESTAMP'].dt.date    
    train[i]['DATE'] = train[i]['TIMESTAMP'].dt.date
    max_power = train[i].groupby(['ZONEID','DATE']).max()

    for date in train[i]['DATE'].unique():
        predictors[i].loc[predictors[i]['DATE'] == date, 'VAR23'] = max_power.ix[1,date]['POWER']
        predictors[i].loc[predictors[i]['DATE'] == date, 'VAR24'] = max_power.ix[2,date]['POWER']
        predictors[i].loc[predictors[i]['DATE'] == date, 'VAR25'] = max_power.ix[3,date]['POWER']

    predictors[i]['VAR26'] = np.cos((predictors[i]['TIMESTAMP'].dt.dayofyear * 2 * np.pi)/365.)
    predictors[i]['VAR27'] = np.sin((predictors[i]['TIMESTAMP'].dt.dayofyear * 2 * np.pi)/365.)

    predictors[i]['VAR28'] = np.cos((predictors[i]['TIMESTAMP'].dt.hour * 2 * np.pi)/24.)
    predictors[i]['VAR29'] = np.sin((predictors[i]['TIMESTAMP'].dt.hour * 2 * np.pi)/24.)

    # Getting the power for each timestamp available in the training data.
    train[i] = train[i][['ZONEID', 'TIMESTAMP', 'POWER']]
    predictors[i] = pd.merge(predictors[i], train[i], how='left', on=['TIMESTAMP', 'ZONEID'])

    del predictors[i]['DATE']
    del predictors[i]['CELSIUS_TEMP']

    predictors[i].to_csv(os.path.join(solar_dir + '/Task ' + str(i+1), 'features' + str(i+1) + '.csv'), 
                         sep=',', index=False)


def main():
    """Load the data, create and write the featurized data to files."""
    
    curr_dir = os.getcwd()
    data_dir = curr_dir + '/../Data'
    solar_dir = data_dir + '/Solar'
    wind_dir = data_dir + '/Wind'
    price_dir = data_dir + '/Price'
    load_dir = data_dir + '/Load'
    
    benchmarks, predictors, train, solutions = load_data(solar_dir)
    
    pool = multiprocessing.Pool()

    func = functools.partial(featurize_data, predictors, train, solar_dir)
    pool.map(func, range(len(predictors)))   

    pool.close()
    pool.join() 


if __name__ == '__main__':

    main()

