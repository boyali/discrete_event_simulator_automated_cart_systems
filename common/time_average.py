import numpy as np


# sars = pd.read_csv('sars_history_constrained_epoch_0.csv', sep='\t')


def time_average(df, column=None):
    unique_cart = df.cartID.unique()
    time_series = {}
    time_average0 = {}
    # column = 'occupancy'

    cartID0 = unique_cart[0]
    time0 = None
    column0 = None

    for cartID in unique_cart:
        time_series[cartID] = {}
        # time_series[cartID]['sim_time'] = []
        time_series[cartID]['time_delta'] = []
        time_series[cartID][column] = []

    for i, row in df.iterrows():
        # time_series[row.cartID]['sim_time'].append(row.sim_time)
        # time_series[row.cartID][column].append(row[column])

        if i == 0 or row.cartID != cartID0:
            cartID0 = row.cartID
            time0 = row.sim_time
            column0 = row[column]

        if i > 0:
            if row[column] != column0:
                time_series[row.cartID]['time_delta'].append(row.sim_time - time0)
                time_series[row.cartID][column].append(column0)

            cartID0 = row.cartID
            time0 = row.sim_time
            column0 = row[column]

    for cartID in unique_cart:
        time_axis = np.array(time_series[cartID]['time_delta'])
        magnitute = time_series[cartID][column]

        time_average0[cartID] = (time_axis * magnitute).sum() / time_axis.sum()

    return (time_series, time_average0)

# averaged = time_average(sars, column='occupancy')
#
# print(averaged[1])
