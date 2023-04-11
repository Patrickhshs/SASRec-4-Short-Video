import os
import numpy as np
import pandas as pd
from utility import to_pickled_df


if __name__ == '__main__':
    data_directory = 'data'
    sampled_sessions = pd.read_pickle(os.path.join(data_directory, 'sampled_sessions.df'))

    total_ids=sampled_sessions.session_id.unique()
    np.random.shuffle(total_ids)

    fractions = np.array([0.8, 0.2])
    # split into 2 parts
    train_ids, test_ids = np.array_split(
        total_ids, (fractions[:-1].cumsum() * len(total_ids)).astype(int))

    train_sessions=sampled_sessions[sampled_sessions['session_id'].isin(train_ids)]
    test_sessions=sampled_sessions[sampled_sessions['session_id'].isin(test_ids)]

    to_pickled_df(data_directory, sampled_train=train_sessions)
    to_pickled_df(data_directory, sampled_test=test_sessions)