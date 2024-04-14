import random
import os
import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv("all_files_label.csv")


for i in range(5):
    m_train, m_test = train_test_split(
        df[df['label'] == 1], test_size=0.28, random_state=random.randint(0, 1000))
    g_train, g_test = train_test_split(
        df[df['label'] == 0], test_size=0.9, random_state=random.randint(0, 1000))

    train = pd.concat([m_train, g_train])
    test = pd.concat([m_test, g_test.iloc[:m_test.shape[0]*3]])
    train.to_csv('./train_{}.csv'.format(i+1), index=False)
    test.to_csv('./test_{}.csv'.format(i+1), index=False)
