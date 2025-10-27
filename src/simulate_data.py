import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from utils import ensure_dir


ensure_dir('data')


def gen_one_course(quarters=10, days_per_quarter=70, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    start = datetime(2020, 1, 1)
    for q in range(quarters):
        for d in range(days_per_quarter):
            date = start + timedelta(days=q * days_per_quarter + d)
            weekday = date.weekday()
            for hour in range(9, 17):
                base = 1.0
                if weekday >= 5:
                    base *= 0.4
                if 11 <= hour <= 13:
                    base *= 2.5
                days_to_deadline = random.randint(0, 20)
                if days_to_deadline < 3:
                    base *= 2.0

                # course popularity
                enroll = random.randint(50, 200)

                # arrivals: Poisson
                lam = base * (enroll / 100.0)
                arrivals = np.random.poisson(lam)

                rows.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'hour': hour,
                    'weekday': weekday,
                    'days_to_deadline': days_to_deadline,
                    'enroll': enroll,
                    'arrivals': int(arrivals)
                })
    df = pd.DataFrame(rows)
    return df


if __name__ == '__main__':
    df = gen_one_course(quarters=6, days_per_quarter=30)
    df.to_csv('data/queue_data.csv', index=False)
    print('Saved data/queue_data.csv with', len(df), 'rows')
