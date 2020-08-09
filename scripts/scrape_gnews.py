#!/usr/bin/env python3 

import pandas as pd 
import json 
import numpy as np
from loguru import logger
import os
from argparse import ArgumentParser as AP
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from tqdm import tqdm, trange 
from pygooglenews import GoogleNews
from unidecode import unidecode
from joblib import Parallel, delayed


START = datetime(2020, 1, 1)
NOW = datetime.now()
DAY = timedelta(days=1)


class API:
    def __init__(self, terms, lang, country):
        self.terms = terms 
        self.client = GoogleNews(lang, country)

    @staticmethod
    def format(x):
        try:
            return unidecode(x['title'])
        except KeyError:
            return ''

    def __call__(self, from_, to_):
        results = []
        for q in self.terms:
            results += self.client.search(
                query=q, from_=from_, to_=to_
            )['entries']
        return list(set(API.format(r) for r in results))


@logger.catch 
def main():
    p = AP()
    p.add_argument('-o', required=True)
    p.add_argument('--rebuild', action='store_true')
    p.add_argument('--lang', default='es')
    p.add_argument('--country', default='MX')
    p.add_argument('--search_terms', nargs='+', 
                    default=['covid', 'coronavirus', 'pandemia', 'cubrebocas', 'tapabocas', 
                             'cuarentena', 'distanciamiento social', 'Quédate en Casa',
                             'sars cov-2'])
    p.add_argument('--test', action='store_true')
    args = p.parse_args() 

    os.makedirs(args.o, exist_ok=True)

    n_days = (NOW - START).days 
    if args.test:
        n_days = 5
    days_iter = np.random.permutation(n_days) 

    api = API(args.search_terms, args.lang, args.country)

    def worker(day):
        out_path = f'{args.o}/{day}.txt'
        if not args.rebuild:
            if os.path.exists(out_path):
                return
        from_ = START + (day * DAY)
        to_ = from_ + DAY 
        from_ = from_.strftime('%Y-%m-%d')
        to_ = to_.strftime('%Y-%m-%d')

        results = api(from_, to_)
        with open(out_path, 'w') as fjson:
            fjson.write('\n'.join(json.dumps(r) for r in results))
    
    Parallel(n_jobs=8, verbose=10, prefer='threads')(
        delayed(worker)(day) for day in days_iter
    )


if __name__ == '__main__':
    main()
