#!/usr/bin/env python3 

import pandas as pd 
import numbers
import numpy as np
from emoji import demojize 
from loguru import logger
from argparse import ArgumentParser as AP
import os
from collections import defaultdict
from datetime import datetime, timezone 
from tqdm import tqdm 
from joblib import Parallel, delayed
from glob import glob 
from itertools import chain


START = datetime(2020, 1, 1, tzinfo=timezone.utc)


def safe(fn):
    def wraps(arg):
        if arg is None or (isinstance(arg, numbers.Number) 
                           and np.isnan(arg)):
            return None
        try:
            return fn(arg)
        except Exception as e:
            logger.warning(e)
            logger.warning(arg)
            return None
    return wraps


@safe
def get_lat(g):
    return g['coordinates'][0]

@safe
def get_lon(g):
    return g['coordinates'][1]

@safe
def get_days(c):
    dt = c.to_pydatetime() 
    if isinstance(dt, np.ndarray):
        dt = dt[0] # some tweets have multiple timestamps, odd
    return (dt - START).days

def conv_hashtags(h):
    if h is None:
        return []
    else:
        return [hh['text'] for hh in h]

@safe
def get_place_type(p):
    return p['place_type']

@safe
def get_place_name(p):
    return p['name']

@safe
def get_country(p):
    return p['country_code']


def convert(in_file, out_file, blocksize):
    try:
        num_lines = sum(1 for line in open(in_file, 'r'))
        dfs = pd.read_json(in_file, lines=True, chunksize=blocksize)
        if blocksize is None:
            dfs = [dfs]
            blocksize = num_lines

        pbar = tqdm(desc=in_file)
        for i, df in enumerate(dfs):
            # logger.info('full text')
            try:
                df['full_text'] = df['full_text'].apply(
                        lambda x: demojize(x).replace('::', ': :')
                    )
            except KeyError:
                logger.error('full_text is missing! skipping')
                return

            # logger.info('geo tag')
            df['lat'] = df['geo'].apply(get_lat)
            df['lon'] = df['geo'].apply(get_lon)

            # logger.info('timestamp')
            df['date'] = df['created_at'].apply(get_days)

            # logger.info('hashtag')
            df['hashtags'] = df['entities'].apply(lambda x: x['hashtags'])
            df['hashtags'] = df['hashtags'].apply(conv_hashtags)

            # logger.info('location')
            df['place_type'] = df['place'].apply(get_place_type)
            df['place_name'] = df['place'].apply(get_place_name)
            df['country'] = df['place'].apply(get_country)

            df.rename(columns={'favorite_count': 'favorites',
                               'retweet_count': 'retweets'},
                      inplace=True)

            df = df[['full_text', 'lat', 'lon', 'date', 'hashtags', 
                     'place_type', 'place_name', 'country',
                     'favorites', 'retweets', 'lang']]

            # logger.info(f'Writing to {out_file}.{i}')
            df.to_json(f'{out_file}.{i}', lines=True, orient='records')
            pbar.update(blocksize)

    except Exception as e:
        logger.error(f'{type(e)}: {e}')
        return



@logger.catch 
def main():
    p = AP()
    p.add_argument('-i', required=True, nargs='+')
    p.add_argument('-o', required=True)
    p.add_argument('--skip_exists', action='store_true')
    p.add_argument('--proc', type=int, default=1) 
    p.add_argument('--block', type=int)
    args = p.parse_args() 

    os.makedirs(args.o, exist_ok=True)

    args.i = list(chain.from_iterable(glob(i) for i in args.i))

    fn_args = []
    for infile in args.i:
        f = infile.split('/')[-1]
        outfile = os.path.join(args.o, f)
        if args.skip_exists and os.path.exists(outfile):
            continue
        fn_args.append((infile, outfile))

    Parallel(n_jobs=args.proc, verbose=10)(
            delayed(convert)(infile, outfile, args.block) for infile, outfile in fn_args
        )



if __name__ == '__main__':
    main()
