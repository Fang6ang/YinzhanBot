import json

import crawler, pandas as pd
from crawler import HoleCrawler
import argparse, logging, requests, time

ARG = argparse.ArgumentParser()
ARG.add_argument('--inter', type=int, default=60, help='Interval gap(s) of fetching.')
ARG.add_argument('--log', type=str, default=None, help='Logging file path.')

ARG = ARG.parse_args()


# def update(_crawler: crawler.HoleCrawler, inter):
#     dic = dict()
#     col = ['text', 'type', 'timestamp', 'reply']
#     while True:
#         _crawler.update_holes()
#         if len(_crawler.holes) > 1000:
#             logging.info('More than 10 holes fetched, saving to ./raw_data/')
#             df = pd.DataFrame(_crawler.holes).set_index('pid').loc[:, col]
#             for pid in df.index:
#                 df.loc[pid, 'reply'] = _crawler.replies[pid]
#             df.to_csv('./raw_data/' + df.index[-1] + '-' + df.index[0] + '.csv')
#             _crawler.holes, _crawler.replies = [], {}
#         time.sleep(inter)


def fetch(_crawler: crawler.HoleCrawler, num_pages=100, no_comments=False):
    col = ['text', 'type', 'timestamp', 'reply']
    for cur_page in range(1, num_pages + 1):
        if _crawler.update_holes(p=cur_page, no_comments=no_comments) is not None:
            break
    df = pd.DataFrame(_crawler.holes).drop_duplicates(subset=['pid']).set_index('pid').loc[:, col]
    if not no_comments:
        for pid in df.index:
            df.loc[pid, 'reply'] = _crawler.replies[pid]
    # df.to_csv('./raw_data/' + df.index[-1] + '-' + df.index[0] + '.csv')
    df.to_csv('./raw_data/no_comments.csv')
    _crawler.holes, _crawler.replies = [], {}


if __name__ == '__main__':
    with open('../config/conf.json', 'r') as f:
        params = json.load(f)
    print(params)

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    if ARG.log is not None:
        logging.basicConfig(filename=ARG.log, level=logging.INFO, format=LOG_FORMAT)
    else:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    sess = requests.Session()
    newHole = HoleCrawler(sess, params)

    fetch(newHole, no_comments=True)


