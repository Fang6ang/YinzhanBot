import json.decoder
import time

import requests, logging, numpy as np
from utils import fix_captcha
from utils import url, url_api, text_valid


class HoleCrawler:
    def __init__(self, _sess: requests.Session, _params: dict):
        self.sess = _sess
        self.param = _params

        self.nums, self.first_id = 0, -1
        self.holes, self.replies = [], {}

    def print_holes(self, num=0):
        num -= 1
        for idx, _hole in enumerate(self.holes):
            print(_hole['pid'] + ': \n' + _hole['text'])
            _rep = self.replies[_hole['pid']]
            for cont in _rep:
                print('\t', cont)
            print()
            if idx == num:
                break

    def update_holes(self, p=1, no_comments=False):
        self.param['action'] = 'getlist'
        self.param['p'] = p
        while True:
            try:
                cashe_hole = requests.get(url_api, params=self.param).json()['data']
                if len(cashe_hole) < 2:
                    logging.warning('Get invalid page, returning')
                    return True
                if not (cashe_hole[0]['text'] == text_valid and cashe_hole[1]['text'] == text_valid):
                    break
                fix_captcha(url, self.param['user_token'])
            except json.decoder.JSONDecodeError:
                logging.warning('Server 503, wait for 10 secs.')
                time.sleep(10)

        del self.param['p']

        _hole_list, _rep_list, cnt = [], [], 0
        for _hole in cashe_hole:
            _hole_list.append(_hole)
            cnt += 1

        if not no_comments:
            _rep_dict = self.get_reply(_hole_list)
            self.replies.update(_rep_dict)
        self.holes = self.holes + _hole_list
        logging.info(f'Update done, {cnt} new holes updated.')

        return

    def get_reply(self, holes):
        rep = {}
        self.param['action'] = 'getcomment'
        self.param['pid'] = ''
        for _hole in holes:
            rep[_hole['pid']] = []
            if _hole['reply'] == '0':
                continue
            self.param['pid'] = _hole['pid']
            while True:
                try:
                    replies = requests.post(url=url_api, params=self.param).json()['data']
                    time.sleep(1)
                    break
                except json.decoder.JSONDecodeError:
                    logging.warning('Server 503, wait for 60 secs.')
                    time.sleep(60)

            replies = [dic['text'] for dic in replies]
            rep[_hole['pid']] = replies

        del self.param['action'], self.param['pid']
        return rep
