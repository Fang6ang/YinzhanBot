from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import time, logging

url = 'https://pkuhelper.pku.edu.cn/hole/'
url_api = 'https://pkuhelper.pku.edu.cn/services/pkuhole/api.php'
url_img = 'https://pkuhelper.pku.edu.cn/services/pkuhole/images/'

text_valid = '为保障树洞信息安全, 请前往树洞网页版进行人机验证, pkuhelper.pku.edu.cn/hole'


def fix_captcha(_url, _token):
    logging.warning('Fixing recaptcha. This may take a minute...')
    option = webdriver.ChromeOptions()
    option.add_experimental_option('excludeSwitches', ['enable-automation'])

    option_head = Options()
    option_head.add_argument('--headless')
    option_head.add_argument('--disable-gpu')

    bro = webdriver.Chrome(r'./chromedriver', options=option, chrome_options=option_head)
    bro.get(_url)
    login_bttn = bro.find_element_by_xpath('//*[@id="root"]/div[4]/div[2]/div/p/a/span')
    login_bttn.click()

    token_input = bro.find_element_by_xpath('//*[@id="pkuhelper_login_popup_anchor"]/div/div[2]/p[6]/input')
    token_input.send_keys(_token)

    load_bttn = bro.find_element_by_xpath('//*[@id="pkuhelper_login_popup_anchor"]/div/div[2]/p[6]/button')
    load_bttn.click()

    time.sleep(5)
    bro.quit()


def valid_test(sess, params, num=-1):
    cnt, text = 0, ''
    while text_valid not in text:
        params['action'] = 'getlist'
        response = sess.get(url=url_api, params=params).json()
        text = ''.join([p['text'] for p in response['data']])
        cnt += 1
        if cnt == num:
            break

    if 'action' in params:
        del params['action']
    return cnt != num or num == -1


