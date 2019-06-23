import requests
import re
import json
import time
import logging

from .callbacks import printJSON

URL = 'https://duckduckgo.com/'
HEADERS = {
    'dnt': '1',
    'accept-encoding': 'gzip, deflate, sdch, br',
    'x-requested-with': 'XMLHttpRequest',
    'accept-language': 'en-GB,en-US;q=0.8,en;q=0.6,ms;q=0.4',
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'referer': 'https://duckduckgo.com/',
    'authority': 'duckduckgo.com',
}


logger = logging.getLogger('duckduck-api')


def get_vqd(keywords):
    params = {'q': keywords}

    logger.debug("Hitting DuckDuckGo for Token")

    #   First make a request to above URL, and parse out the 'vqd'
    #   This is a special token, which should be used in the subsequent request
    res = requests.post(URL, data=params)
    vqd = re.search(r'vqd=([\d-]+)\&', res.text, re.M|re.I)

    if not vqd:
        raise ValueError("VQD Token Parsing Failed !")
    return vqd.group(1)


def search(keywords, *, callback=printJSON, logger=logger):
    print('Will start soon, be patient..')
    SLEEPTIME = 5.
    SLEEPITERATIONS = 10
    logger.debug("Obtained Token")

    params = (
        ('l', 'wt-wt'),
        ('o', 'json'),
        ('q', keywords),
        ('vqd', get_vqd(keywords)),
        ('f', ',,,'),
        ('p', '2'),
    )

    url = URL + "i.js"

    logger.debug("Hitting Url : %s", url)

    while True:
        while True:
            try:
                res = requests.get(url, headers=HEADERS, params=params)
                data = json.loads(res.text)
                break
            except ValueError as e:
                logger.debug("Hitting Url Failure - Sleep and Retry: %s", url)
                for _ in range(SLEEPITERATIONS):
                    time.sleep(SLEEPTIME / SLEEPITERATIONS)
                continue

        logger.debug("Hitting Url Success : %s", url)
        callback(data["results"])

        if "next" not in data:
            logger.debug("No Next Page - Exiting")
            return

        url = URL + data["next"]
