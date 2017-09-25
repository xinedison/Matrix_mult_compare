# -*- coding: utf-8 -*-
'''
Created on 2015年5月8日

@author: kevin
'''
import logging
logger = logging.getLogger('ugsearch')
formatter = logging.Formatter('%(asctime)s-%(levelname)s:%(message)s')

def __initLogger__(logger, formatter):
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

__initLogger__(logger, formatter)

import time
class Timer(object):
    def __init__(self, fname):
        self.fname = fname

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        global logger
        logger.info('%s elapsed time: %f ms' % (self.fname, self.msecs))
