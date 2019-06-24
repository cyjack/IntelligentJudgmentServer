'''
Created on 2017年4月19日

@author: mage
'''
import sys
import os

def main_path():
    '''
    get the path of current script file. 
    Note: the path is file path which calls this method
    '''
    # script path
    path = sys.path[0]
    # if script, return path of script. 
    # Otherwise, return path of compiled script
    if os.path.isdir(path):
        return path
    elif os.path.isfile(path):
        return os.path.dirname(path)
    
