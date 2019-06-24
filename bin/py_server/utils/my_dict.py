# -*- coding: utf-8 -*-
import numpy as np


class MyDict:
    
    def __init__(self):
        """
        constructor to construct a special dict which value is also a dict. 
        Struct of MyDict like as {key, {sub-key, [item1, item2, ...]}}.
        :param model_path: the path of model
        :param model_name: the name of model
        :param model_type: the type of model. optional value 'svm' (svm in sklearn),
                 and 'nn' (neural networks)
        """
        self.my_dict = dict()


    def add(self, key, sub_key, value):
        '''
        add an key-value item to MyDict.
        :param key: primary key item of MyDict
        :param sub_key: sub key item of MyDict
        :param value: list, value item of MyDict
        '''
        # check whether value is a list type
        if not (isinstance(value, list)):
            return 'Parameter \'value\' is not a list type.'
        # add an item to dict
        if key in self.my_dict.keys():
            if sub_key in self.my_dict.get(key).keys():
                return 'Error: sub-item ',sub_key,' has existed ...'
            else:
                self.my_dict.get(key).setdefault(sub_key,value)
                return True
        else:
            self.my_dict.setdefault(key, {sub_key:value})
            return True
        
    def get(self, key, sub_key):
        '''
        get value of MyDict through key and sub-key
        :param key: main key in MyDict
        :param sub_key: sub key in MyDict
        return a list with length of 2
        '''
        return self.my_dict.get(key).get(sub_key)
    def get_sub_item(self, key):
        '''
        get {sub-key, value} of MyDict through key
        :param key: main key in MyDict
        '''
        return self.my_dict.get(key)
        
    def keys(self):
        '''
        get all main keys in MyDict.
        '''
        return self.my_dict.keys()
    
    def sub_keys(self):
        '''
        get all sub-keys.
        '''
        keys = list()
        for key in self.my_dict.keys():
            keys += list(self.my_dict.get(key).keys())
        return keys
    
    def key_sub_keys(self, key):
        '''
        get all sub-keys under key.
        :param key: main key in MyDict
        '''
        if key in self.my_dict.keys():
            return self.my_dict.get(key).keys()
        else:
            return None
        
if __name__ == '__main__':
    my_dict = MyDict()
    for i in range(10):
        my_dict.add('a',['a'+str(i+1),i+1,i*i])
    print(my_dict.get('a','a3'))
    print(my_dict.keys())
    print(my_dict.sub_keys())
    print(my_dict.key_sub_keys('a'))
    a = np.asarray([[1,2,3]])
    b = np.asarray([[4,5,6]])
    print(np.argmax(a+b))
    c = ['a','b','.a','.b']
    c.remove('.a')
    print(c)