# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:49:04 2020

@author: pedro
"""
import re

def treat_num(num):
    try:
        n = int(num.lstrip('0'))
        return str(n)
    except:
        
        n = float(num.lstrip('0'))
        return str(n)

def treat_phenotype(phenotype):      
    allNums = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", 
                         phenotype)
    for index, num in enumerate(allNums):
        try: #only replaces numbers with leading zeros
            _ = eval(num)
        except:
            t_num = treat_num(num)
            phenotype = re.sub("\({num}\),".format(num=num), 
                               "({num2}),".format(num2=t_num), 
                               phenotype)
    return phenotype