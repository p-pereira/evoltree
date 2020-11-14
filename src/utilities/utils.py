# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:49:04 2020

@author: pedro
"""
import re

def roundAllNums(rules):
    allNums = re.findall("\d+\.\d+", rules)
    rules2 = rules
    for a in allNums:
        if float(a) == 1.0:
            a2 = "1"
            rules2 = re.sub(a, a2, rules2)
        
    return(rules2)

# Function to remove leading zeros from phenotypes
def remove_leading_zeros(phenotype):
    return re.sub(r'([^\.^\d])0+(\d)', r'\1\2', phenotype)

def create_lamarck_mapper():
    # TODO create python mapper script
    a=0
    