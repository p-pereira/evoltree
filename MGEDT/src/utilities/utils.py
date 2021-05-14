# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:49:04 2020

@author: pedro
"""
import re
from os import makedirs, path

def roundAllNums(rules):
    allNums = re.findall("\d+\.\d+", rules)
    rules2 = rules
    for a in allNums:
        if float(a) == 1.0:
            a2 = "1"
            rules2 = re.sub(a, a2, rules2)
        
    return(rules2)

# Removes leading zeros from phenotypes
def remove_leading_zeros(phenotype):
    return re.sub(r'([^\.^\d])0+(\d)', r'\1\2', phenotype)
# Automatically creates lammarck mapper file
def create_lamarck_mapper(params):
    # Create mapper folder based on experiment name
    mapper_dir = "./src/utilities/lamarck/{0}".format(params["EXPERIMENT_NAME"])
    makedirs(mapper_dir, exist_ok=True)
    # Read mapper base file (common to all mappers)
    mapper_base_file = "./src/utilities/lamarck/mapper_base"
    with open(mapper_base_file, 'r') as mf:
        content = mf.read()
    # New mapper file
    ## Create file
    init_filename = "{0}/__init__.py".format(mapper_dir)
    _ = open(init_filename, "w+").close()
    mapper_filename = "{0}/mapper.py".format(mapper_dir)
    mapper_file = open(mapper_filename, "w+")
    ## Write base code, common to all mappers
    mapper_file.write(content)
    ## Get data headers
    if params['DATASET_TRAIN'] != "":
        data_file = open(path.join('datasets', 
                                   params['DATASET_TRAIN']), 'r')
        headers = data_file.readline()[:-1] # ignore last character: '\n'.
        data_file.close()
        headers = headers.replace(params["DATASET_DELIMITER"] + params['TARGET'],
                                  "")
        headers_list = headers.split(params["DATASET_DELIMITER"])
    else:
        headers_list = list(params['X_train'].columns)
    ## Get last function number
    f_num = content.split("def p_")[-1].split('(')[0]
    ## Build functions and idx (attributes)
    funcs = "\n"
    idx = "\n"
    for n, header in enumerate(headers_list):
        funcs += 'def p_{0}(t) : "idx : {1}" ; t[0] = [gen_rnd({2}, {3})]\n'\
            .format(str(int(f_num) + n + 1), header, str(n), len(headers_list))
        idx += 't_{header} = "{header}"\n'.format(header = header)
    ## Build tokens
    tokens = "\ntokens = ('NUMBER', 'npwhere', 'x', "
    tokens += " ".join(["'{0}',".format(str(elem)) 
                          for elem in headers_list])
    tokens += ")\n"
    ## Write code to new mapper file
    mapper_file.write(funcs)
    mapper_file.write(tokens)
    mapper_file.write(idx)
    ## End of file
    mapper_file.write("\nfrom ply.lex import lex\n")
    mapper_file.write("from ply.yacc import yacc\n")
    mapper_file.write("lexer = lex()\n")
    mapper_file.write("parser = yacc()\n")
    mapper_file.close()
    
    # Return mapper
    mapper = "src.utilities.lamarck.{0}.mapper".format(params["EXPERIMENT_NAME"])
    return mapper