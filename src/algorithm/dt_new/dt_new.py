# -----------------------------------------------------------------------------
# calc.py
#
# A simple calculator with variables -- all in one file.
# -----------------------------------------------------------------------------
import re

literals  = "+-/*=().[],<>"           ## a single char
t_ignore  = " \t\n'\""             
tokens = ('NUMBER', 'npwhere', 'idapplication', 'idoperator', 'idaffmanager', 
          'idbrowser', 'idpartner', 'idcampaign', 'idverticaltype', 
          'regioncontinent', 'country_name', 'accmanager', 'x', 'none', 
          'verylow', 'low', 'medium', 'high', )

t_npwhere = 'np.where'
t_x = 'x'
t_idapplication = 'idapplication'
t_idoperator = 'idoperator'
t_idaffmanager = 'idaffmanager'
t_idbrowser = 'idbrowser'
t_idpartner = 'idpartner'
t_idcampaign = 'idcampaign'
t_idverticaltype = 'idverticaltype'
t_regioncontinent = 'regioncontinent'
t_country_name = 'country_name'
t_accmanager = 'accmanager'
t_none = 'none'
t_verylow = 'verylow'
t_low = 'low'
t_medium = 'medium'
t_high = 'high'


def t_NUMBER(t):
    r'\d'
    t.value = int(t.value)
    return t

def t_error(t):
    print(f"Illegal character '{t.value[0]}', [{t.lexer.lineno}]")
    t.lexer.skip(1)

import random

# Comentar este e descomentar o outro que deve ser melhor para o GE
#gen_rnd = lambda x, y: x

def gen_rnd(chosen, productions):
  factor = 255 // productions
  return random.randint(0, factor) * productions + chosen


# Build the lexer
from ply.lex import  lex
lexer = lex()

# Parsing rules
"""
precedence = (
    ('left','+','-'),
    ('left','*','/'),
    ('right','UMINUS'),
    )
"""
# symboltable : dictionary of names
ts = { }

# Grammar
def p_a(t) : "result : npwhere '(' x '[' idx ']' comparison '(' value ')' ',' node ',' node ')'"  \
                                            ; t[0] = [gen_rnd(0 ,1)] + t[5] + t[7] + t[9] + t[12] + t[14]
def p_b(t) : "node : result"                ; t[0] = [gen_rnd(0, 2)] + t[1]
def p_c(t) : "node : leaf "                 ; t[0] = [gen_rnd(1, 2)] + t[1]
def p_d(t) : "leaf : none"                  ; t[0] = [gen_rnd(0, 5)]
def p_e(t) : "leaf : verylow"               ; t[0] = [gen_rnd(1, 5)]
def p_f(t) : "leaf : low"                   ; t[0] = [gen_rnd(2, 5)]
def p_g(t) : "leaf : medium"                ; t[0] = [gen_rnd(3, 5)]
def p_h(t) : "leaf : high"                  ; t[0] = [gen_rnd(4, 5)]
def p_i(t) : "digits : digits digit"        ; t[0] = [gen_rnd(0, 2)] + t[1] + t[2]
def p_j(t) : "digits : digit"               ; t[0] = [gen_rnd(1, 2)] + t[1]
def p_k(t) : "digit : NUMBER"               ; t[0] = [gen_rnd(t[1], 10)]
def p_l(t) : "comparison : '=' '='"         ; t[0] = [gen_rnd(0, 4)]
def p_m(t) : "comparison : '<'"             ; t[0] = [gen_rnd(1, 4)]
def p_n(t) : "comparison : '>'"             ; t[0] = [gen_rnd(2, 4)]
def p_o(t) : "comparison : '<' '='"         ; t[0] = [gen_rnd(3, 4)]
def p_p(t) : "idx : idapplication"          ; t[0] = [gen_rnd(0, 10)]
def p_q(t) : "idx : idoperator"             ; t[0] = [gen_rnd(1, 10)]
def p_r(t) : "idx : idaffmanager"           ; t[0] = [gen_rnd(2, 10)]
def p_s(t) : "idx : idbrowser"              ; t[0] = [gen_rnd(3, 10)]
def p_u(t) : "idx : idpartner"              ; t[0] = [gen_rnd(4, 10)]
def p_v(t) : "idx : idcampaign"             ; t[0] = [gen_rnd(5, 10)]
def p_w(t) : "idx : idverticaltype"         ; t[0] = [gen_rnd(6, 10)]
def p_x(t) : "idx : regioncontinent"        ; t[0] = [gen_rnd(7, 10)]
def p_y(t) : "idx : country_name"           ; t[0] = [gen_rnd(8, 10)]
def p_z(t) : "idx : accmanager"             ; t[0] = [gen_rnd(9, 10)]
def p_aa(t) : "value : digits '.' digits"   ; t[0] = [gen_rnd(0, 2)] + t[1] + t[3]
def p_ab(t) : "value : digits"              ; t[0] = [gen_rnd(1, 2)] + t[1]


"""
<result>        ::= np.where(x[<idx>] <comparison> <value>, <node>, <node>)
<node>          ::= <result> |  (<leaf>)
<leaf>          ::= 0 | 1 | 2 | 3 | 4
<digits>        ::= <digits><digit> | <digit>
<digit>         ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
<comparison>    ::= "==" | "<" | ">" | "<="
<idx>           ::= "'idapplication'" | "'idoperator'" | "'idaffmanager'" | 
                "'idbrowser'" | "'idpartner'" | "'idcampaign'" | 
                "'idverticaltype'" | "'regioncontinent'" | "'country_name'" | 
                "'accmanager'"
<value>         ::= <digits>.<digits> | <digits>

"""

def getval(n):   
    if n not in ts: print(f"Undefined name '{n}'")
    return ts.get(n,0)

def p_error(t):
    print(f"Syntax error at '{t.value}', [{t.lexer.lineno}]")

from ply.yacc import yacc
parser = yacc()

def get_genome_from_dt_idf(phenotype):
    phenotype = roundAllNums(phenotype)
    return parser.parse(phenotype)

def roundAllNums(rules):
    allNums = re.findall("\d+\.\d+", rules)
    rules2 = rules
    for a in allNums:
        if float(a) == 1.0:
            a2 = "1"
            rules2 = re.sub(a, a2, rules2)
        #a2 = str(round(float(a), 4))
    """
    allNums2 = re.findall("\d+", rules2)
    for a2 in allNums2:
        if a2 != '0':
            a3 = a2.lstrip('0')
            rules2 = re.sub(a2, a3, rules2)
    """
    return(rules2)
"""
def s():
    while True:
        try:             s = input('> ')   # 
        except EOFError as e: print(e); break
        print(get_genome_from_dt_idf(s))

s()
"""