from utilities.utils import roundAllNums
import random
# Build the lexer

def getval(n):   
    if n not in ts: print(f"Undefined name '{n}'")
    return ts.get(n,0)

def p_error(t):
    print(f"Syntax error at '{t.value}', [{t.lexer.lineno}]")

def get_genome_from_dt_idf(phenotype):
    phenotype = roundAllNums(phenotype)
    return parser.parse(phenotype)

literals  = "+-/*=().[],<>"
t_ignore  = " \t\n'\""

t_npwhere = 'np.where'
t_x = 'x'

def t_NUMBER(t):
    r'\d'
    t.value = int(t.value)
    return t

def t_error(t):
    print(f"Illegal character '{t.value[0]}', [{t.lexer.lineno}]")
    t.lexer.skip(1)

def gen_rnd(chosen, productions):
  factor = 255 // productions
  return random.randint(0, factor) * productions + chosen

# symboltable : dictionary of names
ts = { }
# Grammar
def p_a(t) : "result : npwhere '(' x '[' idx ']' comparison value ',' node ',' node ')'"  \
                                            ; t[0] = [gen_rnd(0 ,1)] + t[5] + t[7] + t[8] + t[10] + t[12]
def p_b(t) : "node : result"                ; t[0] = [gen_rnd(0, 2)] + t[1]
def p_c(t) : "node : '(' leaf ')' "         ; t[0] = [gen_rnd(1, 2)] + t[2]
def p_d(t) : "leaf : NUMBER '.' digits"     ; t[0] = [gen_rnd(0, 2)] + t[3]
def p_e(t) : "leaf : NUMBER"                ; t[0] = [gen_rnd(1, 2)]
def p_f(t) : "digits : digits digit"        ; t[0] = [gen_rnd(0, 2)] + t[1] + t[2]
def p_g(t) : "digits : digit"               ; t[0] = [gen_rnd(1, 2)] + t[1]
def p_h(t) : "digit : NUMBER"               ; t[0] = [gen_rnd(t[1], 10)]
def p_i(t) : "comparison : '=' '='"         ; t[0] = [gen_rnd(0, 4)]
def p_j(t) : "comparison : '<'"             ; t[0] = [gen_rnd(1, 4)]
def p_k(t) : "comparison : '>'"             ; t[0] = [gen_rnd(2, 4)]
def p_l(t) : "comparison : '<' '='"         ; t[0] = [gen_rnd(3, 4)]
def p_m(t) : "value : digits '.' digits"    ; t[0] = [gen_rnd(0, 2)] + t[1] + t[3]
def p_n(t) : "value : digits"               ; t[0] = [gen_rnd(1, 2)] + t[1]

def p_o(t) : "idx : idapplication"          ; t[0] = [gen_rnd(0, 10)]
def p_p(t) : "idx : idoperator"             ; t[0] = [gen_rnd(1, 10)]
def p_q(t) : "idx : idaffmanager"           ; t[0] = [gen_rnd(2, 10)]
def p_r(t) : "idx : idbrowser"              ; t[0] = [gen_rnd(3, 10)]
def p_s(t) : "idx : idpartner"              ; t[0] = [gen_rnd(4, 10)]
def p_t(t) : "idx : idcampaign"             ; t[0] = [gen_rnd(5, 10)]
def p_u(t) : "idx : idverticaltype"         ; t[0] = [gen_rnd(6, 10)]
def p_v(t) : "idx : regioncontinent"        ; t[0] = [gen_rnd(7, 10)]
def p_w(t) : "idx : country_name"           ; t[0] = [gen_rnd(8, 10)]
def p_x(t) : "idx : accmanager"             ; t[0] = [gen_rnd(9, 10)]

tokens = ('NUMBER', 'npwhere', 'x', 'idapplication', 'idoperator', 
          'idaffmanager', 'idbrowser', 'idpartner', 'idcampaign', 
          'idverticaltype', 'regioncontinent', 'country_name', 
          'accmanager', )

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

from ply.lex import  lex
from ply.yacc import yacc
lexer = lex()
parser = yacc()
