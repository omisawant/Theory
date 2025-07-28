'''----------------------------------------------------------------
--KEYWORDS--
- reserved words with special meaning to the Python interpretor'''

from sys import argv
import functools

if 5==5 and 6>5:
    pass
elif False:
    pass
elif True:
    pass
else:
    pass

'''----------------------------------------------------------------
--IDENTIFIERS--
- names for variables, functions, classes assigned by programmer
- some identifiers are reserved
- cannot be KEYWORDS'''

count = 5
hero_name = 'Batman'
fruits = ['apple', 'banana']

class Ghost:
    pass

def say_boo():
    print('boo!')

'''----------------------------------------------------------------
--LITERALS--
- raw values or data given in variables or constants
- types -> string, integer, float, boolean, other
'''

name = 'Albert'
light_speed = 299792458
pi_value = 3.14
fact = True
items_list = ['phone', 'computer']
scores = [78, 83]

'''----------------------------------------------------------------
--OPERATORS--
- types -> assignment, arithmetic, comparison, logical, bitwise, identity, membership
- operator precendence -> https://docs.python.org/3/reference/expressions.html#operator-precedence
'''

total_count = 2 + 3
total_count += 6 # total_count = 11
phrase = 'Wonder' + 'Woman' # concatanation

'''----------------------------------------------------------------
--DELIMITERS--
- delimiter list -> https://docs.python.org/2.0/ref/delimiters.html
'''

soda = ('coke', 'pepsi')
fruits = ['apple', 'banana']
fruit_cost = {'apple' : 5, 'banana' : 6}

def example():
    pass

'''----------------------------------------------------------------
--COMMENTS--
- can be used to explain code and make it more readable
'''

# Single line comment
'''Multi-line comment'''
"""Multi-line comment"""

'''----------------------------------------------------------------
--WHITESPACES--
- types -> spaces, tabs
- every level of indentation is either 1 tab or 4 spaces
'''