from nose.tools import eq_

import codd

def test_tokens():
    example = (
        '     \n\t' # whitespace
        'this_is_a_word_123_with_numbers ' # word
        '123908471234987' # number
        '"double quotes\'<>?&"' # string constant
        "'single quotes \"<>?&'" # string constant
        '+-*/(),=.' # basic symbols
        '?word' # variable
        '$word' # variable - American style
        '<' # less than
        '<=' # less than or equal
        '>' # greater than
        '>=' # greater than or equal
        '!' # not
        '!=' # not equal
    )

    tokens = list(codd.Tokens(example))

    expected = [
        'this_is_a_word_123_with_numbers',
        '123908471234987',
        '"double quotes\'<>?&"',
        '\'single quotes "<>?&\'',
        '+', '-', '*', '/', '(', ')', ',', '=', '.',
        '?word', '$word',
        '<', '<=',
        '>', '>=',
        '!', '!=',
    ]
    eq_(tokens, expected)
