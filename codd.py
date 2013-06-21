"""
codd: named in honor of  the late great Edgar Codd
provides relational alegrbra for functional programs

"""

from collections import namedtuple
import functools
import itertools
import operator
from datetime import datetime
import string
from StringIO import StringIO

from xlocal import xlocal




# tables
class table(object):
  """
  A sequnce of homegnous objects.
  """
  def __init__(self, name, fields, sequence=()):
    self.schema = namedtuple(name, fields)
    self.sequence=sequence
    
  def __call__(self, sequence):
    """Returns a new table with the same definition as the current instance."""
    return success(table(self.schema.__name__, self.schema._fields, sequence))
    
  def __iter__(self):
    return iter(self.sequence)
    
  def __repr__(self):
    return repr(self.sequence)
  
# functional helpers

def success(value):
  return value, None
  
def failure(reason):
  return None, reason
 


def value(mval):
  return mval[0]
  
def error(mval):
  return mval[1]

def bind(mval, mfunc):
  value, error = mval
  if error:
    # if there was an error don't invoke the rest of the chain
    return mval
  else:
    return mfunc(value)

def chain(*fns):
  """
  Return a composite function of all the funcions:
  >>> def fn1(x):
  ...  return x+1
  
  def fn2(x):
  ...   return x*2
  
  >>> fn3 = chain(fn1,fn2)
  >>> fn3(3)
  8
  """
  def chain_(val):
    for f in fns:
      val = f(val)
    return val
    
  return chain_  

def flatten(sequences):
  """
  Turn a sequence of sequences into a single sequence
  [[1,2,3], [4,5,6]] -> [1,2,3,4,5,6]
  """
  return success([
    item
    for sequence in sequences
    for item in sequence
  ])


def pipe(val, *fns):

  m_val = success(val)
  for i,f in enumerate(fns):
    try:
      m_val = bind(m_val, f)
    except Exception, e:
      raise
      # this should never happen
      m_val = failure("{} errored at step {}:  {}".format(f.__name__, i, e))
  return m_val


def cache(func, __marker__={}):
  _cache_ = {}
  def _(*args, **kw):
    key = (args, sorted(kw.items()))
    result = _cache_.get(key, __marker__)
    if result is  __marker__:
      _cache_[key] = result = func(*args, **kw)
    return result

  _.__name__ = "cached({0})".format(func.__name__)
  return _


def select(*columns):

  cols = [
    parse(col)
    for col in columns
  ]
  
  fields = [
    field if type(col_exp) == operator.attrgetter else 'col%d' % index 
    for index, field, col_exp in zip(itertools.count(), columns, cols)
  ]
  

  def _select(seq):
    schema = getattr(seq,'schema',None)
    if schema:
      name = schema.__name__
    else:
      name = "row"

    Record = namedtuple(name, fields)
    
    return success([
      Record(*[col(item) for col in cols])
      for item in seq
    ])
  return _select
  

def join(right, **conditions):
  def cond(lhs, rhs, conditions):
    res = [
      getattr(lhs,left_attr, None) == getattr(rhs, right_attr, None)
      for (left_attr, right_attr) in conditions.items()    
    ]
    return all(res)
  
  err = error(right)
  if err:
    return lambda left: right
  else:
    right = value(right)
    def _join(left):
      # optimization hint, lots of copying going on here
      # could be faster if the resulting table held
      # pointers to the original lhs and rhs record
      Results = table('results', left.schema._fields + right.schema._fields)
      schema = Results.schema
      
      l_attrs = operator.attrgetter(*left.schema._fields)
      r_attrs = operator.attrgetter(*right.schema._fields)
      return Results([
        schema(*(l_attrs(lhs) + r_attrs(rhs)))
        for lhs in left
        for rhs in right
        if cond(lhs, rhs, conditions)
      ])
    return _join

def when(**context):
  """
  Returns a decorator used to create a chain of pattern matched/gaurded
  functions. Useful for overiding functions based on input types.
  
  Example:
  >>> my_func = when()
  >>> @my_func()
  ... def default():
  ...   return "default"
  >>> @my_func("_ == 'bob'")
  ... def welcome_bob(name):
  ...   return "we hate you"
  >>> @my_func("_ != 'bob'")
  ... def welcome(name):
  ...   return "nice to meet you"
  
  
  """
  
  chain = []
  
  def invoke(*args):
    for gaurd, func in chain:        
      if gaurd(args):
        return func(*args)
    
    raise ValueError("No match for %s" % str(args))
    
  def _when(*conditions):
    conditions = [
      compile(cond,'when', 'eval')
      for cond in conditions
    ]
    def gaurd(args):
      if len(args) != len(conditions):
        return False
        
      for arg, cond in zip(args, conditions):
        locals = context.copy()
        locals['_'] = arg
        try:
          if not eval(cond,None, locals):
            return False
        except:
          return False
      return True
    
    def collect(func):
      chain.append((gaurd, func))
      return func
    
    return collect
      
  _when.invoke = invoke
  return _when

def where(clause, **options):
  boolean_op = parse(clause, **options)
  
  def _where(seq, ctx):
    return success([
      item
      for item in seq
      if boolean_op(item, ctx)
    ])
    
  return _where
    
def unit(func):
  "Invoke a function returning (success, error)"

  def _(value):
    try:
      return success(func(value))
    except Exception, e:
      return failure(e)
  return _
    

def each(method):
  fmap = functools.partial(map, method)
  def _each(sec):
    return success(fmap(sec))

  return _each


def parallel(*fns):
  """
  Given a list of functions return a function that takes a single argument.
  The argument will be passed to each of the functions there return values
  will be flattened
  """
  
  def _(val):
    return itertools.chain.from_iterable( fn(val) for fn in fns)
    
  _.__name__ = "parallel({}) ".format((',').join([f.__name__ for f in fns]))
  return _

## parsing routines #######

COMPARISON_OPS = {
  '<' : operator.lt,
  '<=' :operator.le,
  '='  : operator.eq,
  '!=' : operator.ne,
  '>=' : operator.ge,
  '>'  : operator.gt,
}

ADDITIVE_OPS = {
  '+'  : operator.add,
  '-'  : operator.sub,
}

MULTIPLICATIVE_OPS ={
  '*'  : operator.mul,
  '/'  : operator.div
}
SYMBOLS = '+-*/(),='

parse_options = xlocal()

def get_attr(name):
  return lambda row, ctx: getattr(row,name,None)

def parse(statement, get_value=get_attr):
  with parse_options(get_value=get_value):
    tokens = list(Tokens(statement))
    exp = and_exp(tokens)
    return exp

def and_exp(tokens):

  def and_(a, b):
    def _and(row, ctx):
      return a(row,ctx) and b(row,ctx)
    return _and
    
  lhs = or_exp(tokens) 
  while len(tokens) and tokens[0] == 'and':
    tokens.pop(0)
    lhs = and_(lhs, or_exp(tokens))
  return lhs
  
def or_exp(tokens):
  def or_(a, b):
    def _or(row,ctx):
      return a(row,ctx) or b(row,ctx)
    return _or
    
  lhs = comparison_exp(tokens)
  while len(tokens) and tokens[0] == 'or':
    tokens.pop(0)
    lhs = or_(lhs, comparisson_exp(tokens))
  return lhs
  
def comparison_exp(tokens):
  lhs = additive_exp(tokens)

  if len(tokens) and tokens[0] in COMPARISON_OPS:
    op = COMPARISON_OPS[tokens.pop(0)]
    rhs =  additive_exp(tokens)
    def comparison(row,ctx):
      return op(lhs(row,ctx), rhs(row,ctx))
    return comparison
  else:
    return lhs

def additive_exp(tokens):
  lhs = multiplicative_exp(tokens)
  while tokens:
    op = ADDITIVE_OPS.get(tokens[0])
    if op:
      tokens.pop(0)
      rhs =  multiplicative_exp(tokens)
      def additive(row, ctx, lhs=lhs): # bind lhs for each return
        return op(lhs(row,ctx), rhs(row,ctx))
      lhs = additive
    else:
      break
  return lhs

def multiplicative_exp(tokens):
  lhs = unary_exp(tokens)
  while len(tokens):
    op = MULTIPLICATIVE_OPS.get(tokens[0])
    if op:
      tokens.pop(0)
      rhs = unary_exp(tokens)
      def multiplicative(row, ctx, lhs=lhs):
        return op(lhs(row,ctx), rhs(row,ctx))
      lhs = multiplicative
    else:
      break
  return lhs

def unary_exp(tokens):
  assert len(tokens)
  
  if tokens[0] == '-':
    tokens.pop(0)
    value = value_exp(tokens)
    return lambda row, ctx: operator.neg(value(row, ctx))
  elif tokens[0] == 'not':
    tokens.pop(0)
    value = value_exp(tokens)
    return lambda row, ctx: operator.not_(value(row, ctx))
  elif tokens[0] == '+':
    tokens.pop(0)
    
  return value_exp(tokens)

def value_exp(tokens):
  """
  Returns a function that will return a value for the given token
  """
  token = tokens.pop(0)
  
  if token.startswith('$'):
    key = token[1:]
    try:
      key = int(key)
    except ValueError:
      pass
    return operator.itemgetter(key)


  if token.startswith('?'):
    pos = int(token[1:])
    return lambda row, ctx: ctx.get('params',[0])[pos]

  elif token[0] in string.digits:
    return lambda row, ctx: int(token)
  elif token.startswith('"'):
    return lambda row, ctx: token[1:-1]
  elif token == '(':
    return group_exp(tokens)
  elif token in SYMBOLS: 
    return lambda row, ctx: token
  else:

    if tokens and tokens[0] == '(':
      return function_exp(token, tokens)
    else:

      attr = parse_options.get_value(token)

      #attr = lambda row, ctx: get_value#get_value(row)# getattr(row, token)#row.get(token)
      attr.__name__ = str(token)
      return attr 

def group_exp(tokens):
  args = []
  if tokens[0] != ')':
    args.append(and_exp(tokens))
    while tokens[0] == ',':
      tokens.pop(0)
      args.append(and_exp(tokens))
  assert tokens[0] == ')'
  tokens.pop(0)

  def _(record,ctx):
    return [a(record, ctx) for a in args]
  _.__name__ = ",".join([a.__name__ for a in args])
  return _
 

def function_exp(name, tokens):
  token = tokens.pop(0)
  assert token == '('

  get_list = group_exp(tokens)

  def invoke_udf(record, ctx):
    udf = ctx['udf'][name]
    t = get_list(record, ctx)#[a(record, ctx) for a in args]
    return udf(*t)
  invoke_udf.__name__ = str(name)
  return invoke_udf


class Tokens(object):
  def __init__(self, statement):
    self.stream = StringIO(statement)
    self.current_char = None
    self.read_char()
    
  def __iter__(self):
    return self
        
  def next(self):
    self.skip_whitespace()
    if self.at_end():
      raise StopIteration()
    elif self.is_letter():
      return self.read_word()
    elif self.is_number():
      return self.read_number()
    elif self.current_char in ("'", '"'):
      return self.read_string_constant(self.current_char)
    else:
      return self.read_symbol()
      
  def read_char(self):
    self.current_char = self.stream.read(1)
    
  def skip_whitespace(self):
    while ('\0' < self.current_char   <= ' '):
      self.read_char()
  
  def at_end(self):
    return self.current_char == ''
    
  def is_letter(self):
    return self.current_char in string.letters + '_'
    
  def is_number(self):
    return self.current_char in string.digits 
  
  def is_letter_or_digit(self):
    return self.is_letter() or self.is_number()
  
  def read_word(self):
    buff  = []
    while not self.at_end() and self.is_letter_or_digit():
      buff.append(self.current_char)
      self.read_char()
    return ''.join(buff)
  
  def read_number(self):
    buff  = []
    while not self.at_end() and self.is_number():
      buff.append(self.current_char)
      self.read_char()
    return ''.join(buff)
    
  def read_symbol(self):
    if self.current_char in SYMBOLS:
      char = self.current_char
      self.read_char()
      return char
    elif self.current_char in "?$":
      char = self.current_char
      self.read_char()
      return char+self.read_word()

    elif self.current_char == '<':
      self.read_char()
      if self.current_char == '=':
        self.read_char()
        return '<='
      else:
        return '<'
    elif self.current_char == '>':
      self.read_char()
      if self.current_char == '=':
        self.read_char()
        return '>='
      else:
        return '>'
    elif self.current_char == '!':
      self.read_char()
      if self.current_char == '=':
        self.read_char()
        return "!="
      else:
        return '!'
    else:
      raise RuntimeError("Unexpected token " + self.current_char)
    
  def read_until(self, seek):
    buff  = []
    while not self.at_end() and self.current_char != seek:
      buff.append(self.current_char)
      self.read_char()
    return "".join(buff)
    
  def read_string_constant(self, quote_char):

    self.read_char()
    constant = self.read_until(quote_char)
    self.read_char()
    return '"' + constant + '"'

