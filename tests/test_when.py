from nose.tools import eq_, raises
from codd import when

def test_no_args():
  test = when()
  
  @test()
  def function():
    return True
    
  eq_(test.invoke(), True)
  
  raises(ValueError)(test.invoke)(1)
  
def test_multiple_functions():
  test = when()

  @test("isinstance(_, int)")
  def is_int(x):
    return x
    
  @test("isinstance(_, str)")
  def is_str(x):
    return x
    
  raises(ValueError)(test.invoke)()
  
  eq_(test.invoke(1), 1)
  eq_(test.invoke("blah"), "blah")
  raises(ValueError)(test.invoke)({})
  
  @test()
  def recurse():
    return test.invoke("foo")
    
  eq_(test.invoke(), "foo")
  
def test_overiding_func_names():
  test = when()
  
  @test('True', 'True')
  def x(arg1, arg2):
    return 2
    
  @test('True')
  def x(arg1):
    return 1
    
  eq_(test.invoke('with one arg'), 1)
  eq_(test.invoke('one arg', 'two'), 2)
  
def test_namespace():
  test = when(z=1)

  @test("_ == z")
  def myfunc(x):
    assert x == 1
    
  test.invoke(1)