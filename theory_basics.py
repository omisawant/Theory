# ----------------Data Types----------------
# built-in data types -> text
# string_data_type = 'anything'

# built-in data types -> number
# integer_data_type = 123
# float_data_type = 3.14

# built-in data types -> sequence
# list_data_type = [1, 2, 'three', 4, '5ive']
# tuple_data_type = (1, 2, 'three', 4, '5ive')

# built-in data types -> map
# dictionary_data_type = {'apple' : 5, 'banana' : 4}

# built-in data types -> set
# set_data_type = {1, 2, 1, 3, 4} # removes duplicates -> {1, 2, 3, 4}

# built-in data types -> boolean
# boolean_data_type = True

# Data Type -> Built-in Data Type, User Defined Data Type
# all Data Types are Objects
# Objects -> Attributes & Methods

# TIP -> use dir(object) function to get available attributes and methods for an object (including built-in types)
# attributes_and_methods = dir(string_data_type)
# attributes_and_methods_separated = "\n".join(attributes_and_methods)
# print(attributes_and_methods_separated)

# immutable data types: attributes, methods --> out-of-place
# mutable types: attributes, methods --> in-place, out-of-place


# ----------------Dunder mapping----------------
# Behind the scenes --> Best practice way
# pri.__doc__ --> help(pri)
# pri.__class__ --> type(pri)
# pri.__dir__() --> dir(pri)

# pri.__len__() --> len(pri)
# pri.__sizeof__() --> sys.getsizeof(pri)

# pri.__getitem__(x) --> pri[x]
# pri.__getattribute__(x) --> pri.x OR getattr(pri, x)

# pri.__setitem__(x, y) --> pri[x] = y
# pri.__setattr__(x, y) --> pri.x = y OR setattr(pri, x)

# pri.__delitem__(x) --> del pri[x]
# pri.__delattr__(x) --> del pri.x OR delattr(pri, x)

# pri.__contains__(x) --> x in pri

# pri.__abs__() --> abs(pri)
# pri.__bool__() --> bool(pri)

# pri.__add__(sec) --> pri + sec
# pri.__iadd__(sec) --> pri += sec
# pri.__mul__(sec) --> pri * sec
# pri.__rmul__(sec) --> sec * pri
# pri.__imul__(sec) --> pri *= sec

# pri.__eq__(sec) --> pri == sec
# pri.__ne__(sec) --> pri != sec
# pri.__lt__(sec) --> pri < sec
# pri.__le__(sec) --> pri <= sec
# pri.__gt__(sec) --> pri > sec
# pri.__ge__(sec) --> pri >= sec

# pri.__and__(sec) --> pri & sec

# pri.__hash__() --> hash(pri)

# pri.__format__(spec) --> format(pri, spec)
# pri.__repr__() --> repr(pri)
# pri.__str__() --> str(pri)

# pri.__iter__() --> iter(pri)
# pri.__reversed__() --> reversed(pri)


# ----------------Functions----------------
# function structure
# def function1(f1_input1, f1_input2):
#     f1_variable1 = f1_input1 + f1_input2
#     f1_output1 = f1_variable1 * 3
#     print(f1_output1)

# def function2(f2_input1, f2_input2):
#     f2_variable1 = f2_input1 + f2_input2
#     f2_output1 = f2_variable1 * 3
#     return f2_output1

# Python uses "Pass by Object Reference" to pass data to functions
# def function3(f3_input1):
#     print("Inside function3 step 1 --> f3_input1 ID: " + str(id(f3_input1)) + " --> f3_input1 value: " + str(f3_input1))
#     f3_input1 = f3_input1 * 3
#     print("Inside function3 step 2 --> f3_input1 ID: " + str(id(f3_input1)) + " --> f3_input1 value: " + str(f3_input1))

# data_1 = 16
# print("Before calling function3 --> data_1 ID: " + str(id(data_1)) + " --> data_1 value: " + str(data_1))
# function3(data_1)
# print("After calling function3 --> data_1 ID: " + str(id(data_1)) + " --> data_1 value: " + str(data_1))

# immutable types are never modified, mutable types could be modified or remain the same
# def function4(f4_input1):
#     f4_input1.removesuffix("ipsum")

# def function5(f5_input1):
#     f5_input1.remove("ipsum")

# data_2 = "lorem ipsum"
# print("data_2 before calling function4 --> " + data_2)
# function4(data_2)
# print("data_2 after calling function4 --> " + data_2)

# data_3 = ["lorem", "ipsum"]
# print("data_3 before calling function5 --> " + str(data_3))
# function5(data_3)
# print("data_3 after calling function5 --> " + str(data_3))