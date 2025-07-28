data_0 = (1, True, "three", 3.14, 4)
# tuple_example1 = (1, 2, 3)
# tuple_example2 = (1,) # tuple_example2 = (1) creates an int
# tuple_example3 = ("delta",) # tuple_example3 = ("delta") creates a str

# attributes_and_methods = dir(data_0)
# attributes_and_methods_separated = "\n".join(attributes_and_methods)
# print(attributes_and_methods_separated)


# ----------------Tuple element access----------------
# result_tea_1 = data_0[1]
# result_tea_2 = data_0[5] # IndexError: tuple index out of range
# result_tea_3 = data_0[-3]
# result_tea_4 = data_0[-6] # IndexError: tuple index out of range

# tuple object is immutable
# data_0[0] = 'b' # TypeError: 'tuple' object does not support item assignment

# objects inside tuple can be mutable though
# data_tea_1 = (0, [1, 2])
# data_tea_1[1][1] = 5


# ----------------Tuple slicing----------------
# step parameter in all slicing operations is 1 by default. the positive sign sets the default direction as left to right.
# result_ts_1 = data_0[0:2]
# result_ts_2 = data_0[0:4]
# result_ts_3 = data_0[0:]
# result_ts_4 = data_0[:]
# result_ts_5 = data_0[0:4:2]
# result_ts_6 = data_0[-3:-1]
# result_ts_7 = data_0[-1:-3:-1]
# result_ts_8 = data_0[-1::-1]
# result_ts_9 = data_0[::-1]
# result_ts_10 = data_0[0::-1]


# ----------------tuple() function----------------
# data_tf_1 = "lorem ipsum contoso"
# result_tf_1 = tuple(data_tf_1)

# data_tf_2 = ["lorem", "ipsum", "contoso"]
# result_tf_2 = tuple(data_tf_2)

# data_tf_3 = {"lorem": 5, "ipsum": 5, "contoso": 7}
# result_tf_3_1 = tuple(data_tf_3)
# result_tf_3_2 = tuple(data_tf_3.values())
# result_tf_3_3 = tuple(data_tf_3.items())

# data_tf_4 = {"lorem", "ipsum", "contoso"}
# result_tf_4 = tuple(data_tf_4)


# ----------------Built-in functions----------------
# result_bif_1 = help(data_0) # --OOP-- ====> same as result_bif_6 = data_0.__doc__
# result_bif_2 = type(data_0) # --OOP-- ====> same as result_bif_6 = data_0.__class__
# result_bif_3 = id(data_0) # --OOP-- gives memory address of data_0
# result_bif_4 = dir(data_0) # --OOP-- ====> same as result_bif_6 = data_0.__dir__()

# result_bif_5 = len(data_0) # --OOP-- ====> same as result_bif_5 = data_0.__len__()

# result_bif_6 = all(data_0) # --OOP-- if even a single element is False --> return False, otherwise --> return True
# result_bif_7 = any(data_0) # --OOP-- if even a single element is not False --> return True, otherwise --> return False

# data_bif_1 = (1, 2, 5, -1, 3)
# data_bif_2 = ("eight", "three")
# result_bif_8 = min(data_bif_1) # --OOP-- works only with a tuple of int/float/str
# result_bif_9 = max(data_bif_1) # --OOP-- works only with a tuple of int/float/str
# result_bif_10 = sum(data_bif_1) # --OOP-- works only with a tuple of int/float

# result_bif_11 = min(data_bif_2) # --OOP-- works only with a tuple of int/float/str
# result_bif_12 = max(data_bif_2) # --OOP-- works only with a tuple of int/float/str

# result_bif_13 = list(data_0) # --OOP--

# data_bif_3 = ((1, [1,-5]), [0, [2, 3]], {5, (0, 9)})
# result_bif_14 = dict(data_bif_3) # --OOP-- each element in the tuple must be either a tuple, list, or set with two elements inside. the first element must have the __hash__() method implemented.

# data_bif_4 = (1, 2, (0, 4))
# data_bif_5 = (1, 2, [0, 4])
# result_bif_15 = set(data_bif_4) # --OOP-- each element in the tuple must have the __hash__() method implemented. note that calling the __hash__() method on anything calls the __hash__() method on all its components recursively
# result_bif_15 = set(data_bif_5) # TypeError: unhashable type: 'list'


# result_bif_16 = hash(data_0) # --OOP--

# result_bif_17 = iter(data_0) # --OOP--
# result_bif_18 = reversed(data_0) # --OOP--

# all combinations of elements in the tuple should support a comparison using the '<' operator
# result_bif_19_1 = sorted(data_0) # TypeError: '<' not supported between instances of 'str' and 'bool'
# result_bif_19_2 = sorted(data_bif_1) # --OOP--
# result_bif_19_3 = sorted(data_bif_1, reverse=True) # --OOP--

data_bif_6 = ("one", 2, False, "pi", -4)
# result_bif_20_1 = tuple(zip(data_0, ("one", 2, False, "pi", -4))) # --OOP--
# result_bif_20_2 = tuple(zip(data_0, data_bif_6)) # --OOP--
# result_bif_20_3 = tuple(zip(data_0, ("one", 2, False))) # --OOP--


# ----------------Built-in operations (see "Dunder Mapping" section)----------------
# result_bio_1_1 = data_0 + ("one", 2, False, "pi", -4)
# result_bio_1_2 = data_0 + data_bif_6

# data_bio_1 = (1, True, "three", 3.14, 4)
# data_bio_1 += ("one", 2, False, "pi", -4)
# data_bio_2 = (1, True, "three", 3.14, 4)
# data_bio_2 += data_bif_6

# result_bio_2 = data_0 * 3

# data_bio_3 = (1, True, "three", 3.14, 4)
# data_bio_3 *= 3


# ----------------Data Type specific dunder attributes & methods (see "Dunder Mapping" section)----------------
# result_dtsdaam_1 = data_0.__doc__
# result_dtsdaam_2 = data_0.__class__
# result_dtsdaam_3 = data_0.__dir__() # --OOP--

# result_dtsdaam_4 = data_0.__len__() # --OOP--

# result_dtsdaam_5 = data_0.__sizeof__() # --OOP--

# result_dtsdaam_6_1 = data_0.__getitem__(0) # --OOP--
# result_dtsdaam_6_2 = data_0.__getitem__(5) # IndexError: tuple index out of range
# result_dtsdaam_6_3 = data_0.__getitem__(-1) # --OOP--
# result_dtsdaam_6_4 = data_0.__getitem__(-6) # IndexError: tuple index out of range

# result_dtsdaam_7_1 = data_0.__add__(("one", 2, False, "pi", -4)) # --OOP--
# result_dtsdaam_7_2 = data_0.__add__(data_bif_6) # --OOP--

# result_dtsdaam_8 = data_0.__mul__(3) # --OOP--

# result_dtsdaam_9_1 = data_0.__contains__("three") # --OOP--
# result_dtsdaam_9_2 = data_0.__contains__(True) # --OOP--
# result_dtsdaam_9_3 = data_0.__contains__(0) # --OOP--
# result_dtsdaam_9_4 = data_0.__contains__(("three", 3.14)) # --OOP--

# __eq__() possible algo --> same ID?: True --> unequal len?: False --> compare each element
# if two elements are to be compared during this operation, they should support comparison using the '==' operator
# data_dtsdaam_1 = (1, True, "three", 3.14, 4)
# result_dtsdaam_14_1 = data_0.__eq__((1, True, "three", 3.14, 4)) # --OOP--
# result_dtsdaam_14_2 = data_0.__eq__(data_dtsdaam_1) # --OOP--
# result_dtsdaam_14_3 = data_0.__eq__((1, True, "three", 4, 3.14)) # --OOP--

# __ne__() possible algo --> same ID?: False --> unequal len?: True --> compare each element
# if two elements are to be compared during this operation, they should support comparison using the '!=' operator
# result_dtsdaam_15_1 = data_0.__ne__((1, True, "three", 3.14, 4)) # --OOP--
# result_dtsdaam_15_2 = data_0.__ne__(data_dtsdaam_1) # --OOP--
# result_dtsdaam_15_3 = data_0.__ne__((1, True, "three", 4, 3.14)) # --OOP--

# __lt__() possible algo --> same ID?: False --> compare each element till min(len(pri), len(sec)) --> if elements equal then len(pri) < len(sec)?: True
# if two elements are to be compared during this operation, they should support comparison using the '<' operator
# result_dtsdaam_16_1 = data_0.__lt__((1, True, "three", 3.14, 4, -5)) # --OOP--
# result_dtsdaam_16_2 = data_0.__lt__((1, True, "three", 3.14, 4, "five")) # --OOP--
# result_dtsdaam_16_3 = data_0.__lt__((1, True, "three", 3.14)) # --OOP--
# result_dtsdaam_16_4 = data_0.__lt__((1, True, "three", 3.14, 4)) # --OOP--
# result_dtsdaam_16_5 = data_0.__lt__((1, True, "three", 3.14, "four")) # TypeError: '<' not supported between instances of 'int' and 'str'

# __le__() possible algo --> same ID?: True --> compare each element till min(len(pri), len(sec)) --> if elements equal then len(pri) <= len(sec)?: True
# if two elements are to be compared during this operation, they should support comparison using the '<=' operator
# result_dtsdaam_17_1 = data_0.__le__((1, True, "three", 3.14, 4, -5)) # --OOP--
# result_dtsdaam_17_2 = data_0.__le__((1, True, "three", 3.14, 4, "five")) # --OOP--
# result_dtsdaam_17_3 = data_0.__le__((1, True, "three", 3.14)) # --OOP--
# result_dtsdaam_17_4 = data_0.__le__((1, True, "three", 3.14, 4)) # --OOP--
# result_dtsdaam_17_5 = data_0.__le__((1, True, "three", 3.14, "four")) # TypeError: '<' not supported between instances of 'int' and 'str'

# __gt__() possible algo --> same ID?: True --> compare each element till min(len(pri), len(sec)) --> if elements equal then len(pri) > len(sec)?: True
# if two elements are to be compared during this operation, they should support comparison using the '>' operator
# result_dtsdaam_18_1 = data_0.__gt__((1, True, "three", 3.14, 4, -5)) # --OOP--
# result_dtsdaam_18_2 = data_0.__gt__((1, True, "three", 3.14, 4, "five")) # --OOP--
# result_dtsdaam_18_3 = data_0.__gt__((1, True, "three", 3.14)) # --OOP--
# result_dtsdaam_18_4 = data_0.__gt__((1, True, "three", 3.14, 4)) # --OOP--
# result_dtsdaam_18_5 = data_0.__gt__((1, True, "three", 3.14, "four")) # TypeError: '<' not supported between instances of 'int' and 'str'

# __ge__() possible algo --> same ID?: True --> compare each element till min(len(pri), len(sec)) --> if elements equal then len(pri) >= len(sec)?: True
# if two elements are to be compared during this operation, they should support comparison using the '>=' operator
# result_dtsdaam_19_1 = data_0.__ge__((1, True, "three", 3.14, 4, -5)) # --OOP--
# result_dtsdaam_19_2 = data_0.__ge__((1, True, "three", 3.14, 4, "five")) # --OOP--
# result_dtsdaam_19_3 = data_0.__ge__((1, True, "three", 3.14)) # --OOP--
# result_dtsdaam_19_4 = data_0.__ge__((1, True, "three", 3.14, 4)) # --OOP--
# result_dtsdaam_19_5 = data_0.__ge__((1, True, "three", 3.14, "four")) # TypeError: '<' not supported between instances of 'int' and 'str'

# result_dtsdaam_20 = data_0.__hash__() # --OOP--

# result_dtsdaam_21 = data_0.__iter__() # --OOP--

# result_dtsdaam_22 = data_0.__reversed__() # --OOP--


# ----------------Data Type specific non-dunder attributes & methods----------------
# data_dtsnaam_1 = (1, True, "three", 3.14, 4, 3.14, "5ive")
# result_dtsnaam_1_1 = data_dtsnaam_1.index("three") # --OOP--
# result_dtsnaam_1_2 = data_dtsnaam_1.index(3.14) # --OOP--

# result_dtsnaam_2_1 = data_dtsnaam_1.count("three") # --OOP--
# result_dtsnaam_2_2 = data_dtsnaam_1.count(3.14) # --OOP--

pass