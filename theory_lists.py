data_0 = [1, True, "three", 3.14, 4]
# list_example1 = [1, 2, 3]
# list_example2 = [1]
# list_example3 = ["delta"]

# attributes_and_methods = dir(data_0)
# attributes_and_methods_separated = "\n".join(attributes_and_methods)
# print(attributes_and_methods_separated)


# ----------------List element access----------------
# result_lea_1 = data_0[1]
# result_lea_2 = data_0[5] # IndexError: list index out of range
# result_lea_3 = data_0[-3]
# result_lea_4 = data_0[-6] # IndexError: list index out of range

# list object is mutable
# data_lea_1 = [1, True, "three", 3.14, 4]
# data_lea_1[0] = 'b'

# ----------------List slicing----------------
# step parameter in all slicing operations is 1 by default. the positive sign sets the default direction as left to right.
# result_ls_1 = data_0[0:2]
# result_ls_2 = data_0[0:4]
# result_ls_3 = data_0[0:]
# result_ls_4 = data_0[:]
# result_ls_5 = data_0[0:4:2]
# result_ls_6 = data_0[-3:-1]
# result_ls_7 = data_0[-1:-3:-1]
# result_ls_8 = data_0[-1::-1]
# result_ls_9 = data_0[::-1]
# result_ls_10 = data_0[0::-1]


# ----------------list() function----------------
# data_lf_1 = "lorem ipsum contoso"
# result_lf_1 = list(data_lf_1)

# data_lf_2 = ("lorem", "ipsum", "contoso")
# result_lf_2 = list(data_lf_2)

# data_lf_3 = {"lorem": 5, "ipsum": 5, "contoso": 7}
# result_lf_3_1 = list(data_lf_3)
# result_lf_3_2 = list(data_lf_3.values())
# result_lf_3_3 = list(data_lf_3.items())

# data_lf_4 = {"lorem", "ipsum", "contoso"}
# result_lf_4 = list(data_lf_4)


# ----------------List comprehension----------------
# result_lc_1 = [data_lc_1 for data_lc_1 in range(0,5)]
# result_lc_2 = [data_lc_2 for data_lc_2 in range(0,5) if data_lc_2 != 3]
# result_lc_3 = [data_lc_3 * 2 for data_lc_3 in range(0,5)]


# ----------------Built-in functions----------------
# result_bif_1 = help(data_0) # --OOP-- ====> same as result_bif_6 = data_0.__doc__
# result_bif_2 = type(data_0) # --OOP-- ====> same as result_bif_6 = data_0.__class__
# result_bif_3 = id(data_0) # --OOP-- gives memory address of data_0
# result_bif_4 = dir(data_0) # --OOP-- ====> same as result_bif_6 = data_0.__dir__()

# result_bif_5 = len(data_0) # --OOP-- ====> same as result_bif_5 = data_0.__len__()

# result_bif_6 = all(data_0) # --OOP-- if even a single element is False --> return False, otherwise --> return True
# result_bif_7 = any(data_0) # --OOP-- if even a single element is not False --> return True, otherwise --> return False

# data_bif_1 = [1, 2, 5, -1, 3]
# data_bif_2 = ["eight", "three"]
# result_bif_8 = min(data_bif_1) # --OOP-- works only with a list of int/float/str
# result_bif_9 = max(data_bif_1) # --OOP-- works only with a list of int/float/str
# result_bif_10 = sum(data_bif_1) # --OOP-- works only with a list of int/float

# result_bif_11 = min(data_bif_2) # --OOP-- works only with a list of int/float/str
# result_bif_12 = max(data_bif_2) # --OOP-- works only with a list of int/float/str

# result_bif_13 = tuple(data_0) # --OOP--

# data_bif_3 = [(1, [1,-5]), [0, [2, 3]], {5, (0, 9)}]
# result_bif_14 = dict(data_bif_3) # --OOP-- each element in the list must be either a tuple, list, or set with two elements inside. the first element must have the __hash__() method implemented.

# data_bif_4 = [1, 2, (0, 4)]
# result_bif_15 = set(data_bif_4) # --OOP-- each element in the list must have the __hash__() method implemented. note that calling the __hash__() method on anything calls the __hash__() method on all its components recursively

# result_bif_16 = hash(data_0) # TypeError: unhashable type: 'list'

# result_bif_17 = iter(data_0) # --OOP--
# result_bif_18 = reversed(data_0) # --OOP--

# all combinations of elements in the string should support a comparison using the '<' operator
# result_bif_19_1 = sorted(data_0) # TypeError: '<' not supported between instances of 'str' and 'bool'
# result_bif_19_2 = sorted(data_bif_1) # --OOP--
# result_bif_19_3 = sorted(data_bif_1, reverse=True) # --OOP--

# data_bif_5 = ["one", 2, False, "pi", -4]
# result_bif_20_1 = list(zip(data_0, ["one", 2, False, "pi", -4])) # --OOP--
# result_bif_20_2 = list(zip(data_0, data_bif_5)) # --OOP--
# result_bif_20_3 = list(zip(data_0, ["one", 2, False])) # --OOP--


# ----------------Built-in operations (see "Dunder Mapping" section)----------------
# result_bio_1_1 = data_0 + ["one", 2, False, "pi", -4]
# result_bio_1_2 = data_0 + data_bif_5

# data_bio_1 = [1, True, "three", 3.14, 4]
# data_bio_1 += ["one", 2, False, "pi", -4]
# data_bio_2 = [1, True, "three", 3.14, 4]
# data_bio_2 += data_bif_5

# result_bio_2 = data_0 * 3

# data_bio_3 = [1, True, "three", 3.14, 4]
# data_bio_3 *= 3


# ----------------Data Type specific dunder attributes & methods (see "Dunder Mapping" section)----------------
# result_dtsdaam_1 = data_0.__doc__
# result_dtsdaam_2 = data_0.__class__
# result_dtsdaam_3 = data_0.__dir__() # --OOP--

# result_dtsdaam_4 = data_0.__len__() # --OOP--

# result_dtsdaam_5 = data_0.__sizeof__() # --OOP--

# result_dtsdaam_6_1 = data_0.__getitem__(0) # --OOP--
# result_dtsdaam_6_2 = data_0.__getitem__(5) # IndexError: list index out of range
# result_dtsdaam_6_3 = data_0.__getitem__(-1) # --OOP--
# result_dtsdaam_6_4 = data_0.__getitem__(-6) # IndexError: list index out of range

# data_dtsdaam_1 = [1, True, "three", 3.14, 4]
# result_dtsdaam_7_1 = data_dtsdaam_1.__setitem__(0, 2) # --IPNR--
# result_dtsdaam_7_2 = data_dtsdaam_1.__setitem__(5, 6) # IndexError: list assignment index out of range
# result_dtsdaam_7_3 = data_dtsdaam_1.__setitem__(-1, 5) # --IPNR--
# result_dtsdaam_7_4 = data_dtsdaam_1.__setitem__(-6, 0) # IndexError: list assignment index out of range

# result_dtsdaam_8_1 = data_0.__add__(["one", 2, False, "pi", -4]) # --OOP--
# result_dtsdaam_8_2 = data_0.__add__(data_bif_5) # --OOP--

# data_dtsdaam_2 = [1, True, "three", 3.14, 4]
# result_dtsdaam_9_1 = data_dtsdaam_2.__iadd__(["one", 2, False, "pi", -4]) # --IPRR--
# data_dtsdaam_3 = [1, True, "three", 3.14, 4]
# result_dtsdaam_9_2 = data_dtsdaam_3.__iadd__(data_bif_5) # --IPRR--

# result_dtsdaam_10 = data_0.__mul__(3) # --OOP--

# data_dtsdaam_4 = [1, True, "three", 3.14, 4]
# result_dtsdaam_11 = data_dtsdaam_4.__imul__(3) # --IPRR--

# result_dtsdaam_12_1 = data_0.__contains__("three") # --OOP--
# result_dtsdaam_12_2 = data_0.__contains__(True) # --OOP--
# result_dtsdaam_12_3 = data_0.__contains__(0) # --OOP--
# result_dtsdaam_12_4 = data_0.__contains__(["three", 3.14]) # --OOP--

# data_dtsdaam_4 = [1, True, "three", 3.14, 4]
# result_dtsdaam_13_1 = data_dtsdaam_4.__delitem__(0) # --IPNR--

# __eq__() possible algo --> same ID?: True --> unequal len?: False --> compare each element
# if two elements are to be compared during this operation, they should support comparison using the '==' operator
# data_dtsdaam_4 = [1, True, "three", 3.14, 4]
# result_dtsdaam_14_1 = data_0.__eq__([1, True, "three", 3.14, 4]) # --OOP--
# result_dtsdaam_14_2 = data_0.__eq__(data_dtsdaam_4) # --OOP--
# result_dtsdaam_14_3 = data_0.__eq__([1, True, "three", 4, 3.14]) # --OOP--

# __ne__() possible algo --> same ID?: False --> unequal len?: True --> compare each element
# if two elements are to be compared during this operation, they should support comparison using the '!=' operator
# result_dtsdaam_15_1 = data_0.__ne__([1, True, "three", 3.14, 4]) # --OOP--
# result_dtsdaam_15_2 = data_0.__ne__(data_dtsdaam_4) # --OOP--
# result_dtsdaam_15_3 = data_0.__ne__([1, True, "three", 4, 3.14]) # --OOP--

# __lt__() possible algo --> same ID?: False --> compare each element till min(len(pri), len(sec)) --> if elements equal then len(pri) < len(sec)?: True
# if two elements are to be compared during this operation, they should support comparison using the '<' operator
# result_dtsdaam_16_1 = data_0.__lt__([1, True, "three", 3.14, 4, -5]) # --OOP--
# result_dtsdaam_16_2 = data_0.__lt__([1, True, "three", 3.14, 4, "five"]) # --OOP--
# result_dtsdaam_16_3 = data_0.__lt__([1, True, "three", 3.14]) # --OOP--
# result_dtsdaam_16_4 = data_0.__lt__([1, True, "three", 3.14, 4]) # --OOP--
# result_dtsdaam_16_5 = data_0.__lt__([1, True, "three", 3.14, "four"]) # TypeError: '<' not supported between instances of 'int' and 'str'

# __le__() possible algo --> same ID?: True --> compare each element till min(len(pri), len(sec)) --> if elements equal then len(pri) <= len(sec)?: True
# if two elements are to be compared during this operation, they should support comparison using the '<=' operator
# result_dtsdaam_17_1 = data_0.__le__([1, True, "three", 3.14, 4, -5]) # --OOP--
# result_dtsdaam_17_2 = data_0.__le__([1, True, "three", 3.14, 4, "five"]) # --OOP--
# result_dtsdaam_17_3 = data_0.__le__([1, True, "three", 3.14]) # --OOP--
# result_dtsdaam_17_4 = data_0.__le__([1, True, "three", 3.14, 4]) # --OOP--
# result_dtsdaam_17_5 = data_0.__le__([1, True, "three", 3.14, "four"]) # TypeError: '<' not supported between instances of 'int' and 'str'

# __gt__() possible algo --> same ID?: True --> compare each element till min(len(pri), len(sec)) --> if elements equal then len(pri) > len(sec)?: True
# if two elements are to be compared during this operation, they should support comparison using the '>' operator
# result_dtsdaam_18_1 = data_0.__gt__([1, True, "three", 3.14, 4, -5]) # --OOP--
# result_dtsdaam_18_2 = data_0.__gt__([1, True, "three", 3.14, 4, "five"]) # --OOP--
# result_dtsdaam_18_3 = data_0.__gt__([1, True, "three", 3.14]) # --OOP--
# result_dtsdaam_18_4 = data_0.__gt__([1, True, "three", 3.14, 4]) # --OOP--
# result_dtsdaam_18_5 = data_0.__gt__([1, True, "three", 3.14, "four"]) # TypeError: '<' not supported between instances of 'int' and 'str'

# __ge__() possible algo --> same ID?: True --> compare each element till min(len(pri), len(sec)) --> if elements equal then len(pri) >= len(sec)?: True
# if two elements are to be compared during this operation, they should support comparison using the '>=' operator
# result_dtsdaam_19_1 = data_0.__ge__([1, True, "three", 3.14, 4, -5]) # --OOP--
# result_dtsdaam_19_2 = data_0.__ge__([1, True, "three", 3.14, 4, "five"]) # --OOP--
# result_dtsdaam_19_3 = data_0.__ge__([1, True, "three", 3.14]) # --OOP--
# result_dtsdaam_19_4 = data_0.__ge__([1, True, "three", 3.14, 4]) # --OOP--
# result_dtsdaam_19_5 = data_0.__ge__([1, True, "three", 3.14, "four"]) # TypeError: '<' not supported between instances of 'int' and 'str'

# result_dtsdaam_20 = data_0.__hash__() # TypeError: 'NoneType' object is not callable

# result_dtsdaam_21 = data_0.__iter__() # --OOP--

# result_dtsdaam_22 = data_0.__reversed__() # --OOP--


# ----------------Data Type specific non-dunder attributes & methods----------------

# data_dtsnaam_1 = [1, True, "three", 3.14, 4]
# result_dtsnaam_1_1 = data_dtsnaam_1.append("f1ve") # --IPNR--
# result_dtsnaam_1_2 = data_dtsnaam_1.append(["one", -2, "thr33"]) # --IPNR--
# result_dtsnaam_1_3 = data_dtsnaam_1.append([4]) # --IPNR--

# data_dtsnaam_2 = [1, True, "three", 3.14, 4]
# result_dtsnaam_2_1 = data_dtsnaam_2.insert(1, 1.5) # --IPNR--
# result_dtsnaam_2_2 = data_dtsnaam_2.insert(2, [5, 6, 7]) # --IPNR--
# result_dtsnaam_2_3 = data_dtsnaam_2.insert(0, 0) # --IPNR--
# result_dtsnaam_2_4 = data_dtsnaam_2.insert(4, 3.75) # --IPNR--
# result_dtsnaam_2_5 = data_dtsnaam_2.insert(5, "5ive") # --IPNR--
# result_dtsnaam_2_6 = data_dtsnaam_2.insert(8, 20) # --IPNR--
# result_dtsnaam_2_7 = data_dtsnaam_2.insert(-1, 30) # --IPNR--
# result_dtsnaam_2_8 = data_dtsnaam_2.insert(-5, 50) # --IPNR--
# result_dtsnaam_2_9 = data_dtsnaam_2.insert(-8, 80) # --IPNR--

# data_dtsnaam_3 = [1, True, "three", 3.14, 4]
# result_dtsnaam_3_1 = data_dtsnaam_3.extend(["5ive", "6ix", 7]) # --IPNR--
# result_dtsnaam_3_2 = data_dtsnaam_3.extend(8) # TypeError: 'int' object is not iterable
# result_dtsnaam_3_3 = data_dtsnaam_3.extend(("5ive", "6ix", 7)) # --IPNR--
# result_dtsnaam_3_4 = data_dtsnaam_3.extend({"5ive", "6ix", 7}) # --IPNR--
# result_dtsnaam_3_5 = data_dtsnaam_3.extend({"5ive": 5, "6ix": 6, 7: 7}) # --IPNR--
# result_dtsnaam_3_5 = data_dtsnaam_3.extend({"5ive": 5, "6ix": 6, 7: 7}.values()) # --IPNR--
# result_dtsnaam_3_5 = data_dtsnaam_3.extend({"5ive": 5, "6ix": 6, 7: 7}.items()) # --IPNR--

# data_dtsnaam_4 = [1, True, "three", 3.14, 4]
# result_dtsnaam_4_1 = data_dtsnaam_4.pop(1) # --IPCR: Boolean(operation successful?: True)--
# result_dtsnaam_4_2 = data_dtsnaam_4.pop(5) # IndexError: pop index out of range
# result_dtsnaam_4_3 = data_dtsnaam_4.pop(-1) # --IPCR: Boolean(operation successful?: True)--
# result_dtsnaam_4_4 = data_dtsnaam_4.pop(-6) # IndexError: pop index out of range

# data_dtsnaam_5 = [1, True, "three", 3.14, 4, 3.14, "5ive"]
# result_dtsnaam_5_1 = data_dtsnaam_5.remove("three") # --IPNR--
# result_dtsnaam_5_2 = data_dtsnaam_5.remove(3.14) # --IPNR--
# result_dtsnaam_5_3 = data_dtsnaam_5.remove(6) # ValueError: list.remove(x): x not in list

# result_dtsnaam_6 = data_dtsnaam_1.copy() # --IPRR--

# result_dtsnaam_7 = data_dtsnaam_1.clear() # --IPNR--

# data_dtsnaam_6 = [1, True, "three", 3.14, 4, 3.14, "5ive"]
# result_dtsnaam_8_1 = data_dtsnaam_6.index("three") # --OOP--
# result_dtsnaam_8_2 = data_dtsnaam_6.index(3.14) # --OOP--

# result_dtsnaam_9_1 = data_dtsnaam_6.count("three") # --OOP--
# result_dtsnaam_9_2 = data_dtsnaam_6.count(3.14) # --OOP--

# result_dtsnaam_10_1 = data_0.reverse() # --OOP--

# all combinations of elements in the string should support a comparison using the '<' operator
# data_dtsnaam_7 = [1, 2, 7, 4, 9, 10, -1]
# data_dtsnaam_8 = [1, True, "three", 3.14, 4, 3.14, "5ive"]
# result_dtsnaam_11_1 = data_dtsnaam_7.sort() # --IPNR--
# result_dtsnaam_11_2 = data_dtsnaam_7.sort(reverse=True) # --IPNR--
# result_dtsnaam_11_3 = data_dtsnaam_8.sort() # TypeError: '<' not supported between instances of 'str' and 'bool'

pass