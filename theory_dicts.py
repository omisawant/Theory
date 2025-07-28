data_0 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# dict_example1 = {1: 1, 2: 2, 3: 3}
# dict_example2 = {1: "one"}
# dict_example3 = {"delta": 22}

# attributes_and_methods = dir(data_0)
# attributes_and_methods_separated = "\n".join(attributes_and_methods)
# print(attributes_and_methods_separated)


# ----------------dict element access----------------
# result_dea_1 = data_0[1]
# result_dea_2 = data_0[5]
# result_dea_3 = data_0[-3] # KeyError: -3
# result_dea_4 = data_0[-6] # KeyError: -6

# dict object is mutable
# data_dea_1 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# data_dea_1["4our"] = 44


# ----------------dict() function----------------
# data_df_1 = "lorem ipsum contoso"
# result_df_1 = dict(data_df_1) # ValueError: dictionary update sequence element #0 has length 1; 2 is required

# data_df_2 = (("lorem", "1ne"), ["ipsum", 0.2], {"contoso", "thr33"})
# result_df_2 = dict(data_df_2) # each element in the tuple must be either a tuple, list, or set with two elements inside. the first element must have the __hash__() method implemented.

# data_df_3 = [("lorem", "1ne"), ["ipsum", 0.2], {"contoso", "thr33"}]
# result_df_3 = dict(data_df_3) # each element in the list must be either a tuple, list, or set with two elements inside. the first element must have the __hash__() method implemented.

# data_df_4 = {("lorem", "1ne"), ("ipsum", 0.2), ("contoso", "thr33")}
# result_df_4 = dict(data_df_4) # each element in the set must be either a tuple, list, or set with two elements inside. the first element must have the __hash__() method implemented.


# ----------------Dict comprehension----------------
# result_dc_1 = {data_dc_1: 1 for data_dc_1 in range(0,5)}
# result_dc_2 = {data_dc_2: "thr33" for data_dc_2 in range(0,5) if data_dc_2 != 3}
# result_dc_3 = {data_dc_3: "a"*data_dc_3 for data_dc_3 in range(0,5)}
# result_dc_4 = {1: data_dc_1 for data_dc_1 in range(0,5)}


# ----------------Built-in functions----------------
# result_bif_1 = help(data_0) # --OOP-- ====> same as result_bif_1 = data_0.__doc__
# result_bif_2 = type(data_0) # --OOP-- ====> same as result_bif_2 = data_0.__class__
# result_bif_3 = id(data_0) # --OOP-- gives memory address of data_0
# result_bif_4 = dir(data_0) # --OOP-- ====> same as result_bif_4 = data_0.__dir__()

# result_bif_5 = len(data_0) # --OOP-- ====> same as result_bif_5 = data_0.__len__()

# data_bif_1 = {False: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_bif_6 = all(data_bif_1) # --OOP-- if even a single key is False --> return False, otherwise --> return True
# result_bif_7 = any(data_bif_1) # --OOP-- if even a single key is not False --> return True, otherwise --> return False

# data_bif_2 = {1: "one", 2: "2wo", 5: "5ive", -1: "minus one", 3: "thr33"}
# data_bif_3 = {"eight": 8, "three": 3}
# result_bif_8 = min(data_bif_2) # --OOP-- doesn't operate on entire dict, operates on keys by default. all keys must be of the type int/float/str
# result_bif_9 = max(data_bif_2) # --OOP-- doesn't operate on entire dict, operates on keys by default. all keys must be of the type int/float/str
# result_bif_10 = sum(data_bif_2) # --OOP-- doesn't operate on entire dict, operates on keys by default. all keys must be of the type int/float/str

# result_bif_11 = min(data_bif_3) # --OOP-- doesn't operate on entire dict, operates on keys by default. all keys must be of the type int/float/str
# result_bif_12 = max(data_bif_3) # --OOP-- doesn't operate on entire dict, operates on keys by default. all keys must be of the type int/float/str

# result_bif_13 = list(data_0) # --OOP-- doesn't operate on entire dict, operates on keys by default

# result_bif_14 = tuple(data_0) # --OOP-- doesn't operate on entire dict, operates on keys by default

# result_bif_15 = set(data_0) # --OOP-- doesn't operate on entire dict, operates on keys by default

# result_bif_16 = hash(data_0) # TypeError: unhashable type: 'dict'

# result_bif_17 = iter(data_0) # --OOP--
# result_bif_18 = reversed(data_0) # --OOP-- doesn't operate on entire dict, operates on keys by default

# all combinations of elements in the string should support a comparison using the '<' operator
# result_bif_19_1 = sorted(data_0) # TypeError: '<' not supported between instances of 'str' and 'bool'
# result_bif_19_2 = sorted(data_bif_2) # --OOP-- doesn't operate on entire dict, operates on keys by default
# result_bif_19_3 = sorted(data_bif_2, reverse=True) # --OOP-- doesn't operate on entire dict, operates on keys by default

# data_bif_4 = {"one": 1, 2: "2wo", False: 3, "pi": "3.14", -4: "4our"}
# result_bif_20_1 = list(zip(data_0, {"one": 1, 2: "2wo", False: 3, "pi": "3.14", -4: "4our"})) # --OOP-- doesn't operate on entire dict, operates on keys by default. outputs an iterator of tuples
# result_bif_20_2 = list(zip(data_0, data_bif_4)) # --OOP-- doesn't operate on entire dict, operates on keys by default. outputs an iterator of tuples
# result_bif_20_3 = list(zip(data_0, {"one": 1, 2: "two", False: 3.14})) # --OOP-- doesn't operate on entire dict, operates on keys by default. outputs an iterator of tuples


# ----------------Built-in operations (see "Dunder Mapping" section)----------------
# data_bio_1 = {5: "five five", "six": 6, 7: "se7en"}
# result_bio_1 = data_bio_1 | data_0

# data_bio_2 = {5: "five five", "six": 6, 7: "se7en"}
# data_bio_2 |= data_0


# ----------------Data Type specific dunder attributes & methods (see "Dunder Mapping" section)----------------
# result_dtsdaam_1 = data_0.__doc__
# result_dtsdaam_2 = data_0.__class__
# result_dtsdaam_3 = data_0.__dir__() # --OOP--

# result_dtsdaam_4 = data_0.__len__() # --OOP--

# result_dtsdaam_5 = data_0.__sizeof__() # --OOP--

# result_dtsdaam_6_1 = data_0.__getitem__(1) # --OOP--
# result_dtsdaam_6_2 = data_0.__getitem__(2) # KeyError: 2

# data_dtsdaam_1 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsdaam_7_1 = data_dtsdaam_1.__setitem__("6ix", 6) # --IPNR--
# result_dtsdaam_7_2 = data_dtsdaam_1.__setitem__(5, "five five") # --IPNR--

# data_dtsdaam_3 = {5: "five five", "six": 6, 7: "se7en"}
# result_dtsdaam_8 = data_0.__or__(data_dtsdaam_3) # --OOP--

# result_dtsdaam_9 = data_0.__ror__(data_dtsdaam_3) # --OOP--

# data_dtsdaam_2 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsdaam_10 = data_dtsdaam_2.__ior__(data_dtsdaam_3) # --IPRR--

# result_dtsdaam_11 = data_0.__contains__(1) # --OOP--
# result_dtsdaam_12 = data_0.__contains__(2) # --OOP--

# data_dtsdaam_3 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsdaam_13 = data_dtsdaam_3.__delitem__(1) # --IPNR--

# data_dtsdaam_4 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# data_dtsdaam_5 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4}
# result_dtsdaam_14_1 = data_0.__eq__(data_dtsdaam_4) # --OOP--
# result_dtsdaam_14_2 = data_0.__eq__(data_dtsdaam_5) # --OOP--

# result_dtsdaam_15_1 = data_0.__ne__(data_dtsdaam_4) # --OOP--
# result_dtsdaam_15_2 = data_0.__ne__(data_dtsdaam_5) # --OOP--

# result_dtsdaam_16 = data_0.__hash__() # TypeError: 'NoneType' object is not callable

# result_dtsdaam_17 = data_0.__iter__() # --OOP--

# result_dtsdaam_18 = data_0.__reversed__() # --OOP-- doesn't operate on entire dict, operates on keys by default

# ----------------Data Type specific non-dunder attributes & methods----------------
# data_dtsnaam_1 = "a bc"
# data_dtsnaam_2 = [5, '4our']
# data_dtsnaam_3 = (5, '4our')
# data_dtsnaam_4 = {5: "5ive", '4our': 4}
# data_dtsnaam_5 = {5, '4our'}
# result_dtsnaam_1_1 = dict.fromkeys(data_dtsnaam_1, "goose")
# result_dtsnaam_1_2 = dict.fromkeys(data_dtsnaam_2, "goose")
# result_dtsnaam_1_3 = dict.fromkeys(data_dtsnaam_3, "goose")
# result_dtsnaam_1_4 = dict.fromkeys(data_dtsnaam_4, "goose")
# result_dtsnaam_1_5 = dict.fromkeys(data_dtsnaam_5, "goose")

# return the value for key if key is in the dictionary, else the parameter provided. if no value is provided, use None
# result_dtsnaam_2_1 = data_0.get(1, "goose") # --OOP--
# result_dtsnaam_2_2 = data_0.get(2, "goose") # --OOP--

# if key is in the dictionary, return its value. if not, insert key with value provided and return that value. if no value provided, use None
# data_dtsnaam_6 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsnaam_3 = data_dtsnaam_6.setdefault(1, "1ne") # --OOP--
# result_dtsnaam_4 = data_dtsnaam_6.setdefault("six", 6) # --OOP--

# same as pri |= sec
# data_dtsnaam_7 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# data_dtsnaam_8 = (("lorem", "1ne"), ["ipsum", 0.2], {"contoso", "thr33"})
# result_dtsnaam_5_1 = data_dtsnaam_7.update(data_dtsnaam_8) # --IPNR--

# data_dtsnaam_9 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# data_dtsnaam_10 = [("lorem", "1ne"), ["ipsum", 0.2], {"contoso", "thr33"}]
# result_dtsnaam_5_2 = data_dtsnaam_9.update(data_dtsnaam_10) # --IPNR--

# data_dtsnaam_11 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# data_dtsnaam_12 = {("lorem", "1ne"), ("ipsum", 0.2), ("contoso", "thr33")}
# result_dtsnaam_5_3 = data_dtsnaam_11.update(data_dtsnaam_12) # --IPNR--

# data_dtsnaam_13 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# data_dtsnaam_14 = {"6ix": 6, 7: "se7en"}
# result_dtsnaam_5_4 = data_dtsnaam_13.update(data_dtsnaam_14) # --IPNR--

# data_dtsnaam_15 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# data_dtsnaam_16 = {5: "five five", "6ix": 6, 7: "se7en"}
# result_dtsnaam_5_5 = data_dtsnaam_15.update(data_dtsnaam_16) # --IPNR--

# If key is in the dictionary, remove it and return its value, else return default. If default is not given and key is not in the dictionary, a KeyError is raised.
# data_dtsnaam_17 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsnaam_6_1 = data_dtsnaam_17.pop(1) # --IPCR: corresponding value or provided parameter--

# data_dtsnaam_18 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsnaam_6_2 = data_dtsnaam_18.pop(1, "goose") # --IPCR: corresponding value or provided parameter--

# data_dtsnaam_19 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsnaam_6_3 = data_dtsnaam_19.pop(2, "goose") # --IPCR: corresponding value or provided parameter--

# data_dtsnaam_20 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsnaam_6_4 = data_dtsnaam_20.pop(2) # KeyError: 2

# data_dtsnaam_21 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsnaam_7 = data_dtsnaam_21.popitem() # --IPCR: tuple(key,value)--

# result_dtsnaam_22 = data_0.copy() # --IPRR--

# data_dtsnaam_22 = {1: "one", "tw0": 2, 3.14: "pi", "4our": 4, 5: "5ive"}
# result_dtsnaam_23 = data_dtsnaam_22.clear() # --IPNR--

# result_dtsnaam_24 = data_0.keys() # --OOP--

# result_dtsnaam_25 = data_0.values() # --OOP--

# result_dtsnaam_26 = data_0.items() # --OOP--

pass