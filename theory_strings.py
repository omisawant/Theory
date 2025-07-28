data_0 = "alpha"
# string_example1 = 'bravo'
# string_example2 = '''charlie'''
# string_example3 = """delta"""

# attributes_and_methods = dir(data_0)
# attributes_and_methods_separated = "\n".join(attributes_and_methods)
# print(attributes_and_methods_separated)


# ----------------String formatting----------------
# data_sf_1 = "My name is {0}, I'm {1}"
# data_sf_2 = "My name is {name}, I'm {age}"
# result_sf_1_1 = "My name is John, I'm 23"
# result_sf_1_2 = "My name is %s, I'm %d" %("John",23)
# result_sf_1_3 = "My name is {0}, I'm {1}".format("John",23)
# result_sf_1_4 = data_sf_1.format("John",23)
# result_sf_1_5 = "My name is {name}, I'm {age}".format(name="John", age=23)
# result_sf_1_6 = "My name is {name}, I'm {age}".format(age=23, name="John")
# result_sf_1_7 = data_sf_2.format(name="John", age=23)
# name = "John"
# age = 23
# result_sf_1_8 = f"My name is {name}, I'm {age}"


# ----------------Element Access----------------
# result_sea_1_1 = data_0[0]
# result_sea_1_2 = data_0[5] # Error - Index out of Range
# result_sea_1_3 = data_0[-2]
# result_sea_1_4 = data_0[-6] # Error - Index out of Range

# str object is immutable
# data_0[0] = 'b' # TypeError: 'str' object does not support item assignment


# ----------------String slicing----------------
# result_ss_1_1 = data_0[0:2]
# result_ss_1_2 = data_0[0:4]
# result_ss_1_3 = data_0[0:]
# result_ss_1_4 = data_0[:]
# result_ss_1_5 = data_0[0:4:2]
# result_ss_1_6 = data_0[-3:-1]
# result_ss_1_7 = data_0[-1:-3:-1]
# result_ss_1_8 = data_0[-1::-1]
# result_ss_1_9 = data_0[::-1]
# result_ss_1_10 = data_0[0::-1]


# ----------------Built-in functions----------------
# result_bif_1 = len(data_0) # --OOP-- ====> same as result_1 = data_0.__len__()
# result_bif_2 = type(data_0) # --OOP--

# result_bif_3 = int(data_0) # --OOP--
# result_bif_4 = float(data_0) # --OOP--
# result_bif_5 = bool(data_0) # --OOP--
# result_bif_6 = list(data_0) # --OOP--
# result_bif_7 = tuple(data_0) # --OOP--
# result_bif_8 = dict(data_0) # --OOP--
# result_bif_9 = set(data_0) # --OOP--

# result_bif_10 = hash(data_0) # --OOP-- ====> same as result_10 = data_0.__hash__()

# result_bif_11 = iter(data_0) # --OOP--
# result_bif_12 = reversed(data_0) # --OOP--
# result_bif_13 = sorted(data_0) # --OOP--

# result_bif_14_1 = list(zip(data_0, "zyxwv")) # --OOP--
# result_bif_14_2 = list(zip(data_0, "zyx")) # --OOP--


# ----------------Built-in operations (see "Dunder Mapping" section)----------------
# result_bio_1 = data_0 + "zzz" # ====> same as result_15 = data_0.__add__("zzz")
# result_bio_2 = data_0 * 3 # ====> same as result_16 = data_0.__mul__(3)


# ----------------Data Type specific dunder attributes & methods (see "Dunder Mapping" section)----------------
# result_dtsdaam_1 = data_0.__len__() # --OOP--

# result_dtsdaam_2 = data_0.__add__("zzz") # --OOP--
# result_dtsdaam_3 = data_0.__mul__(3) # --OOP--

# result_dtsdaam_4_1 = data_0.__contains__("ph") # --OOP--
# result_dtsdaam_4_2 = data_0.__contains__("pz") # --OOP--

# result_dtsdaam_5_1 = data_0.__eq__("alpha") # --OOP--
# result_dtsdaam_5_2 = data_0.__eq__("alphaz") # --OOP--
# result_dtsdaam_5_3 = data_0.__eq__("lpha") # --OOP--

# result_dtsdaam_6_1 = data_0.__ne__("alpha") # --OOP--
# result_dtsdaam_6_2 = data_0.__ne__("alphaz") # --OOP--
# result_dtsdaam_6_3 = data_0.__ne__("lpha") # --OOP--

# result_dtsdaam_7_1 = data_0.__lt__("aldha") # --OOP--
# result_dtsdaam_7_2 = data_0.__lt__("alzha") # --OOP--
# result_dtsdaam_7_3 = data_0.__lt__("alpha") # --OOP--

# result_dtsdaam_8_1 = data_0.__le__("aldha") # --OOP--
# result_dtsdaam_8_2 = data_0.__le__("alzha") # --OOP--
# result_dtsdaam_8_3 = data_0.__le__("alpha") # --OOP--

# result_dtsdaam_9_1 = data_0.__gt__("aldha") # --OOP--
# result_dtsdaam_9_2 = data_0.__gt__("alzha") # --OOP--
# result_dtsdaam_9_3 = data_0.__gt__("alpha") # --OOP--

# result_dtsdaam_10_1 = data_0.__ge__("aldha") # --OOP--
# result_dtsdaam_10_2 = data_0.__ge__("alzha") # --OOP--
# result_dtsdaam_10_3 = data_0.__ge__("alpha") # --OOP--

# result_dtsdaam_11 = data_0.__hash__() # --OOP--

# result_dtsdaam_12 = data_0.__iter__() # --OOP--


# ----------------Data Type specific non-dunder attributes & methods----------------
# data_dtsnaam_1 = "My name is {0}, I'm {1}"
# result_dtsnaam_1_1 = "My name is {0}, I'm {1}".format("John",23) # --OOP--
# result_dtsnaam_1_2 = data_dtsnaam_1.format("John",23) # --OOP--
# result_dtsnaam_1_3 = "My name is {name}, I'm {age}".format(name="John", age=23) # --OOP--
# result_dtsnaam_1_4 = "My name is {name}, I'm {age}".format(age=23, name="John") # --OOP--

# data_dtsnaam_2 = "we are going to Denver and Las Vegas"
# result_dtsnaam_2 = data_dtsnaam_2.lower() # --OOP--
# result_dtsnaam_3 = data_dtsnaam_2.upper() # --OOP--
# result_dtsnaam_4 = data_dtsnaam_2.capitalize() # --OOP--
# result_dtsnaam_5 = data_dtsnaam_2.title() # --OOP--
# result_dtsnaam_6 = data_dtsnaam_2.swapcase() # --OOP--

# data_dtsnaam_3 = "    we are going to Denver and Las Vegas    "
# result_dtsnaam_7 = data_dtsnaam_3.lstrip() # --OOP--
# result_dtsnaam_8 = data_dtsnaam_3.rstrip() # --OOP--
# result_dtsnaam_9 = data_dtsnaam_3.strip() # --OOP--

# result_dtsnaam_10_1 = data_dtsnaam_2.removeprefix("we are") # --OOP--
# result_dtsnaam_10_2 = data_dtsnaam_2.removeprefix("are going") # --OOP--
# result_dtsnaam_11_1 = data_dtsnaam_2.removesuffix("Las Vegas") # --OOP--
# result_dtsnaam_11_2 = data_dtsnaam_2.removesuffix("and Las") # --OOP--

# data_dtsnaam_3 = "ABCxDEFxGHI"
# result_dtsnaam_12 = data_dtsnaam_3.partition('x') # --OOP--
# result_dtsnaam_13 = data_dtsnaam_3.rpartition('x') # --OOP--
# result_dtsnaam_14 = data_dtsnaam_3.split('x') # --OOP--

# data_dtsnaam_4 = "ABC\nDEF\nGHI"
# result_dtsnaam_15 = data_dtsnaam_4.splitlines() # --OOP--

# data_dtsnaam_5 = "aAbbBBcccCCC"
# result_dtsnaam_16_1 = data_dtsnaam_5.count('c') # --OOP--
# result_dtsnaam_16_2 = data_dtsnaam_5.count('bb') # --OOP--

# result_dtsnaam_17_1 = data_dtsnaam_5.find('b') # --OOP--
# result_dtsnaam_17_2 = data_dtsnaam_5.find('CCC') # --OOP--
# result_dtsnaam_18_1 = data_dtsnaam_5.rfind('b') # --OOP--
# result_dtsnaam_18_2 = data_dtsnaam_5.rfind('CCC') # --OOP--

# result_dtsnaam_19_1 = data_dtsnaam_5.index('b') # --OOP--
# result_dtsnaam_19_2 = data_dtsnaam_5.index('CCC') # --OOP--
# result_dtsnaam_20_1 = data_dtsnaam_5.rindex('b') # --OOP--
# result_dtsnaam_20_2 = data_dtsnaam_5.rindex('CCC') # --OOP--

# data_dtsnaam_6 = "aAabABabcABC"
# result_dtsnaam_21 = data_dtsnaam_6.replace('a', 'z') # --OOP--

# result_dtsnaam_22_1 = data_dtsnaam_2.startswith("we are") # --OOP--
# result_dtsnaam_22_2 = data_dtsnaam_2.startswith("are going") # --OOP--
# result_dtsnaam_23_1 = data_dtsnaam_2.endswith("Las Vegas") # --OOP--
# result_dtsnaam_23_2 = data_dtsnaam_2.endswith("and Las") # --OOP--

# data_dtsnaam_7 = "abc"
# data_dtsnaam_8 = "123"
# result_dtsnaam_24_1 = data_dtsnaam_7.isalpha() # --OOP--
# result_dtsnaam_24_2 = data_dtsnaam_8.isalpha() # --OOP--
# result_dtsnaam_25_1 = data_dtsnaam_7.isdigit() # --OOP--
# result_dtsnaam_25_2 = data_dtsnaam_8.isdigit() # --OOP--

# data_dtsnaam_9 = "123.456"
# result_dtsnaam_26 = data_dtsnaam_9.isdecimal() # --OOP--

# data_dtsnaam_10 = "abc123"
# result_dtsnaam_27 = data_dtsnaam_10.isalnum() # --OOP--

# data_dtsnaam_11 = " "
# result_dtsnaam_28 = data_dtsnaam_11.isspace() # --OOP--

# data_dtsnaam_12 = "Abc"
# result_dtsnaam_29 = data_dtsnaam_12.islower() # --OOP--
# result_dtsnaam_30 = data_dtsnaam_12.isupper() # --OOP--
# result_dtsnaam_31 = data_dtsnaam_12.istitle() # --OOP--

pass