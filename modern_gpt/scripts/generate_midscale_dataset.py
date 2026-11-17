"""Complete generator and validator for the 500-example ModernGPT-S dataset."""

import ast
import hashlib
import json
from pathlib import Path
import sys

DATA_DIR = Path("data/midscale_500")

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def test_code_snippet(code: str, tests: list):
    clean = code.strip()
    if clean.startswith("```python"):
        clean = clean[len("```python"):].strip()
    if clean.endswith("```"):
        clean = clean[:-3].strip()

    ast.parse(clean)
    safe_builtins = {
        "len": len, "range": range, "abs": abs, "min": min, "max": max,
        "isinstance": isinstance, "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "True": True, "False": False, "None": None, "enumerate": enumerate,
        "sum": sum, "reversed": reversed, "sorted": sorted, "round": round,
        "all": all, "any": any,
    }
    scope = {}
    exec(clean, {"__builtins__": safe_builtins}, scope)
    import inspect
    for fn_name, args, expected in tests:
        fn = scope.get(fn_name)
        assert callable(fn), f"Function {fn_name} not found"
        num_params = len(inspect.signature(fn).parameters)
        if num_params == 1:
            call_args = (args,)
        else:
            call_args = args if isinstance(args, tuple) else (args,)
        res = fn(*call_args)
        if isinstance(res, float) and isinstance(expected, float):
            assert abs(res - expected) < 1e-4, f"{fn_name}{args} returned {res}, expected {expected}"
        else:
            assert res == expected, f"{fn_name}{args} returned {res}, expected {expected}"

# Generators for the 25 Python families
def build_python_tasks():
    families = []
    
    # 1. arithmetic_ops
    f1 = ("arithmetic_ops", [
        ("add", "Write a Python function add(a, b) that returns the sum of a and b.", "def add(a, b):\n    return a + b", [("add", (2, 3), 5), ("add", (0, 0), 0), ("add", (-4, 7), 3), ("add", (-5, -5), -10)]),
        ("subtract", "Write a Python function subtract(a, b) that returns a minus b.", "def subtract(a, b):\n    return a - b", [("subtract", (10, 4), 6), ("subtract", (0, 5), -5), ("subtract", (-3, -3), 0)]),
        ("multiply", "Write a Python function multiply(a, b) that returns the product of a and b.", "def multiply(a, b):\n    return a * b", [("multiply", (4, 5), 20), ("multiply", (0, 100), 0), ("multiply", (-3, 6), -18), ("multiply", (-2, -4), 8)]),
        ("int_divide", "Write a Python function int_divide(a, b) that returns integer division of a by b.", "def int_divide(a, b):\n    return a // b", [("int_divide", (14, 3), 4), ("int_divide", (9, 3), 3), ("int_divide", (-10, 3), -4)]),
        ("modulo", "Write a Python function modulo(a, b) that returns the remainder of a divided by b.", "def modulo(a, b):\n    return a % b", [("modulo", (14, 3), 2), ("modulo", (10, 5), 0), ("modulo", (7, 4), 3)]),
        ("negate", "Write a Python function negate(n) that returns the numeric negation of n.", "def negate(n):\n    return -n", [("negate", (10), -10), ("negate", (0), 0), ("negate", (-25), 25)]),
        ("double", "Write a Python function double(n) that returns twice the input value.", "def double(n):\n    return n * 2", [("double", (8), 16), ("double", (0), 0), ("double", (-7), -14)]),
        ("triple", "Write a Python function triple(n) that returns three times the input value.", "def triple(n):\n    return n * 3", [("triple", (4), 12), ("triple", (0), 0), ("triple", (-5), -15)]),
    ])
    families.append(f1)

    # 2. powers_roots
    f2 = ("powers_roots", [
        ("square", "Write a Python function square(n) that returns the square of n.", "def square(n):\n    return n * n", [("square", (5), 25), ("square", (0), 0), ("square", (-6), 36)]),
        ("cube", "Write a Python function cube(n) that returns the cube of n.", "def cube(n):\n    return n * n * n", [("cube", (3), 27), ("cube", (0), 0), ("cube", (-3), -27)]),
        ("power", "Write a Python function power(base, exp) that returns base raised to the exp power.", "def power(base, exp):\n    return base ** exp", [("power", (2, 3), 8), ("power", (5, 0), 1), ("power", (10, 2), 100)]),
        ("power_four", "Write a Python function power_four(n) that returns n raised to the fourth power.", "def power_four(n):\n    return n ** 4", [("power_four", (2), 16), ("power_four", (1), 1), ("power_four", (-2), 16)]),
        ("is_square_zero", "Write a Python function is_square_zero(n) that returns True if n squared is 0.", "def is_square_zero(n):\n    return n * n == 0", [("is_square_zero", (0), True), ("is_square_zero", (4), False), ("is_square_zero", (-2), False)]),
        ("square_add", "Write a Python function square_add(a, b) that returns the square of a plus b.", "def square_add(a, b):\n    return a * a + b", [("square_add", (3, 4), 13), ("square_add", (0, 5), 5), ("square_add", (-2, 6), 10)]),
        ("cube_sub", "Write a Python function cube_sub(a, b) that returns a cubed minus b.", "def cube_sub(a, b):\n    return a * a * a - b", [("cube_sub", (2, 3), 5), ("cube_sub", (0, 4), -4), ("cube_sub", (-1, 2), -3)]),
        ("half_val", "Write a Python function half_val(n) that returns half of n as a float.", "def half_val(n):\n    return n / 2", [("half_val", (10), 5.0), ("half_val", (0), 0.0), ("half_val", (-6), -3.0)]),
    ])
    families.append(f2)

    # 3. boolean_checks
    f3 = ("boolean_checks", [
        ("is_even", "Write a Python function is_even(n) that returns True if n is even and False otherwise.", "def is_even(n):\n    return n % 2 == 0", [("is_even", (4), True), ("is_even", (7), False), ("is_even", (0), True), ("is_even", (-2), True), ("is_even", (-5), False)]),
        ("is_odd", "Write a Python function is_odd(n) that returns True if n is odd and False otherwise.", "def is_odd(n):\n    return n % 2 != 0", [("is_odd", (3), True), ("is_odd", (8), False), ("is_odd", (0), False), ("is_odd", (-3), True)]),
        ("is_positive", "Write a Python function is_positive(n) that returns True if n is strictly greater than 0.", "def is_positive(n):\n    return n > 0", [("is_positive", (5), True), ("is_positive", (0), False), ("is_positive", (-4), False)]),
        ("is_negative", "Write a Python function is_negative(n) that returns True if n is strictly less than 0.", "def is_negative(n):\n    return n < 0", [("is_negative", (-5), True), ("is_negative", (0), False), ("is_negative", (3), False)]),
        ("is_zero", "Write a Python function is_zero(n) that returns True if n is exactly 0.", "def is_zero(n):\n    return n == 0", [("is_zero", (0), True), ("is_zero", (1), False), ("is_zero", (-1), False)]),
        ("is_non_zero", "Write a Python function is_non_zero(n) that returns True if n is not 0.", "def is_non_zero(n):\n    return n != 0", [("is_non_zero", (5), True), ("is_non_zero", (0), False), ("is_non_zero", (-8), True)]),
        ("is_equal_to", "Write a Python function is_equal_to(a, b) that returns True if a equals b.", "def is_equal_to(a, b):\n    return a == b", [("is_equal_to", (5, 5), True), ("is_equal_to", (4, 7), False), ("is_equal_to", (-2, -2), True)]),
        ("is_strictly_between", "Write a Python function is_strictly_between(x, low, high) that returns True if low < x < high.", "def is_strictly_between(x, low, high):\n    return low < x < high", [("is_strictly_between", (5, 1, 10), True), ("is_strictly_between", (1, 1, 10), False), ("is_strictly_between", (12, 1, 10), False)]),
    ])
    families.append(f3)

    # 4. min_max_bounds
    f4 = ("min_max_bounds", [
        ("min_two", "Write a Python function min_two(a, b) that returns the smaller of a and b.", "def min_two(a, b):\n    return a if a < b else b", [("min_two", (3, 7), 3), ("min_two", (9, 2), 2), ("min_two", (-4, -1), -4)]),
        ("max_two", "Write a Python function max_two(a, b) that returns the larger of a and b.", "def max_two(a, b):\n    return a if a > b else b", [("max_two", (3, 7), 7), ("max_two", (9, 2), 9), ("max_two", (-4, -1), -1)]),
        ("min_three", "Write a Python function min_three(a, b, c) that returns the minimum of three numbers.", "def min_three(a, b, c):\n    return min(a, b, c)", [("min_three", (5, 2, 8), 2), ("min_three", (1, 9, 3), 1), ("min_three", (-1, -5, 0), -5)]),
        ("max_three", "Write a Python function max_three(a, b, c) that returns the maximum of three numbers.", "def max_three(a, b, c):\n    return max(a, b, c)", [("max_three", (5, 2, 8), 8), ("max_three", (1, 9, 3), 9), ("max_three", (-1, -5, 0), 0)]),
        ("clamp", "Write a Python function clamp(val, low, high) that restricts val to the range [low, high].", "def clamp(val, low, high):\n    return low if val < low else (high if val > high else val)", [("clamp", (5, 0, 10), 5), ("clamp", (-3, 0, 10), 0), ("clamp", (15, 0, 10), 10)]),
        ("is_within_bounds", "Write a Python function is_within_bounds(val, low, high) that returns True if low <= val <= high.", "def is_within_bounds(val, low, high):\n    return low <= val <= high", [("is_within_bounds", (5, 0, 10), True), ("is_within_bounds", (0, 0, 10), True), ("is_within_bounds", (-1, 0, 10), False)]),
        ("diff_max_min_two", "Write a Python function diff_max_min_two(a, b) that returns max(a, b) - min(a, b).", "def diff_max_min_two(a, b):\n    return abs(a - b)", [("diff_max_min_two", (10, 4), 6), ("diff_max_min_two", (3, 8), 5), ("diff_max_min_two", (-2, 2), 4)]),
        ("abs_diff", "Write a Python function abs_diff(a, b) that returns the absolute difference between a and b.", "def abs_diff(a, b):\n    return abs(a - b)", [("abs_diff", (5, 9), 4), ("abs_diff", (9, 5), 4), ("abs_diff", (-3, 3), 6)]),
    ])
    families.append(f4)

    # 5. parity_signs
    f5 = ("parity_signs", [
        ("abs_val", "Write a Python function abs_val(n) that returns the absolute value of n.", "def abs_val(n):\n    return abs(n)", [("abs_val", (10), 10), ("abs_val", (0), 0), ("abs_val", (-15), 15)]),
        ("sign_num", "Write a Python function sign_num(n) that returns 1 if n > 0, -1 if n < 0, and 0 if n == 0.", "def sign_num(n):\n    return 1 if n > 0 else (-1 if n < 0 else 0)", [("sign_num", (42), 1), ("sign_num", (-12), -1), ("sign_num", (0), 0)]),
        ("negate_bool", "Write a Python function negate_bool(b) that returns the boolean opposite of b.", "def negate_bool(b):\n    return not b", [("negate_bool", (True), False), ("negate_bool", (False), True)]),
        ("both_true", "Write a Python function both_true(a, b) that returns True only if both a and b are True.", "def both_true(a, b):\n    return a and b", [("both_true", (True, True), True), ("both_true", (True, False), False), ("both_true", (False, False), False)]),
        ("either_true", "Write a Python function either_true(a, b) that returns True if either a or b is True.", "def either_true(a, b):\n    return a or b", [("either_true", (True, False), True), ("either_true", (False, False), False), ("either_true", (True, True), True)]),
        ("neither_true", "Write a Python function neither_true(a, b) that returns True if neither a nor b is True.", "def neither_true(a, b):\n    return not a and not b", [("neither_true", (False, False), True), ("neither_true", (True, False), False)]),
        ("xor_bool", "Write a Python function xor_bool(a, b) that returns True if exactly one of a or b is True.", "def xor_bool(a, b):\n    return a != b", [("xor_bool", (True, False), True), ("xor_bool", (False, True), True), ("xor_bool", (True, True), False), ("xor_bool", (False, False), False)]),
        ("same_sign", "Write a Python function same_sign(a, b) that returns True if both numbers have the same non-zero sign.", "def same_sign(a, b):\n    return (a > 0 and b > 0) or (a < 0 and b < 0)", [("same_sign", (3, 5), True), ("same_sign", (-2, -7), True), ("same_sign", (-2, 4), False), ("same_sign", (0, 5), False)]),
    ])
    families.append(f5)

    # 6. string_inspect
    f6 = ("string_inspect", [
        ("string_len", "Write a Python function string_len(s) that returns the number of characters in string s.", "def string_len(s):\n    return len(s)", [("string_len", ("hello"), 5), ("string_len", (""), 0), ("string_len", ("abc 123"), 7)]),
        ("is_empty_str", "Write a Python function is_empty_str(s) that returns True if s is an empty string.", "def is_empty_str(s):\n    return len(s) == 0", [("is_empty_str", (""), True), ("is_empty_str", ("a"), False), ("is_empty_str", (" "), False)]),
        ("is_non_empty_str", "Write a Python function is_non_empty_str(s) that returns True if s is not empty.", "def is_non_empty_str(s):\n    return len(s) > 0", [("is_non_empty_str", ("hello"), True), ("is_non_empty_str", (""), False)]),
        ("starts_with_char", "Write a Python function starts_with_char(s, ch) that returns True if string s starts with character ch.", "def starts_with_char(s, ch):\n    return s.startswith(ch)", [("starts_with_char", ("apple", "a"), True), ("starts_with_char", ("apple", "b"), False), ("starts_with_char", ("", "a"), False)]),
        ("ends_with_char", "Write a Python function ends_with_char(s, ch) that returns True if string s ends with character ch.", "def ends_with_char(s, ch):\n    return s.endswith(ch)", [("ends_with_char", ("python", "n"), True), ("ends_with_char", ("python", "x"), False), ("ends_with_char", ("", "n"), False)]),
        ("contains_char", "Write a Python function contains_char(s, ch) that returns True if character ch is in string s.", "def contains_char(s, ch):\n    return ch in s", [("contains_char", ("cat", "a"), True), ("contains_char", ("cat", "z"), False), ("contains_char", ("", "a"), False)]),
        ("count_char", "Write a Python function count_char(s, ch) that returns the number of times ch appears in s.", "def count_char(s, ch):\n    return s.count(ch)", [("count_char", ("banana", "a"), 3), ("count_char", ("banana", "b"), 1), ("count_char", ("banana", "z"), 0)]),
        ("first_char", "Write a Python function first_char(s) that returns the first character of non-empty string s.", "def first_char(s):\n    return s[0]", [("first_char", ("hello"), "h"), ("first_char", ("world"), "w"), ("first_char", ("a"), "a")]),
    ])
    families.append(f6)

    # 7. string_transform
    f7 = ("string_transform", [
        ("concat_two", "Write a Python function concat_two(a, b) that concatenates strings a and b.", "def concat_two(a, b):\n    return a + b", [("concat_two", ("foo", "bar"), "foobar"), ("concat_two", ("", "test"), "test"), ("concat_two", ("a", ""), "a")]),
        ("concat_three", "Write a Python function concat_three(a, b, c) that concatenates three strings.", "def concat_three(a, b, c):\n    return a + b + c", [("concat_three", ("x", "y", "z"), "xyz"), ("concat_three", ("a", "-", "b"), "a-b")]),
        ("repeat_str", "Write a Python function repeat_str(s, n) that repeats string s n times.", "def repeat_str(s, n):\n    return s * n", [("repeat_str", ("ha", 3), "hahaha"), ("repeat_str", ("x", 0), ""), ("repeat_str", ("abc", 1), "abc")]),
        ("wrap_brackets", "Write a Python function wrap_brackets(s) that surrounds string s with square brackets.", "def wrap_brackets(s):\n    return f'[{s}]'", [("wrap_brackets", ("item"), "[item]"), ("wrap_brackets", (""), "[]")]),
        ("wrap_quotes", "Write a Python function wrap_quotes(s) that surrounds string s with double quotes.", "def wrap_quotes(s):\n    return f'\"{s}\"'", [("wrap_quotes", ("hello"), '"hello"'), ("wrap_quotes", (""), '""')]),
        ("prepend_str", "Write a Python function prepend_str(prefix, s) that places prefix in front of s.", "def prepend_str(prefix, s):\n    return prefix + s", [("prepend_str", ("pre_", "fix"), "pre_fix"), ("prepend_str", ("", "text"), "text")]),
        ("append_str", "Write a Python function append_str(s, suffix) that appends suffix to the end of s.", "def append_str(s, suffix):\n    return s + suffix", [("append_str", ("run", "_fast"), "run_fast"), ("append_str", ("base", ""), "base")]),
        ("format_greeting", "Write a Python function format_greeting(name) that returns 'Hello, ' followed by name and '!'.", "def format_greeting(name):\n    return f'Hello, {name}!'", [("format_greeting", ("Alice"), "Hello, Alice!"), ("format_greeting", ("Bob"), "Hello, Bob!")]),
    ])
    families.append(f7)

    # 8. string_cases
    f8 = ("string_cases", [
        ("to_upper", "Write a Python function to_upper(s) that returns string s converted to uppercase.", "def to_upper(s):\n    return s.upper()", [("to_upper", ("abc"), "ABC"), ("to_upper", ("Hello"), "HELLO"), ("to_upper", (""), "")]),
        ("to_lower", "Write a Python function to_lower(s) that returns string s converted to lowercase.", "def to_lower(s):\n    return s.lower()", [("to_lower", ("ABC"), "abc"), ("to_lower", ("World"), "world"), ("to_lower", (""), "")]),
        ("is_upper_char", "Write a Python function is_upper_char(ch) that returns True if single char ch is uppercase.", "def is_upper_char(ch):\n    return ch.isupper()", [("is_upper_char", ("A"), True), ("is_upper_char", ("a"), False), ("is_upper_char", ("1"), False)]),
        ("is_lower_char", "Write a Python function is_lower_char(ch) that returns True if single char ch is lowercase.", "def is_lower_char(ch):\n    return ch.islower()", [("is_lower_char", ("z"), True), ("is_lower_char", ("Z"), False), ("is_lower_char", ("!"), False)]),
        ("capitalize_word", "Write a Python function capitalize_word(s) that capitalizes the first character of s.", "def capitalize_word(s):\n    return s.capitalize()", [("capitalize_word", ("apple"), "Apple"), ("capitalize_word", ("BANANA"), "Banana")]),
        ("swap_case_str", "Write a Python function swap_case_str(s) that swaps the case of characters in s.", "def swap_case_str(s):\n    return s.swapcase()", [("swap_case_str", ("aBc"), "AbC"), ("swap_case_str", ("HELLO"), "hello")]),
        ("is_alpha_str", "Write a Python function is_alpha_str(s) that returns True if all characters in non-empty s are alphabetic.", "def is_alpha_str(s):\n    return s.isalpha()", [("is_alpha_str", ("Python"), True), ("is_alpha_str", ("py3"), False), ("is_alpha_str", (""), False)]),
        ("is_digit_str", "Write a Python function is_digit_str(s) that returns True if all characters in non-empty s are digits.", "def is_digit_str(s):\n    return s.isdigit()", [("is_digit_str", ("1234"), True), ("is_digit_str", ("12a"), False), ("is_digit_str", (""), False)]),
    ])
    families.append(f8)

    # 9. list_indexing
    f9 = ("list_indexing", [
        ("first_item", "Write a Python function first_item(lst) that returns the first element of a non-empty list.", "def first_item(lst):\n    return lst[0]", [("first_item", ([10, 20, 30]), 10), ("first_item", (["a", "b"]), "a")]),
        ("last_item", "Write a Python function last_item(lst) that returns the last element of a non-empty list.", "def last_item(lst):\n    return lst[-1]", [("last_item", ([10, 20, 30]), 30), ("last_item", (["x"]), "x")]),
        ("second_item", "Write a Python function second_item(lst) that returns the second element of a list with at least 2 items.", "def second_item(lst):\n    return lst[1]", [("second_item", ([10, 20, 30]), 20), ("second_item", ([1, 2]), 2)]),
        ("item_at_index", "Write a Python function item_at_index(lst, idx) that returns the element at index idx of list lst.", "def item_at_index(lst, idx):\n    return lst[idx]", [("item_at_index", ([10, 20, 30], 2), 30), ("item_at_index", ([5, 6, 7], 0), 5)]),
        ("get_first_two", "Write a Python function get_first_two(lst) that returns a slice containing the first two elements of lst.", "def get_first_two(lst):\n    return lst[:2]", [("get_first_two", ([1, 2, 3, 4]), [1, 2]), ("get_first_two", ([5]), [5])]),
        ("get_last_two", "Write a Python function get_last_two(lst) that returns a slice containing the last two elements of lst.", "def get_last_two(lst):\n    return lst[-2:]", [("get_last_two", ([1, 2, 3, 4]), [3, 4]), ("get_last_two", ([9, 8]), [9, 8])]),
        ("swap_two_items", "Write a Python function swap_two_items(lst) that returns a new list swapping the first two elements of a 2-item list.", "def swap_two_items(lst):\n    return [lst[1], lst[0]]", [("swap_two_items", ([1, 2]), [2, 1]), ("swap_two_items", (["a", "b"]), ["b", "a"])]),
        ("penultimate_item", "Write a Python function penultimate_item(lst) that returns the second to last item of list lst.", "def penultimate_item(lst):\n    return lst[-2]", [("penultimate_item", ([10, 20, 30]), 20), ("penultimate_item", ([1, 2]), 1)]),
    ])
    families.append(f9)

    # 10. list_measuring
    f10 = ("list_measuring", [
        ("list_len", "Write a Python function list_len(lst) that returns the number of items in list lst.", "def list_len(lst):\n    return len(lst)", [("list_len", ([1, 2, 3]), 3), ("list_len", ([]), 0), ("list_len", ([9]), 1)]),
        ("is_list_empty", "Write a Python function is_list_empty(lst) that returns True if list lst is empty.", "def is_list_empty(lst):\n    return len(lst) == 0", [("is_list_empty", ([]), True), ("is_list_empty", ([1]), False)]),
        ("is_list_non_empty", "Write a Python function is_list_non_empty(lst) that returns True if list lst contains at least one item.", "def is_list_non_empty(lst):\n    return len(lst) > 0", [("is_list_non_empty", ([1, 2]), True), ("is_list_non_empty", ([]), False)]),
        ("sum_two_items", "Write a Python function sum_two_items(lst) that returns the sum of the first two elements in lst.", "def sum_two_items(lst):\n    return lst[0] + lst[1]", [("sum_two_items", ([3, 7]), 10), ("sum_two_items", ([-2, 5]), 3)]),
        ("sum_three_items", "Write a Python function sum_three_items(lst) that returns the sum of a 3-element list.", "def sum_three_items(lst):\n    return lst[0] + lst[1] + lst[2]", [("sum_three_items", ([1, 2, 3]), 6), ("sum_three_items", ([-1, 0, 1]), 0)]),
        ("sum_list", "Write a Python function sum_list(lst) that returns the sum of all elements in list lst.", "def sum_list(lst):\n    return sum(lst)", [("sum_list", ([1, 2, 3, 4]), 10), ("sum_list", ([]), 0), ("sum_list", ([-5, 5]), 0)]),
        ("count_in_list", "Write a Python function count_in_list(lst, val) that returns how many times val appears in lst.", "def count_in_list(lst, val):\n    return lst.count(val)", [("count_in_list", ([1, 2, 2, 3], 2), 2), ("count_in_list", ([1, 2, 3], 9), 0), ("count_in_list", ([], 1), 0)]),
        ("has_single_item", "Write a Python function has_single_item(lst) that returns True if lst has length exactly 1.", "def has_single_item(lst):\n    return len(lst) == 1", [("has_single_item", ([5]), True), ("has_single_item", ([]), False), ("has_single_item", ([1, 2]), False)]),
    ])
    families.append(f10)

    # 11. list_min_max
    f11 = ("list_min_max", [
        ("list_min", "Write a Python function list_min(lst) that returns the minimum value in non-empty list lst.", "def list_min(lst):\n    return min(lst)", [("list_min", ([3, 1, 4]), 1), ("list_min", ([-5, 2, 0]), -5), ("list_min", ([9]), 9)]),
        ("list_max", "Write a Python function list_max(lst) that returns the maximum value in non-empty list lst.", "def list_max(lst):\n    return max(lst)", [("list_max", ([3, 1, 4]), 4), ("list_max", ([-5, -2, -9]), -2), ("list_max", ([7]), 7)]),
        ("list_span", "Write a Python function list_span(lst) that returns max(lst) - min(lst) for non-empty list lst.", "def list_span(lst):\n    return max(lst) - min(lst)", [("list_span", ([1, 5, 10]), 9), ("list_span", ([4, 4]), 0), ("list_span", ([-2, 6]), 8)]),
        ("min_index_two", "Write a Python function min_index_two(lst) that returns 0 if lst[0] <= lst[1] else 1 for a 2-item list.", "def min_index_two(lst):\n    return 0 if lst[0] <= lst[1] else 1", [("min_index_two", ([3, 7]), 0), ("min_index_two", ([8, 2]), 1), ("min_index_two", ([4, 4]), 0)]),
        ("max_index_two", "Write a Python function max_index_two(lst) that returns 0 if lst[0] >= lst[1] else 1 for a 2-item list.", "def max_index_two(lst):\n    return 0 if lst[0] >= lst[1] else 1", [("max_index_two", ([3, 7]), 1), ("max_index_two", ([8, 2]), 0), ("max_index_two", ([4, 4]), 0)]),
        ("is_first_min", "Write a Python function is_first_min(lst) that returns True if the first element is the minimum of lst.", "def is_first_min(lst):\n    return lst[0] == min(lst)", [("is_first_min", ([1, 2, 3]), True), ("is_first_min", ([3, 1, 2]), False)]),
        ("is_last_max", "Write a Python function is_last_max(lst) that returns True if the last element is the maximum of lst.", "def is_last_max(lst):\n    return lst[-1] == max(lst)", [("is_last_max", ([1, 2, 5]), True), ("is_last_max", ([5, 2, 1]), False)]),
        ("abs_max", "Write a Python function abs_max(lst) that returns the maximum of absolute values in non-empty list lst.", "def abs_max(lst):\n    return max(abs(x) for x in lst)", [("abs_max", ([1, -7, 3]), 7), ("abs_max", ([0, 4]), 4), ("abs_max", ([-10, 10]), 10)]),
    ])
    families.append(f11)

    # 12. list_mutations
    f12 = ("list_mutations", [
        ("append_item", "Write a Python function append_item(lst, x) that returns a new list with x added to the end of lst.", "def append_item(lst, x):\n    return lst + [x]", [("append_item", ([1, 2], 3), [1, 2, 3]), ("append_item", ([], "a"), ["a"])]),
        ("prepend_item", "Write a Python function prepend_item(lst, x) that returns a new list with x added to the front of lst.", "def prepend_item(lst, x):\n    return [x] + lst", [("prepend_item", ([2, 3], 1), [1, 2, 3]), ("prepend_item", ([], 9), [9])]),
        ("reverse_two", "Write a Python function reverse_two(lst) that returns a new reversed list from a 2-item list.", "def reverse_two(lst):\n    return [lst[1], lst[0]]", [("reverse_two", ([4, 8]), [8, 4]), ("reverse_two", (["x", "y"]), ["y", "x"])]),
        ("reverse_three", "Write a Python function reverse_three(lst) that returns a new reversed list from a 3-item list.", "def reverse_three(lst):\n    return [lst[2], lst[1], lst[0]]", [("reverse_three", ([1, 2, 3]), [3, 2, 1]), ("reverse_three", (["a", "b", "c"]), ["c", "b", "a"])]),
        ("drop_first_item", "Write a Python function drop_first_item(lst) that returns a list containing all elements except the first.", "def drop_first_item(lst):\n    return lst[1:]", [("drop_first_item", ([1, 2, 3]), [2, 3]), ("drop_first_item", ([5]), [])]),
        ("drop_last_item", "Write a Python function drop_last_item(lst) that returns a list containing all elements except the last.", "def drop_last_item(lst):\n    return lst[:-1]", [("drop_last_item", ([1, 2, 3]), [1, 2]), ("drop_last_item", ([7]), [])]),
        ("double_items_two", "Write a Python function double_items_two(lst) that doubles each element in a 2-item integer list.", "def double_items_two(lst):\n    return [lst[0] * 2, lst[1] * 2]", [("double_items_two", ([2, 5]), [4, 10]), ("double_items_two", ([0, -3]), [0, -6])]),
        ("repeat_list_twice", "Write a Python function repeat_list_twice(lst) that returns list lst concatenated with itself.", "def repeat_list_twice(lst):\n    return lst + lst", [("repeat_list_twice", ([1, 2]), [1, 2, 1, 2]), ("repeat_list_twice", ([]), [])]),
    ])
    families.append(f12)

    # 13. list_predicates
    f13 = ("list_predicates", [
        ("has_item", "Write a Python function has_item(lst, x) that returns True if x is in list lst.", "def has_item(lst, x):\n    return x in lst", [("has_item", ([1, 2, 3], 2), True), ("has_item", ([1, 2, 3], 9), False), ("has_item", ([], 5), False)]),
        ("all_positive", "Write a Python function all_positive(lst) that returns True if all elements in lst are strictly greater than 0.", "def all_positive(lst):\n    return all(x > 0 for x in lst)", [("all_positive", ([1, 2, 3]), True), ("all_positive", ([1, -2, 3]), False), ("all_positive", ([]), True)]),
        ("any_negative", "Write a Python function any_negative(lst) that returns True if any element in lst is strictly less than 0.", "def any_negative(lst):\n    return any(x < 0 for x in lst)", [("any_negative", ([1, -2, 3]), True), ("any_negative", ([1, 2, 3]), False), ("any_negative", ([]), False)]),
        ("all_zeros", "Write a Python function all_zeros(lst) that returns True if all elements in non-empty lst are 0.", "def all_zeros(lst):\n    return len(lst) > 0 and all(x == 0 for x in lst)", [("all_zeros", ([0, 0, 0]), True), ("all_zeros", ([0, 1]), False), ("all_zeros", ([]), False)]),
        ("is_len_two", "Write a Python function is_len_two(lst) that returns True if list lst has length exactly 2.", "def is_len_two(lst):\n    return len(lst) == 2", [("is_len_two", ([1, 2]), True), ("is_len_two", ([1]), False), ("is_len_two", ([1, 2, 3]), False)]),
        ("is_len_three", "Write a Python function is_len_three(lst) that returns True if list lst has length exactly 3.", "def is_len_three(lst):\n    return len(lst) == 3", [("is_len_three", ([1, 2, 3]), True), ("is_len_three", ([1, 2]), False)]),
        ("first_equals_last", "Write a Python function first_equals_last(lst) that returns True if lst is non-empty and lst[0] == lst[-1].", "def first_equals_last(lst):\n    return len(lst) > 0 and lst[0] == lst[-1]", [("first_equals_last", ([1, 2, 1]), True), ("first_equals_last", ([1, 2, 3]), False), ("first_equals_last", ([]), False)]),
        ("are_items_unique_two", "Write a Python function are_items_unique_two(lst) that returns True if a 2-item list has distinct items.", "def are_items_unique_two(lst):\n    return lst[0] != lst[1]", [("are_items_unique_two", ([1, 2]), True), ("are_items_unique_two", ([5, 5]), False)]),
    ])
    families.append(f13)

    # 14. dict_lookups
    f14 = ("dict_lookups", [
        ("get_val", "Write a Python function get_val(d, k) that returns the value associated with key k in dictionary d.", "def get_val(d, k):\n    return d[k]", [("get_val", ({"a": 1, "b": 2}, "a"), 1), ("get_val", ({"x": 10}, "x"), 10)]),
        ("has_key", "Write a Python function has_key(d, k) that returns True if key k exists in dictionary d.", "def has_key(d, k):\n    return k in d", [("has_key", ({"a": 1}, "a"), True), ("has_key", ({"a": 1}, "b"), False), ("has_key", ({}, "a"), False)]),
        ("dict_len", "Write a Python function dict_len(d) that returns the number of keys in dictionary d.", "def dict_len(d):\n    return len(d)", [("dict_len", ({"a": 1, "b": 2}), 2), ("dict_len", ({}), 0)]),
        ("is_dict_empty", "Write a Python function is_dict_empty(d) that returns True if dictionary d is empty.", "def is_dict_empty(d):\n    return len(d) == 0", [("is_dict_empty", ({}), True), ("is_dict_empty", ({"a": 1}), False)]),
        ("get_val_default", "Write a Python function get_val_default(d, k, default) that returns d.get(k, default).", "def get_val_default(d, k, default):\n    return d.get(k, default)", [("get_val_default", ({"a": 1}, "a", 0), 1), ("get_val_default", ({"a": 1}, "b", 99), 99)]),
        ("keys_count_two", "Write a Python function keys_count_two(d) that returns True if dictionary d has exactly 2 keys.", "def keys_count_two(d):\n    return len(d) == 2", [("keys_count_two", ({"a": 1, "b": 2}), True), ("keys_count_two", ({"a": 1}), False)]),
        ("key_in_both", "Write a Python function key_in_both(d1, d2, k) that returns True if key k is in both dictionaries d1 and d2.", "def key_in_both(d1, d2, k):\n    return k in d1 and k in d2", [("key_in_both", ({"a": 1}, {"a": 2}, "a"), True), ("key_in_both", ({"a": 1}, {"b": 2}, "a"), False)]),
        ("first_dict_key", "Write a Python function first_dict_key(d) that returns the first key in a non-empty dictionary d.", "def first_dict_key(d):\n    return list(d.keys())[0]", [("first_dict_key", ({"name": "Alice"}), "name"), ("first_dict_key", ({"id": 1, "val": 2}), "id")]),
    ])
    families.append(f14)

    # 15. dict_key_vals
    f15 = ("dict_key_vals", [
        ("extract_keys", "Write a Python function extract_keys(d) that returns the list of keys of dictionary d.", "def extract_keys(d):\n    return list(d.keys())", [("extract_keys", ({"a": 1, "b": 2}), ["a", "b"]), ("extract_keys", ({}), [])]),
        ("extract_vals", "Write a Python function extract_vals(d) that returns the list of values of dictionary d.", "def extract_vals(d):\n    return list(d.values())", [("extract_vals", ({"a": 1, "b": 2}), [1, 2]), ("extract_vals", ({}), [])]),
        ("add_pair", "Write a Python function add_pair(d, k, v) that returns a new dictionary with key k and value v added.", "def add_pair(d, k, v):\n    res = dict(d)\n    res[k] = v\n    return res", [("add_pair", ({"a": 1}, "b", 2), {"a": 1, "b": 2}), ("add_pair", ({}, "x", 10), {"x": 10})]),
        ("has_value", "Write a Python function has_value(d, v) that returns True if value v is present in dictionary d.", "def has_value(d, v):\n    return v in d.values()", [("has_value", ({"a": 10, "b": 20}, 20), True), ("has_value", ({"a": 10}, 99), False)]),
        ("sum_dict_vals", "Write a Python function sum_dict_vals(d) that returns the sum of all values in dictionary d.", "def sum_dict_vals(d):\n    return sum(d.values())", [("sum_dict_vals", ({"a": 10, "b": 20}), 30), ("sum_dict_vals", ({}), 0)]),
        ("max_dict_val", "Write a Python function max_dict_val(d) that returns the maximum value in a non-empty dictionary d.", "def max_dict_val(d):\n    return max(d.values())", [("max_dict_val", ({"a": 10, "b": 50, "c": 20}), 50), ("max_dict_val", ({"x": -5}), -5)]),
        ("dict_has_matching_kv", "Write a Python function dict_has_matching_kv(d, k, v) that returns True if d[k] == v.", "def dict_has_matching_kv(d, k, v):\n    return d.get(k) == v", [("dict_has_matching_kv", ({"a": 1}, "a", 1), True), ("dict_has_matching_kv", ({"a": 1}, "a", 2), False)]),
        ("invert_pair_dict", "Write a Python function invert_pair_dict(d) that swaps key and value for a 1-item dictionary.", "def invert_pair_dict(d):\n    k = list(d.keys())[0]\n    return {d[k]: k}", [("invert_pair_dict", ({"a": "b"}), {"b": "a"}), ("invert_pair_dict", ({"hello": "world"}), {"world": "hello"})]),
    ])
    families.append(f15)

    # 16. geometry_math
    f16 = ("geometry_math", [
        ("rect_area", "Write a Python function rect_area(w, h) that returns the area of a rectangle with width w and height h.", "def rect_area(w, h):\n    return w * h", [("rect_area", (4, 5), 20), ("rect_area", (0, 10), 0), ("rect_area", (7, 3), 21)]),
        ("rect_perimeter", "Write a Python function rect_perimeter(w, h) that returns the perimeter of a rectangle.", "def rect_perimeter(w, h):\n    return 2 * (w + h)", [("rect_perimeter", (4, 5), 18), ("rect_perimeter", (10, 2), 24)]),
        ("square_area", "Write a Python function square_area(side) that returns the area of a square with side length side.", "def square_area(side):\n    return side * side", [("square_area", (6), 36), ("square_area", (0), 0)]),
        ("square_perimeter", "Write a Python function square_perimeter(side) that returns the perimeter of a square.", "def square_perimeter(side):\n    return 4 * side", [("square_perimeter", (5), 20), ("square_perimeter", (8), 32)]),
        ("triangle_area", "Write a Python function triangle_area(base, height) that returns the area of a triangle (base * height / 2).", "def triangle_area(base, height):\n    return base * height / 2", [("triangle_area", (10, 4), 20.0), ("triangle_area", (6, 5), 15.0)]),
        ("box_volume", "Write a Python function box_volume(l, w, h) that returns the volume of a rectangular box.", "def box_volume(l, w, h):\n    return l * w * h", [("box_volume", (2, 3, 4), 24), ("box_volume", (5, 5, 2), 50)]),
        ("circle_circumference_approx", "Write a Python function circle_circumference_approx(r) that returns 2 * 3.14 * r.", "def circle_circumference_approx(r):\n    return 2 * 3.14 * r", [("circle_circumference_approx", (10), 62.8), ("circle_circumference_approx", (0), 0.0)]),
        ("circle_area_approx", "Write a Python function circle_area_approx(r) that returns 3.14 * r * r.", "def circle_area_approx(r):\n    return 3.14 * r * r", [("circle_area_approx", (10), 314.0), ("circle_area_approx", (0), 0.0)]),
    ])
    families.append(f16)

    # 17. unit_conversions
    f17 = ("unit_conversions", [
        ("c_to_f", "Write a Python function c_to_f(c) that converts Celsius to Fahrenheit using (c * 9/5) + 32.", "def c_to_f(c):\n    return (c * 9 / 5) + 32", [("c_to_f", (0), 32.0), ("c_to_f", (100), 212.0), ("c_to_f", (-40), -40.0)]),
        ("f_to_c", "Write a Python function f_to_c(f) that converts Fahrenheit to Celsius using (f - 32) * 5/9.", "def f_to_c(f):\n    return (f - 32) * 5 / 9", [("f_to_c", (32), 0.0), ("f_to_c", (212), 100.0), ("f_to_c", (-40), -40.0)]),
        ("km_to_m", "Write a Python function km_to_m(km) that converts kilometers to meters (km * 1000).", "def km_to_m(km):\n    return km * 1000", [("km_to_m", (5), 5000), ("km_to_m", (0), 0), ("km_to_m", (2.5), 2500.0)]),
        ("m_to_cm", "Write a Python function m_to_cm(m) that converts meters to centimeters (m * 100).", "def m_to_cm(m):\n    return m * 100", [("m_to_cm", (3), 300), ("m_to_cm", (0.5), 50.0)]),
        ("hours_to_mins", "Write a Python function hours_to_mins(h) that converts hours to minutes (h * 60).", "def hours_to_mins(h):\n    return h * 60", [("hours_to_mins", (2), 120), ("hours_to_mins", (0), 0), ("hours_to_mins", (1.5), 90.0)]),
        ("mins_to_secs", "Write a Python function mins_to_secs(m) that converts minutes to seconds (m * 60).", "def mins_to_secs(m):\n    return m * 60", [("mins_to_secs", (5), 300), ("mins_to_secs", (1), 60)]),
        ("days_to_hours", "Write a Python function days_to_hours(d) that converts days to hours (d * 24).", "def days_to_hours(d):\n    return d * 24", [("days_to_hours", (3), 72), ("days_to_hours", (1), 24)]),
        ("weeks_to_days", "Write a Python function weeks_to_days(w) that converts weeks to days (w * 7).", "def weeks_to_days(w):\n    return w * 7", [("weeks_to_days", (2), 14), ("weeks_to_days", (0), 0)]),
    ])
    families.append(f17)

    # 18. pair_math
    f18 = ("pair_math", [
        ("swap_pair", "Write a Python function swap_pair(p) that returns a new tuple (p[1], p[0]) swapping elements of pair p.", "def swap_pair(p):\n    return (p[1], p[0])", [("swap_pair", ((1, 2)), (2, 1)), ("swap_pair", (("a", "b")), ("b", "a"))]),
        ("pair_sum", "Write a Python function pair_sum(p) that returns the sum p[0] + p[1] of tuple p.", "def pair_sum(p):\n    return p[0] + p[1]", [("pair_sum", ((4, 6)), 10), ("pair_sum", ((-3, 3)), 0)]),
        ("pair_product", "Write a Python function pair_product(p) that returns the product p[0] * p[1] of tuple p.", "def pair_product(p):\n    return p[0] * p[1]", [("pair_product", ((5, 6)), 30), ("pair_product", ((-2, 4)), -8)]),
        ("pair_diff", "Write a Python function pair_diff(p) that returns p[0] - p[1] of tuple p.", "def pair_diff(p):\n    return p[0] - p[1]", [("pair_diff", ((10, 4)), 6), ("pair_diff", ((3, 9)), -6)]),
        ("pair_distance", "Write a Python function pair_distance(p) that returns abs(p[0] - p[1]) of tuple p.", "def pair_distance(p):\n    return abs(p[0] - p[1])", [("pair_distance", ((10, 4)), 6), ("pair_distance", ((3, 9)), 6)]),
        ("pair_midpoint", "Write a Python function pair_midpoint(p) that returns (p[0] + p[1]) / 2 for pair p.", "def pair_midpoint(p):\n    return (p[0] + p[1]) / 2", [("pair_midpoint", ((2, 6)), 4.0), ("pair_midpoint", ((0, 10)), 5.0)]),
        ("pair_has_zero", "Write a Python function pair_has_zero(p) that returns True if either element of pair p is 0.", "def pair_has_zero(p):\n    return p[0] == 0 or p[1] == 0", [("pair_has_zero", ((0, 5)), True), ("pair_has_zero", ((4, 0)), True), ("pair_has_zero", ((3, 4)), False)]),
        ("is_pair_equal", "Write a Python function is_pair_equal(p) that returns True if p[0] == p[1].", "def is_pair_equal(p):\n    return p[0] == p[1]", [("is_pair_equal", ((5, 5)), True), ("is_pair_equal", ((4, 7)), False)]),
    ])
    families.append(f18)

    # 19. counting_filters
    f19 = ("counting_filters", [
        ("count_evens", "Write a Python function count_evens(lst) that returns the number of even integers in lst.", "def count_evens(lst):\n    return sum(1 for x in lst if x % 2 == 0)", [("count_evens", ([1, 2, 3, 4, 6]), 3), ("count_evens", ([1, 3, 5]), 0), ("count_evens", ([]), 0)]),
        ("count_odds", "Write a Python function count_odds(lst) that returns the number of odd integers in lst.", "def count_odds(lst):\n    return sum(1 for x in lst if x % 2 != 0)", [("count_odds", ([1, 2, 3, 4]), 2), ("count_odds", ([2, 4]), 0)]),
        ("count_positives", "Write a Python function count_positives(lst) that returns how many numbers in lst are strictly greater than 0.", "def count_positives(lst):\n    return sum(1 for x in lst if x > 0)", [("count_positives", ([1, -2, 3, 0]), 2), ("count_positives", ([-1, -2]), 0)]),
        ("count_negatives", "Write a Python function count_negatives(lst) that returns how many numbers in lst are strictly less than 0.", "def count_negatives(lst):\n    return sum(1 for x in lst if x < 0)", [("count_negatives", ([1, -2, -3, 0]), 2), ("count_negatives", ([1, 2]), 0)]),
        ("count_zeros", "Write a Python function count_zeros(lst) that returns how many numbers in lst are exactly 0.", "def count_zeros(lst):\n    return sum(1 for x in lst if x == 0)", [("count_zeros", ([0, 1, 0, 2]), 2), ("count_zeros", ([1, 2, 3]), 0)]),
        ("count_greater_than", "Write a Python function count_greater_than(lst, val) that counts items in lst strictly greater than val.", "def count_greater_than(lst, val):\n    return sum(1 for x in lst if x > val)", [("count_greater_than", ([1, 5, 8, 2], 4), 2), ("count_greater_than", ([1, 2], 5), 0)]),
        ("count_less_than", "Write a Python function count_less_than(lst, val) that counts items in lst strictly less than val.", "def count_less_than(lst, val):\n    return sum(1 for x in lst if x < val)", [("count_less_than", ([1, 5, 8, 2], 4), 2), ("count_less_than", ([5, 8], 2), 0)]),
        ("count_equal_to", "Write a Python function count_equal_to(lst, val) that counts occurrences of val in lst.", "def count_equal_to(lst, val):\n    return sum(1 for x in lst if x == val)", [("count_equal_to", ([4, 2, 4, 4], 4), 3), ("count_equal_to", ([1, 2], 9), 0)]),
    ])
    families.append(f19)

    # 20. sequence_orders
    f20 = ("sequence_orders", [
        ("is_asc_two", "Write a Python function is_asc_two(lst) that returns True if lst[0] <= lst[1] in a 2-item list.", "def is_asc_two(lst):\n    return lst[0] <= lst[1]", [("is_asc_two", ([2, 5]), True), ("is_asc_two", ([5, 2]), False), ("is_asc_two", ([3, 3]), True)]),
        ("is_desc_two", "Write a Python function is_desc_two(lst) that returns True if lst[0] >= lst[1] in a 2-item list.", "def is_desc_two(lst):\n    return lst[0] >= lst[1]", [("is_desc_two", ([5, 2]), True), ("is_desc_two", ([2, 5]), False), ("is_desc_two", ([3, 3]), True)]),
        ("is_asc_three", "Write a Python function is_asc_three(lst) that returns True if lst[0] <= lst[1] <= lst[2] for a 3-item list.", "def is_asc_three(lst):\n    return lst[0] <= lst[1] <= lst[2]", [("is_asc_three", ([1, 2, 3]), True), ("is_asc_three", ([1, 3, 2]), False)]),
        ("is_desc_three", "Write a Python function is_desc_three(lst) that returns True if lst[0] >= lst[1] >= lst[2] for a 3-item list.", "def is_desc_three(lst):\n    return lst[0] >= lst[1] >= lst[2]", [("is_desc_three", ([3, 2, 1]), True), ("is_desc_three", ([1, 2, 3]), False)]),
        ("are_equal_two", "Write a Python function are_equal_two(lst) that returns True if lst[0] == lst[1] in a 2-item list.", "def are_equal_two(lst):\n    return lst[0] == lst[1]", [("are_equal_two", ([7, 7]), True), ("are_equal_two", ([7, 8]), False)]),
        ("are_equal_three", "Write a Python function are_equal_three(lst) that returns True if lst[0] == lst[1] == lst[2] for a 3-item list.", "def are_equal_three(lst):\n    return lst[0] == lst[1] == lst[2]", [("are_equal_three", ([4, 4, 4]), True), ("are_equal_three", ([4, 4, 5]), False)]),
        ("are_distinct_two", "Write a Python function are_distinct_two(lst) that returns True if lst[0] != lst[1] in a 2-item list.", "def are_distinct_two(lst):\n    return lst[0] != lst[1]", [("are_distinct_two", ([1, 2]), True), ("are_distinct_two", ([3, 3]), False)]),
        ("are_distinct_three", "Write a Python function are_distinct_three(lst) that returns True if all 3 elements of lst are unique.", "def are_distinct_three(lst):\n    return lst[0] != lst[1] and lst[1] != lst[2] and lst[0] != lst[2]", [("are_distinct_three", ([1, 2, 3]), True), ("are_distinct_three", ([1, 2, 1]), False)]),
    ])
    families.append(f20)

    # 21. set_primitives (VALIDATION)
    f21 = ("set_primitives", [
        ("set_size", "Write a Python function set_size(s) that returns the number of elements in set s.", "def set_size(s):\n    return len(s)", [("set_size", ({1, 2, 3}), 3), ("set_size", (set()), 0)]),
        ("set_has", "Write a Python function set_has(s, x) that returns True if element x is in set s.", "def set_has(s, x):\n    return x in s", [("set_has", ({1, 2}, 1), True), ("set_has", ({1, 2}, 9), False)]),
        ("set_union_size", "Write a Python function set_union_size(s1, s2) that returns the number of elements in s1 union s2.", "def set_union_size(s1, s2):\n    return len(s1 | s2)", [("set_union_size", ({1, 2}, {2, 3}), 3), ("set_union_size", (set(), {1}), 1)]),
        ("set_inter_size", "Write a Python function set_inter_size(s1, s2) that returns the number of elements in s1 intersection s2.", "def set_inter_size(s1, s2):\n    return len(s1 & s2)", [("set_inter_size", ({1, 2}, {2, 3}), 1), ("set_inter_size", ({1}, {2}), 0)]),
        ("set_diff_size", "Write a Python function set_diff_size(s1, s2) that returns the number of elements in s1 minus s2.", "def set_diff_size(s1, s2):\n    return len(s1 - s2)", [("set_diff_size", ({1, 2, 3}, {2}), 2), ("set_diff_size", ({1}, {1}), 0)]),
        ("is_subset_of", "Write a Python function is_subset_of(s1, s2) that returns True if s1 is a subset of s2.", "def is_subset_of(s1, s2):\n    return s1.issubset(s2)", [("is_subset_of", ({1}, {1, 2}), True), ("is_subset_of", ({1, 3}, {1, 2}), False)]),
        ("are_disjoint", "Write a Python function are_disjoint(s1, s2) that returns True if s1 and s2 share no elements.", "def are_disjoint(s1, s2):\n    return s1.isdisjoint(s2)", [("are_disjoint", ({1, 2}, {3, 4}), True), ("are_disjoint", ({1, 2}, {2, 3}), False)]),
        ("is_set_empty", "Write a Python function is_set_empty(s) that returns True if set s has 0 elements.", "def is_set_empty(s):\n    return len(s) == 0", [("is_set_empty", (set()), True), ("is_set_empty", ({1}), False)]),
    ])
    families.append(f21)

    # 22. advanced_str_checks (VALIDATION)
    f22 = ("advanced_str_checks", [
        ("is_palindrome_simple", "Write a Python function is_palindrome_simple(s) that returns True if string s equals its reverse.", "def is_palindrome_simple(s):\n    return s == s[::-1]", [("is_palindrome_simple", ("radar"), True), ("is_palindrome_simple", ("hello"), False), ("is_palindrome_simple", (""), True)]),
        ("count_vowels_basic", "Write a Python function count_vowels_basic(s) that counts occurrences of 'a','e','i','o','u' in lowercase s.", "def count_vowels_basic(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')", [("count_vowels_basic", ("apple"), 2), ("count_vowels_basic", ("sky"), 0), ("count_vowels_basic", ("EDUCATION"), 5)]),
        ("has_whitespace", "Write a Python function has_whitespace(s) that returns True if string s contains any whitespace characters.", "def has_whitespace(s):\n    return ' ' in s", [("has_whitespace", ("hello world"), True), ("has_whitespace", ("nospace"), False)]),
        ("remove_spaces_simple", "Write a Python function remove_spaces_simple(s) that returns string s with all spaces removed.", "def remove_spaces_simple(s):\n    return s.replace(' ', '')", [("remove_spaces_simple", ("a b c"), "abc"), ("remove_spaces_simple", ("test"), "test")]),
        ("count_words_space", "Write a Python function count_words_space(s) that returns the number of words in s split by whitespace.", "def count_words_space(s):\n    return len(s.split())", [("count_words_space", ("one two three"), 3), ("count_words_space", (""), 0)]),
        ("join_with_dash", "Write a Python function join_with_dash(lst) that joins a list of strings with a hyphen '-'.", "def join_with_dash(lst):\n    return '-'.join(lst)", [("join_with_dash", (["a", "b", "c"]), "a-b-c"), ("join_with_dash", (["single"]), "single")]),
        ("join_with_space", "Write a Python function join_with_space(lst) that joins a list of strings with a single space.", "def join_with_space(lst):\n    return ' '.join(lst)", [("join_with_space", (["hello", "world"]), "hello world"), ("join_with_space", ([]), "")]),
        ("reverse_string", "Write a Python function reverse_string(s) that returns string s in reverse order.", "def reverse_string(s):\n    return s[::-1]", [("reverse_string", ("abc"), "cba"), ("reverse_string", (""), "")]),
    ])
    families.append(f22)

    # 23. list_subsets (VALIDATION)
    f23 = ("list_subsets", [
        ("take_n", "Write a Python function take_n(lst, n) that returns the first n elements of list lst.", "def take_n(lst, n):\n    return lst[:n]", [("take_n", ([1, 2, 3, 4], 2), [1, 2]), ("take_n", ([1, 2], 5), [1, 2])]),
        ("drop_n", "Write a Python function drop_n(lst, n) that returns the list lst after dropping the first n elements.", "def drop_n(lst, n):\n    return lst[n:]", [("drop_n", ([1, 2, 3, 4], 2), [3, 4]), ("drop_n", ([1, 2], 5), [])]),
        ("slice_range", "Write a Python function slice_range(lst, start, end) that returns lst[start:end].", "def slice_range(lst, start, end):\n    return lst[start:end]", [("slice_range", ([10, 20, 30, 40, 50], 1, 4), [20, 30, 40]), ("slice_range", ([1, 2], 0, 1), [1])]),
        ("middle_item_three", "Write a Python function middle_item_three(lst) that returns the middle element of a 3-item list.", "def middle_item_three(lst):\n    return lst[1]", [("middle_item_three", ([10, 20, 30]), 20), ("middle_item_three", (["a", "b", "c"]), "b")]),
        ("repeat_elements_two", "Write a Python function repeat_elements_two(lst) that duplicates each item of a 2-item list: [x, x, y, y].", "def repeat_elements_two(lst):\n    return [lst[0], lst[0], lst[1], lst[1]]", [("repeat_elements_two", ([1, 2]), [1, 1, 2, 2]), ("repeat_elements_two", (["a", "b"]), ["a", "a", "b", "b"])]),
        ("are_lists_equal", "Write a Python function are_lists_equal(l1, l2) that returns True if list l1 equals list l2.", "def are_lists_equal(l1, l2):\n    return l1 == l2", [("are_lists_equal", ([1, 2], [1, 2]), True), ("are_lists_equal", ([1, 2], [2, 1]), False)]),
        ("combine_lists", "Write a Python function combine_lists(l1, l2) that concatenates list l1 and list l2.", "def combine_lists(l1, l2):\n    return l1 + l2", [("combine_lists", ([1, 2], [3, 4]), [1, 2, 3, 4]), ("combine_lists", ([], [5]), [5])]),
        ("get_edges", "Write a Python function get_edges(lst) that returns a list containing only the first and last elements of lst (len >= 2).", "def get_edges(lst):\n    return [lst[0], lst[-1]]", [("get_edges", ([1, 2, 3, 4]), [1, 4]), ("get_edges", ([9, 8]), [9, 8])]),
    ])
    families.append(f23)

    # 24. division_remainders (VALIDATION)
    f24 = ("division_remainders", [
        ("div_rem_tuple", "Write a Python function div_rem_tuple(a, b) that returns a tuple (a // b, a % b).", "def div_rem_tuple(a, b):\n    return (a // b, a % b)", [("div_rem_tuple", (14, 3), (4, 2)), ("div_rem_tuple", (10, 5), (2, 0))]),
        ("is_divisible_by", "Write a Python function is_divisible_by(a, b) that returns True if a is evenly divisible by b.", "def is_divisible_by(a, b):\n    return a % b == 0", [("is_divisible_by", (15, 5), True), ("is_divisible_by", (14, 5), False)]),
        ("is_multiple_of_ten", "Write a Python function is_multiple_of_ten(n) that returns True if n is a multiple of 10.", "def is_multiple_of_ten(n):\n    return n % 10 == 0", [("is_multiple_of_ten", (100), True), ("is_multiple_of_ten", (45), False), ("is_multiple_of_ten", (0), True)]),
        ("next_multiple_of_five", "Write a Python function next_multiple_of_five(n) that returns n if n % 5 == 0 else n + (5 - (n % 5)).", "def next_multiple_of_five(n):\n    return n if n % 5 == 0 else n + (5 - (n % 5))", [("next_multiple_of_five", (10), 10), ("next_multiple_of_five", (12), 15), ("next_multiple_of_five", (16), 20)]),
        ("int_average_two", "Write a Python function int_average_two(a, b) that returns the integer average (a + b) // 2.", "def int_average_two(a, b):\n    return (a + b) // 2", [("int_average_two", (10, 20), 15), ("int_average_two", (5, 6), 5)]),
        ("last_digit", "Write a Python function last_digit(n) that returns the last digit abs(n) % 10.", "def last_digit(n):\n    return abs(n) % 10", [("last_digit", (147), 7), ("last_digit", (-29), 9), ("last_digit", (0), 0)]),
        ("remove_last_digit", "Write a Python function remove_last_digit(n) that returns n // 10 for positive n.", "def remove_last_digit(n):\n    return n // 10", [("remove_last_digit", (147), 14), ("remove_last_digit", (9), 0)]),
        ("is_factor_of", "Write a Python function is_factor_of(f, n) that returns True if f is a factor of n (n % f == 0).", "def is_factor_of(f, n):\n    return n % f == 0", [("is_factor_of", (3, 12), True), ("is_factor_of", (5, 12), False)]),
    ])
    families.append(f24)

    # 25. stats_summaries (VALIDATION)
    f25 = ("stats_summaries", [
        ("mean_two", "Write a Python function mean_two(a, b) that returns the arithmetic mean (a + b) / 2.", "def mean_two(a, b):\n    return (a + b) / 2", [("mean_two", (4, 8), 6.0), ("mean_two", (0, 10), 5.0)]),
        ("mean_three", "Write a Python function mean_three(a, b, c) that returns (a + b + c) / 3.", "def mean_three(a, b, c):\n    return (a + b + c) / 3", [("mean_three", (3, 6, 9), 6.0), ("mean_three", (0, 0, 0), 0.0)]),
        ("midrange_two", "Write a Python function midrange_two(a, b) that returns (min(a, b) + max(a, b)) / 2.", "def midrange_two(a, b):\n    return (min(a, b) + max(a, b)) / 2", [("midrange_two", (2, 8), 5.0), ("midrange_two", (-4, 4), 0.0)]),
        ("range_three", "Write a Python function range_three(a, b, c) that returns max(a, b, c) - min(a, b, c).", "def range_three(a, b, c):\n    return max(a, b, c) - min(a, b, c)", [("range_three", (1, 5, 9), 8), ("range_three", (4, 4, 4), 0)]),
        ("median_three", "Write a Python function median_three(a, b, c) that returns the middle value when sorted.", "def median_three(a, b, c):\n    return sorted([a, b, c])[1]", [("median_three", (1, 9, 5), 5), ("median_three", (8, 2, 4), 4)]),
        ("sum_squares_two", "Write a Python function sum_squares_two(a, b) that returns a*a + b*b.", "def sum_squares_two(a, b):\n    return a * a + b * b", [("sum_squares_two", (3, 4), 25), ("sum_squares_two", (0, 5), 25)]),
        ("diff_of_squares", "Write a Python function diff_of_squares(a, b) that returns a*a - b*b.", "def diff_of_squares(a, b):\n    return a * a - b * b", [("diff_of_squares", (5, 4), 9), ("diff_of_squares", (3, 3), 0)]),
        ("variance_two_approx", "Write a Python function variance_two_approx(a, b) that returns ((a - b) ** 2) / 2.", "def variance_two_approx(a, b):\n    return ((a - b) ** 2) / 2", [("variance_two_approx", (2, 6), 8.0), ("variance_two_approx", (4, 4), 0.0)]),
    ])
    families.append(f25)

    return families
