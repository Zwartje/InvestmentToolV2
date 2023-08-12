# This file contains useful utility functions.

import pandas as pd
import numpy as np
from scipy import stats


def remove_illegal_symbols(s):
    """
    This function checks if a string variable contains any "illegal" characters that prevent the string to appear in a
    file name in the Windows OS.

    :param s (str): a string variable containing potentially illegal characters in Windows file name.
    :return: a string variable with any illegal characters removed.
    """

    illegal_symbols = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '^']
    cleaned_string = s

    for char in illegal_symbols:
        cleaned_string = cleaned_string.replace(char, '')

    return cleaned_string

