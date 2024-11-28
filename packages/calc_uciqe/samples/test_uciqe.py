#!/usr/bin/env python
"""
Sample script that uses the calc_uciqe module created using
MATLAB Compiler SDK.

Refer to the MATLAB Compiler SDK documentation for more information.
"""

import calc_uciqe
# Import the matlab module only after you have imported
# MATLAB Compiler SDK generated Python modules.
import matlab

my_calc_uciqe = calc_uciqe.initialize()

file_pathIn = "D:\\images\\valA_test\\56_img_.png"
uciqe_resultOut = my_calc_uciqe.calc_uciqe(file_pathIn)
print(uciqe_resultOut, sep='\n')

my_calc_uciqe.terminate()
