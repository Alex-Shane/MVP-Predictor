#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 12:05:00 2023

@author: Alex
"""

Usage Notes:
    expected columns are:
        First Name
        Last Name
        Rk (just like an index, does not affect the model realize we should have taken this out)
        Tm (team abbrev)
        Lg (AL or NL)
        PA
        HR
        RBI
        SB
        BA
        OPS
        rOBA
        WAR
        G
        GS (games started)
        rTot (Baseball reference version of DRS)
        Pos
        MVP (binary, 1 or 0, should be all zero if testing on new data)
        WinPercentage (team win %)

    Any dataset following the above parameters will work on either model. To change what dataset you're
    testing, just change the file passed into the "data" variable in either tester file. Both models
    should print top 5 expected MVP candidates when run. 