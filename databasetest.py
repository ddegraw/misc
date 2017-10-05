# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 11:11:35 2016

@author: 4126694
"""

import pyodbc
import pandas.io.sql as psql

cnxn = pyodbc.connect('DRIVER={SQL Server};SERVER=localhost;PORT=1433;DATABASE=SRV_EU_AGC;UID=D4126694;PWD=pass') 
cursor = cnxn.cursor()
sql = "SELECT * FROM TABLE"

df = psql.frame_query(sql, cnxn)
cnxn.close()

