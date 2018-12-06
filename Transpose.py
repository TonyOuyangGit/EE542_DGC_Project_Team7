import sys
import numpy
import pandas as pd 

pathGTex = "/Users/Tony/Desktop/tmp/GTex_ReadCount.csv"
pathOutPout = "/Users/Tony/Desktop/tmp/GTex_Transpose.csv"

df = pd.read_csv(pathGTex)

df = df.set_index("rid").T

df["file_id"] = df.index

df.to_csv(pathOutPout)