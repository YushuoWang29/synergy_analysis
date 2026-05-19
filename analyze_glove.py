import pandas as pd  
xls = pd.ExcelFile('f:/Research/SGLab/cable_driven/Dexterous_Hand/code/synergy_analysis/MyoKi Dataset/Glove_calibration.xlsx')  
for sn in xls.sheet_names:  
    print(f'===== Sheet: {sn} =====')  
    df = pd.read_excel(xls, sheet_name=sn)  
    print(df.to_string()) 
