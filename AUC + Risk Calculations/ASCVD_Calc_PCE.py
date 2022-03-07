# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:13:53 2019

@author: jyk306
"""

# %% prepatory 

import pandas as pd
import numpy as np

# %% create coefs data set and baseline variables

coefs = pd.DataFrame({
                      'category': [],
                      'ln-age':[],
                      'ln-age-sq':[],
                      'ln-tot-chol':[],
                      'ln-ageX™ln-tot-chol':[],
                      'ln-HDL':[],
                      'ln-ageXln-HDL':[],
                      'ln-SBP':[],
                      'ln-ageXSBP':[],
                      'ln-unSBP':[],
                      'ln-ageXunSBP':[],
                      'smoker':[],
                      'ln-ageXsmoker':[],
                      'dm':[]
                      })

coefs_list = [['WF', -29.799, 4.884, 13.540, -3.114, -13.578, 3.149, 2.019, 0, 1.957, 0, 7.574, -1.665, 0.661],
         ['BF', 17.114, 0, 0.940, 0, -18.920, 4.475, 29.291, -6.432, 27.820, -6.087, 0.691, 0, 0.874],
         ['WM', 12.344, 0, 11.853, -2.664, -7.990, 1.769, 1.797, 0, 1.764, 0, 7.837, -1.795, 0.658],
         ['BM', 2.469, 0, 0.302, 0, -0.307, 0, 1.916, 0, 1.809, 0, 0.549, 0, 0.645]]

coefs_list = pd.DataFrame(coefs_list)
coefs_list.columns = coefs.columns

coefs = coefs.append(coefs_list)

del coefs_list

baseline_nums = pd.DataFrame({
                              'category': ['WF', 'BF', 'WM', 'BM'],
                              'mean': [-29.18, 86.61, 61.18, 19.54],
                              'surv': [.9665, .9533, .9144, .8954]
                             })

# %% input variables 

def input_vars_function():

    global vars_df
    
    input_vars = pd.DataFrame()
    
    print('race')
    input_vars['race'] = input()
    print('gender')
    input_vars['gender'] = input()
    print('age')
    input_vars['age'] = input()
    print('total cholesterol')
    input_vars['tot_chol'] = input()
    print('HDL-C')
    input_vars['hdl'] = input()
    print('SBP')
    input_vars['sbp'] = input()
    print('treated for hypertension (0 for no, 1 for yes)')
    input_vars['treated'] = input()
    print('smoker(0 for no, 1 for yes') 
    input_vars['smoker'] = input()
    print('diabetes')
    input_vars['diabetes']= input()
    
    return vars_df

# %% variable calculation 

def ascvd_calc(input_vars):
    
    global final_output, vars_calc, pt_categorys
        
    vars_calc = pd.DataFrame({
                              'race': [],
                              'gender': [],
                              'ln_age':[],
                              'ln_age_sq':[],
                              'ln_tot_chol':[],
                              'ln_ageXln_tot_chol':[],
                              'ln_HDL':[],
                              'ln_ageXln_HDL':[],
                              'ln_SBP':[],
                              'ln_ageXSBP':[],
                              'ln_unSBP':[],
                              'ln_ageXunSBP':[],
                              'smoker':[],
                              'ln_ageXsmoker':[],
                              'dm':[]
                              })
    
    # prelimnary calculations    
    vars_calc.race = input_vars.race
    vars_calc.gender = input_vars.gender
    
    vars_calc.ln_age = np.log(input_vars.age)
    vars_calc.ln_age_sq = np.square(vars_calc.ln_age)
    
    vars_calc.ln_tot_chol = np.log(input_vars.tot_chol)
    vars_calc.ln_ageXln_tot_chol = np.log(input_vars.age) * vars_calc.ln_tot_chol
    
    vars_calc.ln_HDL = np.log(input_vars.hdl)
    vars_calc.ln_ageXln_HDL = np.log(input_vars.age) * vars_calc.ln_HDL
    
    input_vars['treated_index'] = 1
    input_vars.loc[input_vars['treated'] == 1, 'treated_index'] = 0 
    vars_calc.ln_SBP = np.log(input_vars.sbp) * input_vars.treated
    vars_calc.ln_unSBP = np.log(input_vars.sbp) * input_vars.treated_index
    vars_calc.ln_ageXSBP = vars_calc.ln_age * vars_calc.ln_SBP 
    vars_calc.ln_ageXunSBP = vars_calc.ln_age * vars_calc.ln_unSBP
        
    vars_calc.smoker = input_vars.smoker
    vars_calc.ln_ageXsmoker = np.log(input_vars.age) * vars_calc.smoker
    vars_calc.dm = input_vars.diabetes
    
    vars_calc.loc[(vars_calc.race == 'white') & (vars_calc.gender == 'F'), 'category'] = 'WF'
    vars_calc.loc[(vars_calc.race == 'black') & (vars_calc.gender == 'F'), 'category'] = 'BF'
    vars_calc.loc[(vars_calc.race == 'white') & (vars_calc.gender == 'M'), 'category'] = 'WM'
    vars_calc.loc[(vars_calc.race == 'black') & (vars_calc.gender == 'M'), 'category'] = 'BM'
    
    final_output = pd.DataFrame()
    
    for i in range(len(vars_calc)):
        coef = vars_calc.iloc[i, [-1]][0]
        coef = coefs.loc[coefs.category == coef, 'ln-age' : 'dm']
        output = pd.DataFrame(np.array(vars_calc.iloc[i, 2:15]) * np.array(coef))
        if i == 0:
            final_output = output
        else:
            final_output = pd.concat([final_output, output])
    
    final_output.columns = vars_calc.columns[2:15]
    final_output['sum'] = final_output.apply(np.sum, axis= 1)
    final_output = final_output.reset_index()
    final_output = final_output.drop(columns = 'index')
    vars_calc = vars_calc.reset_index()
    vars_calc = vars_calc.drop(columns = 'index')
    final_output['category'] = vars_calc['category']

    final_output = pd.merge(final_output, baseline_nums, how = 'left', on = 'category')
    
    final_output['risk'] = round(1 - np.power(final_output['surv'],np.exp(final_output['sum'] - final_output['mean'])),3)

    return(final_output)
    

def convert_df(input_df):

  input_df = input_df.rename(columns = {'chol' : 'tot_chol',
                 'cig'  : 'smoker',
                 'dm03' : 'diabetes',
                 'htnmed': 'treated'})

  # change race and gender categories
  input_df.loc[input_df['race'] == 1, 'race'] = 'white'
  input_df.loc[input_df['race'] == 3, 'race'] = 'white'
  input_df.loc[input_df['race'] == 4, 'race'] = 'white'
  input_df.loc[input_df['race'] == 2, 'race'] = 'black'
  input_df.loc[input_df['race'] == 5, 'race'] = 'white'
  input_df.loc[input_df['race'] == 6, 'race'] = 'white'

  input_df.loc[input_df['gender'] == 2, 'gender'] = 'F'
  input_df.loc[input_df['gender'] == 1, 'gender'] = 'M'

  input_df['age'] = input_df['age'].astype(int)
  input_df['sbp'] = input_df['sbp'].astype(int)
  input_df['hdl'] = input_df['hdl'].astype(int)
  input_df['tot_chol'] = input_df['tot_chol'].astype(int)
  input_df['dbp'] = input_df['dbp'].astype(int)

  return input_df

def pce_prediction(input_df, pred_time):
    
  # get max before threshold 

  max_age_before_threshold = input_df.loc[input_df['times'] <= pred_time, :].groupby('id').times.max().reset_index()
  
  input_df = pd.merge(input_df, max_age_before_threshold)
    
  # calculation 
  prediction_df = ascvd_calc(input_df)

  return prediction_df

def pce_pred_df_tab(input_df, pred_time, pred_time_index):
  

    
  # get max before threshold 
  max_age_before_threshold = input_df.loc[input_df['times'] <= pred_time, :].groupby('id').times.max().reset_index()
  
  input_df = pd.merge(input_df, max_age_before_threshold)

  input_df = pd.DataFrame(input_df)
  #input_df ['x', 'sbp', 'dbp', 'hdl', 'chol', 'age', 'cig', 'dm03', 'htnmed', 'race', 'gender']

  input_df = convert_df(input_df)                    

  prediction_df = ascvd_calc(input_df)
    
  return prediction_df, input_df
    
    