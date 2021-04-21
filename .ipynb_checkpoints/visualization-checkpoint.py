#### Dynamic-DeepHit Prediction Visualization ####
#### Kevin Yu ####

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# get appropriate data
def get_data(patient_num, risk_all, te_data_raw):
    '''Gets the individual predictions from the model output'''
    
    # predictions
    death_pred = risk_all[0][patient_num]
    ascvd_pred = risk_all[1][patient_num]
    
    # patient clinica data
    pce_data = pd.DataFrame(te_data_raw[patient_num])
        
    return death_pred, ascvd_pred, pce_data


# reformat dataset 
def viz_dataset(pred_data):
    '''Converts the ndarray into pandas'''
    
    pred_data = pred_data.tolist()
    # rearrange predictions
    for i in range(len(pred_data)):
        if i == 0:
            pred_data_final = pred_data[i]
        else:
            pred_data_final = pred_data_final + pred_data[i]
    
    return pred_data_final


def dup_list(list_elements, dup_times):
    '''Create pandas indecies suchas prediction day and evaluation day'''
    list_elements_t = list_elements
    for i in range(dup_times-1):
        list_elements = list_elements + list_elements_t 

    return list_elements


def create_outcome_df(pred_df, pred_time, eval_time):
    '''Create outcome dataset - death or ascvd'''

    pred_data_final = viz_dataset(pred_df)
    pred_times = dup_list(pred_time, len(pred_time))
    pred_times = sorted(pred_times)
    eval_times = dup_list(eval_time, len(eval_time))

    pred_df = pd.DataFrame({'prediction': pred_data_final,
                       'pred_time': pred_times,
                       'eval_time': eval_times})

    pred_df['actual_time'] = pred_df['pred_time'] + pred_df['eval_time'] - 1 
    
    return pred_df


def create_pce_df(pce_data):
    '''Create PCE dataset'''
    
    pce_data = pd.DataFrame(pce_data)
    pce_data.columns = ['x', 'sbp', 'dbp', 'hdl', 'chol', 'age', 'cig', 'dm03', 'htnmed', 'race', 'gender']
    
    pce_data['x'] = [1,3,5,6,10,17]
    pce_data = pce_data.rename(columns = {'x':'actual_time'})

    return pce_data
    
    
def create_outcome_final_df(ascvd_pred, death_pred, pred_time, eval_time):
    
    ascvd_df = create_outcome_df(ascvd_pred, pred_time, eval_time)
    death_df = create_outcome_df(death_pred, pred_time, eval_time)
    ascvd_df = ascvd_df.rename(columns = {'prediction' : 'ascvd_prediction'})
    death_df = death_df.rename(columns = {'prediction' : 'death_prediction'})
    
    barrier = dup_list([9999] + pred_time[1:], len(pred_time))
    barrier = sorted(barrier)

    outcome_df = pd.merge(ascvd_df, death_df, how = 'inner')
    outcome_df['barrier'] = barrier
    outcome_df['flag1'] = outcome_df['actual_time'] <= barrier
    outcome_df['flag2'] = outcome_df['actual_time'] == barrier
    
    outcome_df_dup = outcome_df.loc[outcome_df['flag2'] == True, :]
    outcome_df_dup.loc[:, 'flag1'] = False
    outcome_df_final = pd.concat([outcome_df, outcome_df_dup], axis = 0)
    
    return outcome_df_final


def viz_plot(outcome_df, pce_data, te_label, te_time, pred_time):
    '''Create final visualization demonstrating risk'''

    fig, ax1 = plt.subplots()
    fig.set_size_inches(12, 8)
    ax1.set(xlim = (0,20))
    data_1 = outcome_df.loc[outcome_df['flag1'] == True, :]
    data_2 = outcome_df.loc[outcome_df['flag1'] == False, :]
    sns.lineplot(x = 'actual_time', y = 'ascvd_prediction', hue = 'pred_time', data = data_1, legend = False, palette = 'Blues', hue_norm = (.7, .8)) # Prediction with present data
    #sns.lineplot(x = 'actual_time', y = 'ascvd_prediction', hue = 'pred_time', data = data_2, legend = False, alpha = .25, palette = 'Blues', hue_norm = (.7, .8)) # Prediction with present data
    sns.lineplot(x = 'actual_time', y = 'death_prediction', hue = 'pred_time', data = data_1, legend = False, palette = 'Oranges', hue_norm = (.3, .4)) # Prediction with present data
    #sns.lineplot(x = 'actual_time', y = 'death_prediction', hue = 'pred_time', data = data_2, legend = False, alpha = .25, palette = 'Oranges', hue_norm = (.3, .4)) # Prediction with present data
    ax1.set_xlabel('Actual Time (Years)')
    ax1.set_ylabel('Predication - Death(Brown) / ASCVD(Blue)')
    ax2 = ax1.twinx()
    ax2.set(ylim = (0, 400))
    
    pce_data = pce_data.loc[pce_data['sbp'] + pce_data['chol'] > 0, :]
    
    sns.lineplot(x = 'actual_time', y = 'sbp', data = pce_data, color = 'lightpink')
    sns.lineplot(x = 'actual_time', y = 'chol', data = pce_data, color = 'orange')
    ax2.set_ylabel('SBP(Red) / Cholesterol(Yellow)')
    for i in range(len(pred_time)):
        plt.axvline(pred_time[i], 0, 0.95, linestyle = '--', alpha = .5)
    sbp_std = pce_data.sbp.std()
    pce_data['med_change'] = pce_data['htnmed'] - pce_data.shift(+1)['htnmed'] 
    #pce_data.loc[pce_data['med_change'] != 1, 'med_change'] = 0
    #pce_data.loc[pce_data['med_change'] == 1, 'med_change'] = 1

    pce_data_med_change = pce_data.loc[pce_data['med_change'] == 1, :]
    
    if len(pce_data_med_change) == 0:
        print('No Med Change')
    else:
        x = pce_data_med_change['actual_time'].values
        y = pce_data_med_change['sbp'].values[0] - sbp_std
        plt.text(x, y, s = 'Med Change', color = 'green', horizontalalignment='center')

        
    if te_label == 0:
        plt.text(x = te_time, y = 150, s = 'right censored', horizontalalignment='left')
    elif te_label == 1: 
        plt.text(x = te_time, y = 150, s = 'death (other)', horizontalalignment='left')
    elif te_label == 2: 
        plt.text(x = te_time, y = 150, s = 'ASCVD', horizontalalignment='left')
    
    plt.show()
        
    return