def prediction_df(risk_all, event):
    """
    Gets the individual predictions from risk_all
    event = 0 -> 
    """
    
    for i in range(len(risk_all[event])): 
        
        chunk = risk_all[event][i].tolist()
        chunk = pd.DataFrame(chunk)
        chunk['pt_id'] = i
        if i == 0:
            df = chunk
        else: 
            df = pd.concat([df, chunk])

    df = df.reset_index()
    df = df.rename(columns = {'index':'pred_time'})
    df['pred_time'] = df['pred_time'] * 2 + 1 # need to change if eval or pred time is edited

    df = pd.melt(df, id_vars=['pt_id', 'pred_time'], value_vars=[0]).sort_values(['pt_id', 'pred_time', 'variable'])

    df = df.rename(columns = {'variable' : 'eval_time'})
    
    return df

def prediction_auc(pred_t, eval_t , event_n, df):
    """Calculates the AUC at each prediction time and prediction time"""
    
    global true 
    
    time_horizon = pred_t + eval_t 
    true = (te_time <= 15) * (te_label == 1).astype(int)

    auc = roc_auc_score(true.tolist(), df.loc[(df.pred_time == pred_t) & (df.eval_time == eval_t), 'value'].tolist())
            
    return auc 

def prediction_pred_values(pred_t, eval_t , event_n, df):
    """Gets the predication values and true labels for AUC CI calculation"""
    
    global true 
    
    time_horizon = pred_t + eval_t 
    true = (te_time <= time_horizon) * (te_label == event_n).astype(int)

    pred_values = df.loc[(df.pred_time == pred_t) & (df.eval_time == eval_t), 'value'].tolist()
            
    pred_values_df = pd.DataFrame(true, columns = ['true_labels'])
    pred_values_df['pred_values'] = pred_values    
    
    return pred_values_df

def prediction_ROC(pred_t, eval_t , event_n, df):
    """Calculates the AUC at each prediction time and prediction time"""
    
    global true 
    
    time_horizon = pred_t + eval_t 
    true = (te_time <= 15) * (te_label == 1).astype(int)

    fpr, tpr, thresh = roc_curve(true.tolist(), df.loc[(df.pred_time == pred_t) & (df.eval_time == eval_t), 'value'].tolist())
            
    return fpr, tpr, thresh

def get_all_auc(risk_all):
    """Get the auc matrix"""

    pred_df = prediction_df(risk_all, 0)

    for p, p_t in enumerate(pred_time):
        p_t_row = []
        for e, e_t in enumerate(eval_time):
            p_t_row.append(round(prediction_auc(p_t, e, 1, pred_df),3))
        if p == 0:
            auc_df = pd.DataFrame(p_t_row).T
        else: 
            auc_df = pd.concat([auc_df, pd.DataFrame(p_t_row).T], axis = 0)

    auc_df.columns = ['eval_time_' + str(i) for i in eval_time]
    auc_df.index = ['pred_time_' + str(i) for i in pred_time]
    
    return auc_df, pred_df