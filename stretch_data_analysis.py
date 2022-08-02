import re
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from matplotlib import pyplot as plt
from easygui import *
from scipy.optimize import curve_fit
from matplotlib.widgets import SpanSelector 
from iteration_utilities import grouper
from scipy.stats import pearsonr 

with open('Z:/All_Staff/Grandl Lab/Michael Sindoni/initial_frequency_screen/stretch/initial_freq_screen_complete.csv', 'r') as fhand:
    df = pd.read_csv(fhand)
print(df)

#input what conditions are being analyzed
conditions = ['con', '1', '5', '10', '20', '50', '100', '200']


##################################################################################################
def max_current_box_plot(df, conditions): 
    condition_list = conditions
    conditions_dict = {}
    
    #generates items (key/value) in the conditions dictionary. The key is the condition and the value is an empty list
    for i in range(len(condition_list)):
        conditions_dict[condition_list[i]] = [] 

    #go through each trial, get max current, and add to correct value list in dictionary
    grouped = df.groupby(['condition', 'day', 'trial'])
    for name, group in grouped:
        max_value = group['current'].max()
        ind_condition = name[0] #gets the condition
        conditions_dict[ind_condition].append(max_value) #appends list for that condition in the original dictionary

    #converts dictionary into a dataframe and creates box/strip plot
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in conditions_dict.items() ])).melt().dropna()

    #creating palette for box plot color scheme. Will correct for number of conditions with 8 being the max (con-200Hz)
    pal_colors = ['grey', '#464196', '#ffcfdc', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet']
    pal_dict = {}
    for i in range(len(condition_list)):
        pal_dict[conditions[i]] = pal_colors[i]
    sns.boxplot(data = df, x = 'variable', y = 'value', palette=pal_dict)
    sns.stripplot(data = df, x = 'variable', y = 'value', color = 'black')
    plt.ylim(0, 70)
    plt.show()

def p50_box_plot(df, conditions):
    condition_list = conditions
    conditions_dict = {}

    #generates items (key/value) in the conditions dictionary. The key is the condition and the value is an empty list
    for i in range(len(condition_list)):
        conditions_dict[condition_list[i]] = [] 

    #go through each trial, get p50, and add to correct value list in dictionary
    grouped = df.groupby(['condition', 'day', 'trial'])
    for name, group in grouped:
        p50_value = list(group['p50'])[0] #converts pandas core series to list and takes first element since they are all the same per group
        ind_condition = name[0] #gets the condition
        conditions_dict[ind_condition].append(p50_value) #appends list for that condition in the original dictionary


    #converts dictionary into a dataframe and creates box/strip plot
    df2 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in conditions_dict.items() ])).melt().dropna()

    #adding 5 mmHg to correct for not starting at 0mmHg
    df2['value'] = df2['value'] + 5

    #creating palette for box plot color scheme. Will correct for number of conditions with 8 being the max (con-200Hz)
    pal_colors = ['grey', '#464196', '#ffcfdc', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet']
    pal_dict = {}
    for i in range(len(condition_list)):
        pal_dict[conditions[i]] = pal_colors[i]
    sns.boxplot(data = df2, x = 'variable', y = 'value', palette=pal_dict)
    sns.stripplot(data = df2, x = 'variable', y = 'value', color = 'black')
    plt.ylim(0, 40)
    plt.show()

    #getting n for each category
    n_grouped = df2.groupby(['variable'])
    for name, group in n_grouped:
        len_category = len(group)
        print(name, ':', len_category)


def p50_current_cutoff_plot(df, cutoff_list):
    for i in range(len(cutoff_list)):
        cutoff = cutoff_list[i]
        #first making the raw p50 plots
        grouped = df.groupby(['condition', 'trial'])
        current_avg_list = []
        mmHg_list = []
        category_list = []
        for name,group in grouped:
            current_list = group['current'].tolist()
            mmHg_list2 = group['mmHg'].tolist()
            if max(current_list) > i:
                for i in range(len(current_list)):
                    current_avg_list.append(current_list[i])
                    mmHg_list.append(mmHg_list2[i])
                    category_list.append(name[0])
            else:
                continue

        #adding 5mmHg to each sweep to correct for misreading pressure protocol
        mmHg_list = [x + 5 for x in mmHg_list]

        df2 = pd.DataFrame({'category': category_list,
                            'mmHg' : mmHg_list,
                        'current_avg' : current_avg_list})

        my_pal = ('grey', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet')
        hue_order = ['con', '10', '20', '50', '100', '200']
        sns.pointplot(data = df2, x = 'mmHg', y = 'current_avg', hue = 'category', hue_order = hue_order, palette = my_pal, ci = 'sd', estimator=np.mean,
        scale = 0.7, errwidth = 1, capsize = 0.5, legend = False)
        plt.show()

        #second, making a normalized p50 curve
        norm_i_avg_list = []
        mmHg_list = []
        category_list = []
        for name,group in grouped:
            current_list = group['norm_i'].tolist()
            max_current_cutoff = group['current'].max()
            mmHg_list2 = group['mmHg'].tolist()
            if max_current_cutoff > cutoff:
                for i in range(len(current_list)):
                    norm_i_avg_list.append(current_list[i])
                    mmHg_list.append(mmHg_list2[i])
                    category_list.append(name[0])
                else:
                    continue
                
        #adding 5mmHg to each sweep to correct for misreading pressure protocol
        mmHg_list = [x + 5 for x in mmHg_list]

        df2 = pd.DataFrame({'category': category_list,
                            'mmHg' : mmHg_list,
                        'norm_i' : norm_i_avg_list})

        my_pal = ('grey', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet')
        hue_order = ['con', '10', '20', '50', '100', '200']
        sns.pointplot(data = df2, x = 'mmHg', y = 'norm_i', hue = 'category', hue_order = hue_order, palette = my_pal, ci = 'sd', estimator=np.mean,
        scale = 0.7, errwidth = 1, capsize = 0.5, legend = False)
        plt.show()


        #third, making boxplots of p50 for each condition that had a max current greater than the cutoff
        condition_list = conditions
        conditions_dict = {}

        #generates items (key/value) in the conditions dictionary. The key is the condition and the value is an empty list
        for j in range(len(condition_list)):
            conditions_dict[condition_list[j]] = [] 

        #go through each trial, get p50, and add to correct value list in dictionary
        grouped = df.groupby(['condition', 'day', 'trial'])
        for name, group in grouped:
            p50_value = list(group['p50'])[0] #converts pandas core series to list and takes first element since they are all the same per group
            if group['current'].max() > cutoff:
                ind_condition = name[0] #gets the condition
                conditions_dict[ind_condition].append(p50_value) #appends list for that condition in the original dictionary

        #converts dictionary into a dataframe and creates box/strip plot
        df3 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in conditions_dict.items() ])).melt().dropna()

        #adding 5 mmHg to correct for not starting at 0mmHg
        df3['value'] = df3['value'] + 5

        #creating palette for box plot color scheme. Will correct for number of conditions with 8 being the max (con-200Hz)
        pal_colors = ['grey', '#464196', '#ffcfdc', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet']
        pal_dict = {}
        for i in range(len(condition_list)):
            pal_dict[conditions[i]] = pal_colors[i]
        sns.boxplot(data = df3, x = 'variable', y = 'value', palette=pal_dict)
        sns.stripplot(data = df3, x = 'variable', y = 'value', color = 'black')
        plt.ylim(0, 40)
        plt.show()

        #getting n for each category
        n_grouped = df3.groupby(['variable'])
        for name, group in n_grouped:
            len_category = len(group)
            print(name, ':', len_category)

def current_time_correlation(df):
    grouped = df.groupby(['condition', 'day', 'trial'])
    hz_10_post_shaking = []
    hz_10_max_current = []
    hz_10_category = []

    con_post_shaking = []
    con_max_current = []
    con_category = []

    hz_20_post_shaking = []
    hz_20_max_current = []
    hz_20_category = []

    hz_50_post_shaking = []
    hz_50_max_current = []
    hz_50_category = []

    hz_100_post_shaking = []
    hz_100_max_current = []
    hz_100_category = []

    hz_200_post_shaking = []
    hz_200_max_current = []
    hz_200_category = []    

    for name, group in grouped:
        if group['condition'].iat[0] == '10':
            hz_10_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_10_max_current.append(float(group['current'].max())) 
            hz_10_category.append(10)

        if group['condition'].iat[0] == '20':
            hz_20_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_20_max_current.append(float(group['current'].max())) 
            hz_20_category.append(20)

        if group['condition'].iat[0] == '50':
            hz_50_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_50_max_current.append(float(group['current'].max())) 
            hz_50_category.append(50)

        if group['condition'].iat[0] == '100':
            hz_100_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_100_max_current.append(float(group['current'].max())) 
            hz_100_category.append(100)

        if group['condition'].iat[0] == '200':
            hz_200_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_200_max_current.append(float(group['current'].max())) 
            hz_200_category.append(200)

        if group['condition'].iat[0] == 'con':
            con_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            con_max_current.append(float(group['current'].max())) 
            con_category.append('con')

            
    max_current = con_max_current + hz_10_max_current + hz_20_max_current + hz_50_max_current + hz_100_max_current + hz_200_max_current
    time_post_shaking = con_post_shaking + hz_10_post_shaking + hz_20_post_shaking + hz_50_post_shaking + hz_100_post_shaking + hz_200_post_shaking
    category = con_category + hz_10_category + hz_20_category + hz_50_category + hz_100_category + hz_200_category

    df2 = pd.DataFrame({'max_current': max_current,
                    'time_post_shaking' : time_post_shaking,
                    'category' : category})
    my_pal = {'con': 'grey', 10: 'maroon', 20:'goldenrod', 50:'navy', 100:'olivedrab', 200:'blueviolet'}
    sns.scatterplot(data = df2, x = 'time_post_shaking', y = 'max_current', hue = 'category', palette = my_pal)
    plt.ylim(0, 65)
    plt.show()
    print(type(my_pal))

    #breaking apart each category, plotting, and finding the correlation coefficient
    sub_grouped = df2.groupby('category')
    for name, group in sub_grouped:
        correlation = pearsonr(group['max_current'], group['time_post_shaking'])[0]
        pval = pearsonr(group['max_current'], group['time_post_shaking'])[1]

        # sns.scatterplot(data = group, x = 'time_post_shaking', y = 'max_current')
        sns.lmplot(data = group, x = 'time_post_shaking', y = 'max_current', line_kws={'color': 'black'}, ci = None)
        correlation_text = 'r = ' + str(correlation)
        pval_text = 'p = ' + str(pval)

        plt.text(10+0.2, 10, correlation_text, horizontalalignment='left', size = 20, color='black', weight='semibold')
        plt.text(10+0.2, 4.5, pval_text, horizontalalignment='left', size= 20, color='black', weight='semibold')
        plt.ylim(0, 65)
        plt.xlabel('time_post_shaking', size = 20)
        plt.ylabel('max_i', size = 20)
        plt.title(name, size = 20)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        plt.show()

def t63_plot(df):
    grouped = df.groupby(['condition', 'mmHg'])
    t63_avg_list = []
    mmHg_list = []
    category_list = []
    for name,group in grouped:
        print(group)
        t63_avg = group['t63'].tolist()
        for i in t63_avg:
            t63_avg_list.append(i)
            mmHg_list.append(name[1])
            category_list.append(name[0])
   #adding 5 mmHg to correct for not starting at 0mmHg
    mmHg_list = [i + 5 for i in mmHg_list]
    
    df2 = pd.DataFrame({'category': category_list,
                        'mmHg' : mmHg_list,
                    't63_avg' : t63_avg_list})
      
    my_pal = {'grey', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet'}
    hue_order = ['con', '10', '20', '50', '100', '200']
    sns.pointplot(data = df2, x = 'mmHg', y = 't63_avg', hue = 'category', hue_order = hue_order, palette = my_pal, ci = 'sd', estimator=np.mean,
    scale = 0.7, errwidth = 1, capsize = 0.5, legend = False)
    plt.legend(loc='upper center')
    plt.show()

    sns.stripplot(data = df2, x = 'mmHg', y = 't63_avg', hue = 'category', hue_order = hue_order, palette = my_pal)
    plt.show()

def ssc_peak_ratio_plot(df):
    grouped = df.groupby(['condition', 'mmHg'])
    ssc_ratio_avg_list = []
    mmHg_list = []
    category_list = []
    for name,group in grouped:
        print(group)
        ssc_ratio_avg = group['ssc/peak'].tolist()
        for i in ssc_ratio_avg:
            ssc_ratio_avg_list.append(i)
            mmHg_list.append(name[1])
            category_list.append(name[0])

   #adding 5 mmHg to correct for not starting at 0mmHg
    mmHg_list = [i + 5 for i in mmHg_list]

    df2 = pd.DataFrame({'category': category_list,
                        'mmHg' : mmHg_list,
                    'ssc/peak' : ssc_ratio_avg_list})

 
    my_pal = {'grey', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet'}
    hue_order = ['con', '10', '20', '50', '100', '200']
    sns.pointplot(data = df2, x = 'mmHg', y = 'ssc/peak', hue = 'category', hue_order = hue_order, palette =my_pal, ci = 'sd', estimator=np.mean,
    scale = 0.7, errwidth = 1, capsize = 0.5)
    plt.ylim(0, 1)
    plt.show()

    sns.stripplot(data = df2, x = 'mmHg', y = 'ssc/peak', hue = 'category', hue_order = hue_order, palette = my_pal)
    plt.ylim(0, 1)
    plt.show()


def p50_time_correlation(df):
    grouped = df.groupby(['condition', 'day', 'trial'])
    hz_10_post_shaking = []
    hz_10_p50 = []
    hz_10_category = []

    con_post_shaking = []
    con_p50 = []
    con_category = []

    hz_20_post_shaking = []
    hz_20_p50 = []
    hz_20_category = []

    hz_50_post_shaking = []
    hz_50_p50 = []
    hz_50_category = []

    hz_100_post_shaking = []
    hz_100_p50 = []
    hz_100_category = []

    hz_200_post_shaking = []
    hz_200_p50 = []
    hz_200_category = []

    for name, group in grouped:
        if group['condition'].iat[0] == '10':
            hz_10_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_10_p50.append(float(group['p50'].max())) 
            hz_10_category.append(10)

        if group['condition'].iat[0] == '20':
            hz_20_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_20_p50.append(float(group['p50'].max())) 
            hz_20_category.append(20)

        if group['condition'].iat[0] == '50':
            hz_50_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_50_p50.append(float(group['p50'].max())) 
            hz_50_category.append(50)

        if group['condition'].iat[0] == '100':
            hz_100_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_100_p50.append(float(group['p50'].max())) 
            hz_100_category.append(100)

        if group['condition'].iat[0] == 'con':
            con_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            con_p50.append(float(group['p50'].max())) 
            con_category.append('con')

        if group['condition'].iat[0] == '200':
            hz_200_post_shaking.append(float(group['time_post_shaking'].iat[0]))
            hz_200_p50.append(float(group['p50'].max())) 
            hz_200_category.append(200)
            
    p50 = con_p50 + hz_10_p50 + hz_20_p50 + hz_50_p50 + hz_100_p50 + hz_200_p50
    #adding 5 mmHg to correct for not starting at 0mmHg
    p50 = [i + 5 for i in p50]

    time_post_shaking = con_post_shaking + hz_10_post_shaking + hz_20_post_shaking + hz_50_post_shaking + hz_100_post_shaking + hz_200_post_shaking
    category = con_category + hz_10_category + hz_20_category + hz_50_category + hz_100_category + hz_200_category

    df2 = pd.DataFrame({'p50': p50,
                    'time_post_shaking' : time_post_shaking,
                    'category' : category})
    my_pal = {'con': 'grey', 10: 'maroon', 20:'goldenrod', 50:'navy', 100:'olivedrab', 200:'blueviolet'}
    sns.scatterplot(data = df2, x = 'time_post_shaking', y = 'p50', hue = 'category', palette = my_pal)
    plt.ylim(0, 40)
    plt.show()
    print(type(my_pal))

    #breaking apart each category, plotting, and finding the correlation coefficient
    sub_grouped = df2.groupby('category')
    for name, group in sub_grouped:
        correlation = pearsonr(group['p50'], group['time_post_shaking'])[0]
        pval = pearsonr(group['p50'], group['time_post_shaking'])[1]

        # sns.scatterplot(data = group, x = 'time_post_shaking', y = 'max_current')
        sns.lmplot(data = group, x = 'time_post_shaking', y = 'p50', line_kws={'color': 'black'}, ci = None)
        correlation_text = 'r = ' + str(correlation)
        pval_text = 'p = ' + str(pval)

        plt.text(10+0.2, 10, correlation_text, horizontalalignment='left', size = 20, color='black', weight='semibold')
        plt.text(10+0.2, 4.5, pval_text, horizontalalignment='left', size= 20, color='black', weight='semibold')
        plt.ylim(0, 40)
        plt.xlabel('time_post_shaking', size = 20)
        plt.ylabel('p50', size = 20)
        plt.title(name, size = 20)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        plt.show()

def pip_resistance(df):
    condition_list = conditions
    conditions_dict = {}

    #generates items (key/value) in the conditions dictionary. The key is the condition and the value is an empty list
    for i in range(len(condition_list)):
        conditions_dict[condition_list[i]] = [] 

    #go through each trial, get p50, and add to correct value list in dictionary
    grouped = df.groupby(['condition', 'day', 'trial'])
    for name, group in grouped:
        pip_r = list(group['pip_resistance'])[0] #converts pandas core series to list and takes first element since they are all the same per group
        ind_condition = name[0] #gets the condition
        conditions_dict[ind_condition].append(pip_r) #appends list for that condition in the original dictionary


    #converts dictionary into a dataframe and creates box/strip plot
    df2 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in conditions_dict.items() ])).melt().dropna()

    #creating palette for box plot color scheme. Will correct for number of conditions with 8 being the max (con-200Hz)
    pal_colors = ['grey', '#464196', '#ffcfdc', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet']
    pal_dict = {}
    for i in range(len(condition_list)):
        pal_dict[conditions[i]] = pal_colors[i]
    sns.boxplot(data = df2, x = 'variable', y = 'value', palette=pal_dict)
    sns.stripplot(data = df2, x = 'variable', y = 'value', color = 'black')
    plt.ylim(0,)
    plt.show()

    #getting n for each category
    n_grouped = df2.groupby(['variable'])
    for name, group in n_grouped:
        len_category = len(group)
        print(name, ':', len_category)

#############################################################################################################
max_current_plot = max_current_box_plot(df,conditions) #cleaned
p50_plot = p50_box_plot(df, conditions) #cleaned
current_time_plot = current_time_correlation(df)
t63 = t63_plot(df)
ssc_peak = ssc_peak_ratio_plot(df)
p50_time_plot = p50_time_correlation(df)
cutoff_plots = p50_current_cutoff_plot(df, (20, 30, 40)) #partially cleaned
resistances = pip_resistance(df) #cleaned
#############################################################################################################
cutoff_list = [20, 30, 40]

for i in range(len(cutoff_list)):
    condition_list = conditions
    conditions_dict = {}
    
    #generates items (key/value) in the conditions dictionary. The key is the condition and the value is an empty list
    for j in range(len(condition_list)):
        conditions_dict[condition_list[j]] = [] 

    #go through each trial, get p50, and add to correct value list in dictionary
    grouped = df.groupby(['condition', 'day', 'trial'])
    for name, group in grouped:
        p50_value = list(group['p50'])[0] #converts pandas core series to list and takes first element since they are all the same per group
        cutoff = cutoff_list[i]
        if group['current'].max() > cutoff:
            ind_condition = name[0] #gets the condition
            conditions_dict[ind_condition].append(p50_value) #appends list for that condition in the original dictionary

    #converts dictionary into a dataframe and creates box/strip plot
    df2 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in conditions_dict.items() ])).melt().dropna()

    #adding 5 mmHg to correct for not starting at 0mmHg
    df2['value'] = df2['value'] + 5

    #creating palette for box plot color scheme. Will correct for number of conditions with 8 being the max (con-200Hz)
    pal_colors = ['grey', '#464196', '#ffcfdc', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet']
    pal_dict = {}
    for i in range(len(condition_list)):
        pal_dict[conditions[i]] = pal_colors[i]
    sns.boxplot(data = df2, x = 'variable', y = 'value', palette=pal_dict)
    sns.stripplot(data = df2, x = 'variable', y = 'value', color = 'black')
    plt.ylim(0, 40)
    plt.show()

    #getting n for each category
    n_grouped = df2.groupby(['variable'])
    for name, group in n_grouped:
        len_category = len(group)
        print(name, ':', len_category)


###################################################################
for i in range(len(cutoff_list)):
    cutoff = cutoff_list[i]
    #first making the raw p50 plots
    grouped = df.groupby(['condition', 'trial'])
    current_avg_list = []
    mmHg_list = []
    category_list = []
    for name,group in grouped:
        current_list = group['current'].tolist()
        mmHg_list2 = group['mmHg'].tolist()
        if max(current_list) > i:
            for i in range(len(current_list)):
                current_avg_list.append(current_list[i])
                mmHg_list.append(mmHg_list2[i])
                category_list.append(name[0])
        else:
            continue

    #adding 5mmHg to each sweep to correct for misreading pressure protocol
    mmHg_list = [x + 5 for x in mmHg_list]

    df2 = pd.DataFrame({'category': category_list,
                        'mmHg' : mmHg_list,
                    'current_avg' : current_avg_list})

    my_pal = ('grey', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet')
    hue_order = ['con', '10', '20', '50', '100', '200']
    sns.pointplot(data = df2, x = 'mmHg', y = 'current_avg', hue = 'category', hue_order = hue_order, palette = my_pal, ci = 'sd', estimator=np.mean,
    scale = 0.7, errwidth = 1, capsize = 0.5, legend = False)
    plt.show()

    #second, making a normalized p50 curve
    norm_i_avg_list = []
    mmHg_list = []
    category_list = []
    for name,group in grouped:
        current_list = group['norm_i'].tolist()
        max_current_cutoff = group['current'].max()
        mmHg_list2 = group['mmHg'].tolist()
        if max_current_cutoff > cutoff:
            for i in range(len(current_list)):
                norm_i_avg_list.append(current_list[i])
                mmHg_list.append(mmHg_list2[i])
                category_list.append(name[0])
            else:
                continue
            
    #adding 5mmHg to each sweep to correct for misreading pressure protocol
    mmHg_list = [x + 5 for x in mmHg_list]

    df2 = pd.DataFrame({'category': category_list,
                        'mmHg' : mmHg_list,
                    'norm_i' : norm_i_avg_list})

    my_pal = ('grey', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet')
    hue_order = ['con', '10', '20', '50', '100', '200']
    sns.pointplot(data = df2, x = 'mmHg', y = 'norm_i', hue = 'category', hue_order = hue_order, palette = my_pal, ci = 'sd', estimator=np.mean,
    scale = 0.7, errwidth = 1, capsize = 0.5, legend = False)
    plt.show()


    #third, making boxplots of p50 for each condition that had a max current greater than the cutoff
    condition_list = conditions
    conditions_dict = {}

    #generates items (key/value) in the conditions dictionary. The key is the condition and the value is an empty list
    for j in range(len(condition_list)):
        conditions_dict[condition_list[j]] = [] 

    #go through each trial, get p50, and add to correct value list in dictionary
    grouped = df.groupby(['condition', 'day', 'trial'])
    for name, group in grouped:
        p50_value = list(group['p50'])[0] #converts pandas core series to list and takes first element since they are all the same per group
        if group['current'].max() > cutoff:
            ind_condition = name[0] #gets the condition
            conditions_dict[ind_condition].append(p50_value) #appends list for that condition in the original dictionary

    #converts dictionary into a dataframe and creates box/strip plot
    df3 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in conditions_dict.items() ])).melt().dropna()

    #adding 5 mmHg to correct for not starting at 0mmHg
    df3['value'] = df3['value'] + 5

    #creating palette for box plot color scheme. Will correct for number of conditions with 8 being the max (con-200Hz)
    pal_colors = ['grey', '#464196', '#ffcfdc', 'maroon', 'goldenrod', 'navy', 'olivedrab', 'blueviolet']
    pal_dict = {}
    for i in range(len(condition_list)):
        pal_dict[conditions[i]] = pal_colors[i]
    sns.boxplot(data = df3, x = 'variable', y = 'value', palette=pal_dict)
    sns.stripplot(data = df3, x = 'variable', y = 'value', color = 'black')
    plt.ylim(0, 40)
    plt.show()

    #getting n for each category
    n_grouped = df3.groupby(['variable'])
    for name, group in n_grouped:
        len_category = len(group)
        print(name, ':', len_category)