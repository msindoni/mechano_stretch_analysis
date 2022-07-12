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

with open('C:/Users/mjs164/Box Sync/Data/initial_frequency_screen/stretch/initial_freq_screen_p50.csv', 'r') as fhand:
    df = pd.read_csv(fhand)
    # df['mmHg'] = df['mmHg'] + 5
    # print(df)

def max_current_box_plot(df):
    grouped = df.groupby(['condition', 'day', 'trial'])
    hz_10_max_current = []
    hz_10_category = []
    con_max_current = []
    con_category = []
    hz_20_max_current = []
    hz_20_category = []
    hz_50_max_current = []
    hz_50_category = []
    hz_100_max_current = []
    hz_100_category = []
    hz_200_max_current = []
    hz_200_category = []


    for name, group in grouped:
        if group['condition'].iat[0] == 'con':
            con_max_current.append(float(group['current'].max()))
            con_category.append('con')
        if group['condition'].iat[0] == '10':
            hz_10_max_current.append(float(group['current'].max()))
            hz_10_category.append(10)
        if group['condition'].iat[0] == '20':
            hz_10_max_current.append(float(group['current'].max()))
            hz_10_category.append(20)
        if group['condition'].iat[0] == '50':
            hz_10_max_current.append(float(group['current'].max()))
            hz_10_category.append(50)
        if group['condition'].iat[0] == '100':
            hz_10_max_current.append(float(group['current'].max()))
            hz_10_category.append(100)
        if group['condition'].iat[0] == '200':
            hz_20_max_current.append(float(group['current'].max()))
            hz_20_category.append(200)

    category = con_category + hz_10_category + hz_20_category + hz_50_category + hz_100_category + hz_200_category
    max_current = con_max_current + hz_10_max_current + hz_20_max_current + hz_50_max_current + hz_100_max_current + hz_200_max_current

    df2 = pd.DataFrame({'category': category,
                    'max_current' : max_current})

    #getting n for each category
    n_grouped = df2.groupby(['category'])
    for name, group in n_grouped:
        len_category = len(group)
        print(name, ':', len_category)

    my_pal = {'con': 'grey', 10: 'maroon', 20:'goldenrod', 50:'navy', 100:'olivedrab', 200:'blueviolet'}
    sns.boxplot(data = df2, x = 'category', y = 'max_current', order=["con", 10, 20, 50, 100, 200],  palette=my_pal)
    sns.stripplot(data = df2, x = 'category', y = 'max_current', color = 'black', order=["con", 10, 20, 50, 100, 200])
    plt.ylim(0, 70)
    plt.show()

def p50_box_plot(df):
    grouped = df.groupby(['condition', 'day', 'trial'])

    hz_10_p50 = []
    hz_10_category = []
    con_p50 = []
    con_category = []
    hz_20_p50 = []
    hz_20_category = []
    hz_50_p50 = []
    hz_50_category = []
    hz_100_p50 = []
    hz_100_category = []
    hz_200_p50 = []
    hz_200_category = []

    for name, group in grouped:
        if group['condition'].iat[0] == '10':
            hz_10_p50.append(float(group['p50'].iat[0]))
            hz_10_category.append(10)
        if group['condition'].iat[0] == '20':
            hz_20_p50.append(float(group['p50'].iat[0]))
            hz_20_category.append(20)
        if group['condition'].iat[0] == '50':
            hz_50_p50.append(float(group['p50'].iat[0]))
            hz_50_category.append(50)
        if group['condition'].iat[0] == '100':
            hz_100_p50.append(float(group['p50'].iat[0]))
            hz_100_category.append(100)
        if group['condition'].iat[0] == 'con':
            con_p50.append(float(group['p50'].iat[0]))
            con_category.append('con')
        if group['condition'].iat[0] == 'con':
            hz_200_p50.append(float(group['p50'].iat[0]))
            hz_200_category.append(200)

    category = con_category + hz_10_category + hz_20_category + hz_50_category + hz_100_category + hz_200_category
    p50 = con_p50 + hz_10_p50 + hz_20_p50 + hz_50_p50 + hz_100_p50 + hz_200_p50

    #adding 5 mmHg to correct for not starting at 0mmHg
    p50 = [i + 5 for i in p50]

    df2 = pd.DataFrame({'category': category,
                    'P50' : p50})

    #getting n for each category
    n_grouped = df2.groupby(['category'])
    for name, group in n_grouped:
        len_category = len(group)
        print(name, ':', len_category)

    my_pal = {'con': 'grey', 10: 'maroon', 20:'goldenrod', 50:'navy', 100:'olivedrab', 200:'blueviolet'}
    sns.boxplot(data = df2, x = 'category', y = 'P50', palette = my_pal,  order=["con", 10, 20, 50, 100, 200])
    sns.stripplot(data = df2, x = 'category', y = 'P50', color = 'black', order=["con", 10, 20, 50, 100, 200])
    plt.ylim(0, 50)
    plt.show()

def p50_current_cutoff_plot(df, cutoff_list):
    for i in cutoff_list:
        cutoff = i
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
        hz_10_p50 = []
        hz_10_category = []
        con_p50 = []
        con_category = []
        hz_20_p50 = []
        hz_20_category = []
        hz_50_p50 = []
        hz_50_category = []
        hz_100_p50 = []
        hz_100_category = []
        hz_200_p50 = []
        hz_200_category = []

        for name, group in grouped:
            if group['current'].max() > cutoff:
                if group['condition'].iat[0] == '10':
                    hz_10_p50.append(float(group['p50'].iat[0]))
                    hz_10_category.append(10)
                if group['condition'].iat[0] == '20':
                    hz_20_p50.append(float(group['p50'].iat[0]))
                    hz_20_category.append(20)
                if group['condition'].iat[0] == '50':
                    hz_50_p50.append(float(group['p50'].iat[0]))
                    hz_50_category.append(50)
                if group['condition'].iat[0] == '100':
                    hz_100_p50.append(float(group['p50'].iat[0]))
                    hz_100_category.append(100)
                if group['condition'].iat[0] == 'con':
                    con_p50.append(float(group['p50'].iat[0]))
                    con_category.append('con')
                if group['condition'].iat[0] == 'con':
                    hz_200_p50.append(float(group['p50'].iat[0]))
                    hz_200_category.append(200)
            else:
                continue

        category = con_category + hz_10_category + hz_20_category + hz_50_category + hz_100_category + hz_200_category
        p50 = con_p50 + hz_10_p50 + hz_20_p50 + hz_50_p50 + hz_100_p50 + hz_200_p50
        
        #adding 5 mmHg to correct for not starting at 0mmHg
        p50 = [i + 5 for i in p50]
        
        df2 = pd.DataFrame({'category': category,
                        'P50' : p50})

        #getting n for each group with cutoff max current applied
        n_grouped = df2.groupby(['category'])
        for name, group in n_grouped:
            len_category = len(group)
            print(name, ':', len_category)

        my_pal = {'con': 'grey', 10: 'maroon', 20:'goldenrod', 50:'navy', 100:'olivedrab', 200:'blueviolet'}
        sns.boxplot(data = df2, x = 'category', y = 'P50', palette = my_pal,  order=["con", 10, 20, 50, 100, 200])
        sns.stripplot(data = df2, x = 'category', y = 'P50', color = 'black', order=["con", 10, 20, 50, 100, 200])
        plt.ylim(0, 50)
        plt.show()

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
    grouped = df.groupby(['condition', 'day', 'trial'])
    hz_10_pipr = []
    hz_10_category = []
    con_pipr = []
    con_category = []
    hz_20_pipr = []
    hz_20_category = []
    hz_50_pipr = []
    hz_50_category = []
    hz_100_pipr = []
    hz_100_category = []
    hz_200_pipr = []
    hz_200_category = []

    for name, group in grouped:
        if group['condition'].iat[0] == 'con':
            con_pipr.append(float(group['pip_resistance'].max()))
            con_category.append('con')
        if group['condition'].iat[0] == '10':
            hz_10_pipr.append(float(group['pip_resistance'].max()))
            hz_10_category.append(10)
        if group['condition'].iat[0] == '20':
            hz_10_pipr.append(float(group['pip_resistance'].max()))
            hz_10_category.append(20)
        if group['condition'].iat[0] == '50':
            hz_10_pipr.append(float(group['pip_resistance'].max()))
            hz_10_category.append(50)
        if group['condition'].iat[0] == '100':
            hz_10_pipr.append(float(group['pip_resistance'].max()))
            hz_10_category.append(100)
        if group['condition'].iat[0] == '200':
            hz_20_pipr.append(float(group['pip_resistance'].max()))
            hz_20_category.append(200)

    category = con_category + hz_10_category + hz_20_category + hz_50_category + hz_100_category + hz_200_category
    pipr = con_pipr + hz_10_pipr + hz_20_pipr + hz_50_pipr + hz_100_pipr + hz_200_pipr

    df2 = pd.DataFrame({'category': category,
                    'pip_r' : pipr})
    my_pal = {'con': 'grey', 10: 'maroon', 20:'goldenrod', 50:'navy', 100:'olivedrab', 200:'blueviolet'}
    sns.boxplot(data = df2, x = 'category', y = 'pip_r', order=["con", 10, 20, 50, 100, 200],  palette=my_pal)
    sns.stripplot(data = df2, x = 'category', y = 'pip_r', color = 'black', order=["con", 10, 20, 50, 100, 200])
    plt.ylim(0, 4)
    plt.show()    

#############################################################################################################
max_current_plot = max_current_box_plot(df)
p50_plot = p50_box_plot(df)
current_time_plot = current_time_correlation(df)
t63 = t63_plot(df)
ssc_peak = ssc_peak_ratio_plot(df)
p50_time_plot = p50_time_correlation(df)
cutoff_plots = p50_current_cutoff_plot(df, (20, 30, 40))
resistances = pip_resistance(df) 
#############################################################################################################
