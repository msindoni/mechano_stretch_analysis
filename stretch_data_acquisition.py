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


def load_dataframe(file):
  with open(file, 'r') as fhand:
    #removes spaces and separates string at \n
    raw_file = fhand.read().strip().split('\n')

  line_index = []
  count = 0
  #finding the lines that are not headers/have text in them/are blank and indexing them
  for line in raw_file:
    if re.search(r'[a-z]+', line) == None:
      line_index.append(count)
    count += 1

  #picking out data lines and adding them to this new list of lists
  processed_file = [raw_file[i].strip().replace(" ", "").split(",") for i in line_index]

  #determining the number of sweeps
  #original file has title (1 line) and each sweep has a header (2 lines)
  nsweeps = int((len(raw_file) - len(processed_file)-1)/2)

  #determining column names based on the length of  processed_file[0]
  if len(processed_file[0]) == 5:
      colnames = ['index','ti','i','tp','p']
  else:
      colnames = ['index','ti','i','tp','p','tv','v']

  df = pd.DataFrame(columns = colnames, data = processed_file)
  df = df.apply(pd.to_numeric)
  df = df.dropna(axis=0)

  #adding in sweeps
  datapoint_per_sweep = len(df) / nsweeps
  df['sweep'] = np.repeat(np.arange(nsweeps), datapoint_per_sweep)

  #converting values to more user friendly units
  df['p'] = df['p'] / 0.02
  df['ti'] *= 1000
  df['i'] *= 1e12
  df['tp'] *= 1000
  return(df)

def sigmoid(x, x0, k):
    y = 1 / (1 + np.exp(-k*(x-x0)))
    return y

def isolate_sweeps(df):

  def onselect_function(min_value, max_value):
    minmax_list.append(min_value)
    minmax_list.append(max_value)
    return min_value, max_value

  #isolate each sweep
  grouped = df.groupby('sweep')
  keep_list = []
  triage_list = []
  minmax_list = []
  baseline_i_master_list = []
  for name, group in grouped:
    fig, ax = plt.subplots()
    ax.plot(group.ti, group.i, color = 'pink')
    plt.xlim(4000, 5400)

    #spanselector specific details
    span =SpanSelector(
    ax,
    onselect = onselect_function,
    direction = 'horizontal',
    useblit = True,
    span_stays = False,
    button = 1,
    rectprops = dict(alpha = 0.3, facecolor = 'orange'))
    plt.show()
    baseline_i = group.loc[group['ti'].between(minmax_list[0], minmax_list[1], inclusive = True)]
    baseline_i = baseline_i['i'].mean() 

    print(baseline_i)

    #go through each trace and determine if it should be kept using a gui
    message = 'Keep?'
    title = 'GfG - EasyGUI'
    output = ynbox(message, title)
    print(output)
    
    #only keeps the sweep and baseline average if the overall trace looks good
    if output:
      keep_list.append(name)
      baseline_i_master_list.append(baseline_i)
    else:
      triage_list.append(name)
  print(keep_list)
  print(triage_list)
  print(baseline_i_master_list)
  #create a dataframe with only the sweeps that are useable
  df_isolated = df.loc[df['sweep'].isin(keep_list)]

  #going through and baseline subtracting the currents
  grouped = df_isolated.groupby('sweep')
  baseline_sub_list = []
  index = 0
  for name, group in grouped:
    i_list = group['i'].to_list()
    for i in i_list:
      #subtracting baseline average from every datapoint and adding to a list
      baseline_sub =  i - baseline_i_master_list[index]
      baseline_sub_list.append(baseline_sub)
    index += 1

  #replacing original i with baseline subtracted i
  df_isolated['i'] = baseline_sub_list
  print(df_isolated)
  return df_isolated

def max_current_df(df_isolated):
  #defining the onselect function to add a min and max value to a later list
  def onselect_function(min_value, max_value):
    minmax_list.append(min_value)
    minmax_list.append(max_value)
    return min_value, max_value

  #grouping by sweep and selecting time to isolate max current
  grouped = df_isolated.groupby('sweep')
  minmax_list = []
  max_current = []
  pressure_step = []
  for name, group in grouped:
    fig, ax = plt.subplots()
    ax.plot(group.ti, group.i, color = 'pink')
    plt.xlim(4900, 5400)

    #spanselector specific details
    span =SpanSelector(
    ax,
    onselect = onselect_function,
    direction = 'horizontal',
    useblit = True,
    span_stays = False,
    button = 1,
    rectprops = dict(alpha = 0.3, facecolor = 'orange'))
    plt.show()

    #using the times from span selector to isolate df and get min (max) value from each and add them to a list and pressure to another
    min_values = group.loc[group['ti'].between(minmax_list[0], minmax_list[1], inclusive = True)]
    min_i = min_values['i'].min()
    pressure_step.append(name * 5)
    max_current.append(min_i * -1)
    minmax_list.clear()
  print(pressure_step)
  print(max_current)

  sns.scatterplot(x = pressure_step, y = max_current)
  plt.show()
  #combining max current and pressure lists into a dataframe
  df_max_currents = pd.DataFrame({'current' : max_current,
                      'mmHg' : pressure_step})

  return df_max_currents

def p50_curve(df_max_currents):
  
  print(df_max_currents)
  #create a new column with normalized max currents
  df = df_max_currents
  df['norm_i'] = df_max_currents['current'] / df_max_currents['current'].max()
  print(df)
  #generates the calculated p50 curve values
  popt, pcov = curve_fit(sigmoid, df.mmHg, df.norm_i)
  x = np.linspace(df['mmHg'], max(df['mmHg']), 100)
  y = sigmoid(x, *popt)

  #plotting the current v pressure to get the p50 curve
  plt.plot(x, y, color = 'red')
  plt.scatter(df['mmHg'], df['norm_i'], color = 'blue')
  plt.xlabel("Pressure (mmHg)")
  plt.ylabel("Open Channel %")
  plt.show()
  #popt gives the p50 and slope (I believe) at the p50
  print(popt)
  return popt
  
def ssc_peak_ratio(df_isolated, df_max_currents):
  df = df_max_currents
  #go through each sweep and determine steady state current
  grouped = df_isolated.groupby('sweep')
  ssc_list = []
  for name, group in grouped:
    #ssc come from the 35ms averager near the end of the sweep
    steady_state = group.loc[(group['ti'] > 5250) & (group['ti'] < 5285) ]
    avg_ssc = steady_state['i'].mean()
    ssc_list.append(avg_ssc)

  #make a new column for ssc and ssc/peak ratio
  df['ssc'] = ssc_list 
  df['ssc/peak'] = df['ssc'] / df['current'] * -1
  df = df.drop([0])
  print(df['ssc/peak'].to_string(index = False))

  sns.scatterplot(data = df, x = 'mmHg', y = 'ssc/peak')
  plt.show()

def t63(df_isolated, df_max_currents):
  grouped = df_isolated.groupby('sweep')
  iteration_index = 0
  t63_master_list = []
  mmHg_list = []

  #Goes through and averages currents by whole miliseconds to make finding a t63 less subject to noise
  for name, group in grouped:
    time_index = 0
    avg_i = []
    time = []
    df_time_list = group['i'].to_list()
    #goes through the list in chunks of ten to the  be able to average
    for group in grouper(df_time_list, 10):
      avg = sum(list(group)) / len(list(group))
      avg_i.append(avg)

      #creates a new time list so times match avg i indexing
      time.append(time_index)
      time_index += 1


    avg_i = avg_i[5000:5300]
    time =time[5000:5300]
    #makes the max peak rounded upward to make determing where the peak is easier
    max_i = math.ceil(min(avg_i[:100]))
    peak_index = 0
    for i in avg_i:
      # print(i)
      if i > max_i:
        peak_index += 1
      if i < max_i:
        break
    #determining where the t63 values is
    t63_value = max_i * 0.63
    #index is time (ms) after the peak
    t63_index = 0 
    past_peak_list = avg_i[peak_index:]
    for i in past_peak_list:
      if i < t63_value:
        t63_index += 1
      if i > t63_value:
        break
    t63_master_list.append(t63_index)

    #plotting the peak and t63 and the current graph to check
    plt.plot(time, avg_i, linewidth = 3)
    plt.scatter(peak_index + 5000, max_i, s = 500)
    #for plotting, need to add the peak index to get it to align with everyhing else
    plt.scatter(t63_index + peak_index + 5000, t63_value, s = 500)
    plt.axis('off')
    plt.show()

    #increase iteration by one to get to the next group/sweep
    iteration_index += 1
  #adds t63 values to the now master df_max_currents dataframe
  df_max_currents['t63'] = t63_master_list
  return df_max_currents


##################################################################################################################
df = load_dataframe('C:/Users/mjs164/Box Sync/Data/initial_frequency_screen/stretch/2022_04_18_n2a_stretch_100hz_OWG_maxamp/2022_04_18_n2a_stretch_con_25.asc')
df_isolated = isolate_sweeps(df)
df_max_currents = max_current_df(df_isolated)
p50 = p50_curve(df_max_currents)
ssc_ratio = ssc_peak_ratio(df_isolated, df_max_currents)
t63_added = t63(df_isolated, df_max_currents)

###################################################################################################################
t63_added.to_clipboard(excel = True, index = False, header = None) 
print(t63_added)
print(p50)