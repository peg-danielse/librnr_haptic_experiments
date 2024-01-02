import os
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib as mpl
import seaborn as sns

def trace_to_df(filename : str) -> pd.DataFrame:
    with open(filename, 'r') as file:
        datatypes = {'s', 'v', 'p', 'f', 'b', 'h', 'k'}

        default_values = {'time': 0, 'controller_index' : 0, 'changed' : 0, 'isActive' : 0 , "lastChanged" : 0, 'vtype' : 0,
                         'o.x' : 0.0, 'o.y' : 0.0, 'o.z' : 0.0, 'o.w' : 0.0, 
                         'p.x' : 0.0, 'p.y' : 0.0, 'p.z' : 0.0, 
                         'u' : 0.0, 'r' : 0.0, 'd' : 0.0, 'l' : 0.0, 
                         'type' : '-', 'input' : 'default_input', 'l.basespace' : 'default_basespace'}

        trace_dict = {key: [] for key in datatypes}

        # Get individual lines into the correct datatype buckets
        for line in file:
            for char in line:
                if not char.isdigit() and not char.isspace():
                    trace_dict[char].append(line.strip().split(' '))
                    break

        # handle data cleanup and type assignments for space
        space_names = ['time', 'type', 'input', 'errorcase', 'o.x', 'o.y', 'o.z', 'o.w', 'p.x', 'p.y', 'p.z', 'l.basespace']
        space_dataframe = pd.DataFrame(trace_dict['s'], columns=space_names)
        space_dataframe.drop('errorcase', axis=1, inplace=True)

        space_dataframe[['input','type', 'l.basespace']] = space_dataframe[['input','type', 'l.basespace']].astype(str)
        space_dataframe[['o.x', 'o.y', 'o.z', 'o.w', 'p.x', 'p.y', 'p.z']] = space_dataframe[['o.x', 'o.y', 'o.z', 'o.w', 'p.x', 'p.y', 'p.z']].astype(float)
        
        space_dataframe['time'] = space_dataframe[['time']].apply(pd.to_numeric)

        # handle data cleanup and type assignments for view
        view_names = ['time', 'type', 'input', 'errorcase', 'o.x', 'o.y', 'o.z', 'o.w', 'p.x', 'p.y', 'p.z', 'u', 'r', 'd', 'l', 'vtype', 'controller_index']
        view_dataframe = pd.DataFrame(trace_dict['v'], columns=view_names)
        view_dataframe.drop('errorcase', axis=1, inplace=True)

        view_dataframe[['input','type']] = view_dataframe[['input','type']].astype(str)
        view_dataframe[['o.x', 'o.y', 'o.z', 'o.w', 'p.x', 'p.y', 'p.z']] = view_dataframe[['o.x', 'o.y', 'o.z', 'o.w', 'p.x', 'p.y', 'p.z']].astype(float)
        view_dataframe[['u', 'r', 'd', 'l']] = view_dataframe[['u', 'r', 'd', 'l']].astype(float)

        view_dataframe[['time', 'vtype', 'controller_index']] = view_dataframe[['time', 'vtype', 'controller_index']].apply(pd.to_numeric)

        # handle data cleanup and type assignments for position
        position_names = ['time', 'type', 'input', 'errorcase', 'changed','isActive','lastChanged','p.x', 'p.y']
        position_dataframe = pd.DataFrame(trace_dict['p'], columns=position_names)
        position_dataframe.drop('errorcase', axis=1, inplace=True)

        position_dataframe[['input','type']] = position_dataframe[['input','type']].astype(str)
        position_dataframe[['changed','isActive','lastChanged']] = position_dataframe[['changed','isActive','lastChanged']].apply(pd.to_numeric)
        position_dataframe[['p.x', 'p.y']] = position_dataframe[['p.x', 'p.y']].astype(float)
        
        position_dataframe[['time']] = position_dataframe[['time']].apply(pd.to_numeric)
        
        # handle data cleanup and type assignments for float
        float_names = ['time', 'type', 'input', 'errorcase', 'changed','isActive','lastChanged','p.x']
        float_dataframe = pd.DataFrame(trace_dict['f'], columns=float_names)
        float_dataframe.drop('errorcase', axis=1, inplace=True)

        float_dataframe[['input','type']] = float_dataframe[['input','type']].astype(str)
        float_dataframe[['time', 'changed','isActive','lastChanged']] = float_dataframe[['time', 'changed','isActive','lastChanged']].apply(pd.to_numeric)
        float_dataframe[['p.x']] = float_dataframe[['p.x']].astype(float)

        # handle data cleanup and type assignments for boolean
        boolean_names = ['time', 'type', 'input', 'errorcase', 'changed','isActive','lastChanged','p.x']
        boolean_dataframe = pd.DataFrame(trace_dict['b'], columns=boolean_names)
        boolean_dataframe.drop('errorcase', axis=1, inplace=True)

        boolean_dataframe[['input','type']] = boolean_dataframe[['input','type']].astype(str)
        boolean_dataframe[['time', 'changed','isActive','lastChanged']] = boolean_dataframe[['time', 'changed','isActive','lastChanged']].apply(pd.to_numeric)
        boolean_dataframe[['p.x']] = boolean_dataframe[['p.x']].astype(float)


        # handle data cleanup and type assignments for haptic
        haptic_names = ['time', 'type', 'input', 'errorcase', 'p.x']
        haptic_dataframe = pd.DataFrame(trace_dict['h'], columns=haptic_names)
        haptic_dataframe.drop('errorcase', axis=1, inplace=True)

        haptic_dataframe[['input','type']] = haptic_dataframe[['input','type']].astype(str)
        haptic_dataframe[['time']] = haptic_dataframe[['time']].apply(pd.to_numeric)
        haptic_dataframe[['p.x']] = haptic_dataframe[['p.x']].astype(float)

        # handle data cleanup and type assignments for haptic
        hapticS_names = ['time', 'type', 'input', 'errorcase', 'p.x']
        hapticS_dataframe = pd.DataFrame(trace_dict['k'], columns=hapticS_names)
        hapticS_dataframe.drop('errorcase', axis=1, inplace=True)

        hapticS_dataframe[['input','type']] = hapticS_dataframe[['input','type']].astype(str)
        hapticS_dataframe[['time']] = hapticS_dataframe[['time']].apply(pd.to_numeric)
        hapticS_dataframe[['p.x']] = hapticS_dataframe[['p.x']].astype(float)

        # Concat the data into one dataframe
        trace_df = pd.concat([hapticS_dataframe, haptic_dataframe, view_dataframe, space_dataframe, boolean_dataframe, float_dataframe, position_dataframe], ignore_index=True)

        # sort on time and fill na with default value for that column
        trace_df.sort_values(by=['time'], inplace=True)
        trace_df.reset_index(drop=True, inplace=True)

        trace_df.fillna(default_values, inplace=True)
        trace_df[['changed','isActive','lastChanged']] = trace_df[['changed','isActive','lastChanged']].astype("int64")
        trace_df[['vtype', 'controller_index']] = trace_df[['vtype', 'controller_index']].astype("int64")
        
        return trace_df
    
def df_to_trace(filename : str, trace_df: pd.DataFrame) :
    with open(filename, 'w') as file:
        for i in trace_df.index :
            columns = []
            
            if(trace_df['type'][i] == 's'):
                columns = ['time', 'type', 'input', 'o.x', 'o.y', 'o.z', 'o.w', 'p.x', 'p.y', 'p.z', 'l.basespace']
            elif(trace_df['type'][i] == 'v'):
                columns = ['time', 'type', 'input', 'o.x', 'o.y', 'o.z', 'o.w', 'p.x', 'p.y', 'p.z', 'u', 'r', 'd', 'l', 'vtype', 'controller_index']
            elif(trace_df['type'][i] == 'f'):
                columns = ['time', 'type', 'input', 'changed','isActive','lastChanged','p.x']
            elif(trace_df['type'][i] == 'p'):
                columns = ['time', 'type', 'input', 'changed','isActive','lastChanged','p.x', 'p.y']
            elif(trace_df['type'][i] == 'b'):
                columns = ['time', 'type', 'input', 'changed','isActive','lastChanged','p.x']
            elif(trace_df['type'][i] == 'h'):
                columns = ['time', 'type', 'input', 'p.x']
            
            line = ""
            for column_name in columns:
                line += str(trace_df[column_name][i]) + " "

            file.write(line + "\n")

def myFormatter(x, pos):
       return pd.to_datetime(x).strftime('%M:%S.%f')[:-3]

def myFormatter2(x, pos):
       return pd.to_datetime(x).strftime('%M:%S')

def myFormatter3(x, pos):
       return pd.to_datetime(x).strftime('.%f')[:-3]

def myFormatter4(x, pos):
       return pd.to_datetime(x).strftime('%S')


def makeFrameRateDf(path : str) -> pd.DataFrame :
    column_names = ['Time','Framerate']
    
    df = pd.read_csv(path)
    df = df[[' Time','Framerate           ']]
    
    df.columns = column_names

    df.astype({"Framerate" : 'float32'})
    df['dateTime'] = pd.to_datetime(df['Time'], dayfirst=True)

    # filter out null stuff
    df = df[df['Framerate'] != 0.0]

    position = df.columns.get_loc('dateTime')
    df['elapsed'] =  df.iloc[1:, position] - df.iat[0, position]
    return df

def prepTrace(path : str, type : str, input : str) -> pd.DataFrame :
    trace_df = trace_to_df(path)
    trace_df = trace_df[trace_df['type'] == type]
    trace_df = trace_df[['time', 'input', 'p.x']]
    trace_df = trace_df[trace_df['input'] == input]
    
    return trace_df

def prepTrace1(path : str) -> pd.DataFrame :
    trace_df = trace_to_df(path)
    trace_df = trace_df[trace_df['type'] == 'h']
    trace_df = trace_df[['time', 'input', 'p.x']]
    trace_df = trace_df[trace_df['input'] == '/user/hand/left/output/haptic']
    
    return trace_df


def makeOverviewGraphs() :  
    hostRec = makeFrameRateDf('./complete_2.0/record-HardwareMonitoring.txt')
    hostRpl = makeFrameRateDf('./complete_2.0/replay-HardwareMonitoring.txt')
    trace_df = prepTrace('complete_2.0/record-beat_saber_sync.txt', 'h', '/user/hand/right/output/haptic')
    sync_df = prepTrace('complete_2.0/31-12-2023_20-54-33-replay-sync.txt', 'h', '/user/hand/right/output/haptic')

    sns.set()
    sns.set_style("darkgrid")
    # Create a figure with a narrow width
    fig, (record_ax, replay_ax, sync_ax) = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))

    sns.set_palette("deep")
    deep_palette = sns.color_palette("deep", 10)

    # Plot using Seaborn on the axes

    x1_limit = int(hostRec["elapsed"].iloc[-1].seconds) * 10**9
    x2_limit = int(hostRpl["elapsed"].iloc[-1].seconds) * 10**9

    x_limit = max(x1_limit, x2_limit)

    sns.lineplot(ax=record_ax, data=hostRec, x="elapsed", y="Framerate")
    record_ax.set_title('Recording of /user/hand/right/output/haptic and Framerate')
    record_ax.set_xlim([0, x_limit])
    record_ax.set_ylim([0, 80])
    record_ax.set_xlabel("Time Elapsed")

    sns.lineplot(ax=replay_ax, data=hostRpl, x="elapsed", y="Framerate")
    replay_ax.set_title('Replay of /user/hand/right/output/haptic and Framerate')
    replay_ax.set_xlim([0, x_limit])
    replay_ax.set_ylim([0, 80])
    replay_ax.set_xlabel("Time Elapsed")

    sync_ax.set_title('/user/hand/right/output/haptic for record and replay overlayed')
    sync_ax.set_yticks([])
    sync_ax.set_xlabel("Time Elapsed")

    record_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter))
    replay_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter))
    sync_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter))

    trace_df['time'] = trace_df['time'] - trace_df['time'].iloc[0] # first h should sync on: 31-12-2023_20-34-52, first k on: 31-12-2023_20-34-35
    trace_df['time'] = trace_df['time'] + (hostRec[hostRec['dateTime'] == '2023-12-31 20:34:52'].iloc[0]['elapsed'].seconds * 10**9)

    sync_df['time'] = sync_df['time'] - sync_df['time'].iloc[0]
    sync_df['time'] = sync_df['time'] + (hostRpl[hostRpl['dateTime'] == '2023-12-31 20:55:03'].iloc[0]['elapsed'].seconds * 10**9)

    td = pd.Timedelta(1,'ns')

    for idx, row in trace_df.iterrows():
        record_ax.axvline(x=row['time'], color=deep_palette[3], linestyle='dashed', linewidth=1.0)

    for idx, row in sync_df.iterrows():
        replay_ax.axvline(x=row['time'], color=deep_palette[3], linestyle='dashed', linewidth=1.0)

    sync_trace_df = trace_df
    sync_sync_df = sync_df
    sync_sync_df['time'] = sync_sync_df['time'] - sync_df['time'].iloc[0]
    sync_sync_df['time'] = sync_sync_df['time'] + sync_trace_df['time'].iloc[0]

    for idx, row in sync_trace_df.iterrows():
        sync_ax.axvline(x=row['time'], color=deep_palette[1], linestyle='solid', linewidth=1.0)

    for idx, row in sync_sync_df.iterrows():
        sync_ax.axvline(x=row['time'], color=deep_palette[4], linestyle='dashed', linewidth=1.0)

    legend_elements = [Line2D([0], [0], color=deep_palette[0], lw=1, label='Framerate'),
                    Line2D([0], [0], color=deep_palette[3], linestyle='dashed', label='HapticApply')]

    legend_2_elements = [Line2D([0], [0], color=deep_palette[1], linestyle='solid', label='HapticApply-Record'),
                        Line2D([0], [0], color=deep_palette[4], linestyle='dashed', label='HapticApply-Replay')]

    record_ax.legend(handles=legend_elements, loc='best', framealpha=1.0)
    sync_ax.legend(handles=legend_2_elements, loc='best', framealpha=1.0)

    plt.tight_layout()
    plt.savefig('./haptic_trace_rnr_overview.pdf')


def makeDetailGraphs() :
    trace_df = prepTrace('complete_2.0/record-beat_saber_sync.txt', 'h', '/user/hand/right/output/haptic')
    sync_df = prepTrace('complete_2.0/31-12-2023_20-54-33-replay-sync.txt', 'h', '/user/hand/right/output/haptic')

    stop_df = prepTrace('complete_2.0/record-beat_saber_sync.txt', 'k', '/user/hand/right/output/haptic')

    sns.set()
    sns.set_style("darkgrid")

    fig, axs = plt.subplots(nrows=3, ncols= 1, figsize=(8, 8))

    # Create the first column of figures
    record_ax = axs[0]
    replay_ax = axs[1]
    sync_ax = axs[2]

    sns.set_palette("deep")
    deep_palette = sns.color_palette("deep", 10)

    record_ax.set_title('/user/hand/right/output/haptic data shape')
    record_ax.set_yticks([])
    record_ax.set_xlim([53000000000, 53200000000])
    
    replay_ax.set_title('/user/hand/right/output/haptic overlayed data shape')
    replay_ax.set_yticks([])
    replay_ax.set_xlim([39500000000, 43500000000])
    
    sync_ax.set_title('/user/hand/right/output/haptic for the HapticStop event')
    sync_ax.set_yticks([])
    sync_ax.set_xlabel("Time Elapsed")
    sync_ax.set_xlim([20000000000, 40000000000])

    record_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter3))
    replay_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter4))
    sync_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter2))

    trace_df['time'] = trace_df['time'] - trace_df['time'].iloc[0]

    sync_df['time'] = sync_df['time'] - sync_df['time'].iloc[0]

    td = pd.Timedelta(1,'ns')

    # Record detail
    for idx, row in trace_df.iterrows():
        record_ax.axvline(x=row['time'], color=deep_palette[3], linestyle='solid', linewidth=1.0)

    # Record and replay detail
    for idx, row in trace_df.iterrows():
        replay_ax.axvline(x=row['time'], color=deep_palette[3], linestyle='solid', linewidth=1.0)

    for idx, row in sync_df.iterrows():
        replay_ax.axvline(x=row['time'], color=deep_palette[4], linestyle='dashed', linewidth=1.0)
    

    # Record HapticStop
    for idx, row in stop_df.iterrows():
        sync_ax.axvline(x=row['time'], color=deep_palette[3], linestyle='dashed', linewidth=1.0)


    legend_elements = [Line2D([0], [0], color=deep_palette[3], linestyle='solid', label='HapticApply-record'),
                       Line2D([0], [0], color=deep_palette[4], linestyle='dashed', label='HapticApply-replay')]

    legend_2_elements = [Line2D([0], [0], color=deep_palette[3], linestyle='dashed', label='HapticStop-record')]

    record_ax.legend(handles=legend_elements, loc='best', framealpha=1.0)
    sync_ax.legend(handles=legend_2_elements, loc='lower right', framealpha=1.0)

    plt.tight_layout()
    plt.savefig('./haptic_trace_rnr_detail.pdf')

makeDetailGraphs()