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
       return pd.to_datetime(x).strftime('%M:%S')

def myFormatter2(x, pos):
       return pd.to_datetime(x).strftime('%S')

def myFormatter3(x, pos):
       return pd.to_datetime(x).strftime('.%f')[:-3]

def myFormatter4(x, pos):
       return pd.to_datetime(x).strftime('%S')

def myFormatter5(x, pos):
       return pd.to_datetime(x).strftime('%M')[1:]


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
    fig, (sync_ax, detail_ax) = plt.subplots(nrows=2, ncols=1, figsize=(6.5, 3.5))
    

    sns.set_palette("deep")
    deep_palette = sns.color_palette("deep", 10)

    # Plot using Seaborn on the axes
    x1_limit = int(hostRec["elapsed"].iloc[-1].seconds) * 10**9
    x2_limit = int(hostRpl["elapsed"].iloc[-1].seconds) * 10**9

    x_limit = max(x1_limit, x2_limit)

    sync_ax.set_title('/user/hand/right/output/haptic for record and replay overlaid')
    sync_ax.set_xlim([0, x_limit])
    sync_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter5))

    detail_ax.set_xlim([39500000000, 43500000000])
    detail_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter2))
    

    trace_df['time'] = trace_df['time'] - trace_df['time'].iloc[0]

    sync_df['time'] = sync_df['time'] - sync_df['time'].iloc[0]
    sync_df = sync_df[sync_df['time'] <= sync_df['time'].iloc[-11] ]

    td = pd.Timedelta(1,'ns')

    # sync_trace_df = trace_df
    # sync_sync_df = sync_df
    # sync_sync_df['time'] = sync_sync_df['time'] - sync_df['time'].iloc[0]
    # sync_sync_df['time'] = sync_sync_df['time'] + sync_trace_df['time'].iloc[0]

    # Sync
    for idx, row in trace_df.iterrows():
        sync_ax.axvline(x=row['time'], color=deep_palette[0], linestyle='solid', linewidth=1.0)

    for idx, row in sync_df.iterrows():
        sync_ax.axvline(x=row['time'], color=deep_palette[1], linestyle='dashed', linewidth=1.0)

    # Detail
    for idx, row in trace_df.iterrows():
        detail_ax.axvline(x=row['time'], color=deep_palette[0], linestyle='solid', linewidth=1.0)

    for idx, row in sync_df.iterrows():
        detail_ax.axvline(x=row['time'], color=deep_palette[1], linestyle='dashed', linewidth=1.0)



    legend_2_elements = [Line2D([0], [0], color=deep_palette[0], linestyle='solid', label='HapticApply-Record'),
                         Line2D([0], [0], color=deep_palette[1], linestyle='dashed', label='HapticApply-Replay')]
    
    

    sync_ax.set_xlabel("Elapsed time [M]")
    detail_ax.set_xlabel("Elapsed time [S]")

    sync_ax.set_yticks([])
    detail_ax.set_yticks([])


    sync_ax.legend(handles=legend_2_elements, loc='best', framealpha=1.0)

    plt.tight_layout()
    plt.savefig('./haptic_trace_rnr_overview.pdf')


def makeDetailGraphs() :
    trace_df = prepTrace('complete_2.0/record-beat_saber_sync.txt', 'h', '/user/hand/right/output/haptic')
    sync_df = prepTrace('complete_2.0/31-12-2023_20-54-33-replay-sync.txt', 'h', '/user/hand/right/output/haptic')

    stop_df = prepTrace('complete_2.0/record-beat_saber_sync.txt', 'k', '/user/hand/right/output/haptic')

    sns.set()
    sns.set_style("darkgrid")

    fig, (error_ax, cum_ax) = plt.subplots(nrows=2, ncols= 1, figsize=(7, 3.5))


    error_ax.set_title('Event error for record and replay overlaid')
    cum_ax.set_title('Cumulative error for record and replay overlaid')

    sns.set_palette("deep")
    deep_palette = sns.color_palette("deep", 10)
    
    error_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter5))
    cum_ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(myFormatter5))

    trace_df['time'] = trace_df['time'] - trace_df['time'].iloc[0]
    sync_df['time'] = sync_df['time'] - sync_df['time'].iloc[0]
    sync_df = sync_df[sync_df['time'] <= sync_df['time'].iloc[-11] ]

    trace_df = trace_df.reset_index()
    sync_df = sync_df.reset_index()

    print(f"record:{len(trace_df)}, replay:{len(sync_df)}")

    diff = trace_df['time'].sub(sync_df['time'])
    acc = diff.cumsum()

    # make it seconds from nano seconds.
    diff = diff / 1e9
    acc = acc / 1e9

    sns.lineplot(y=diff, x=sync_df['time'], ax=error_ax)
    sns.lineplot(y=acc, x=sync_df['time'], ax=cum_ax)


    # error_ax.legend(handles=legend_2_elements, loc='best', framealpha=1.0)

    error_ax.set_ylabel("Time [S]")
    error_ax.set_xlabel("Elaped time [M]")

    cum_ax.set_xlabel("Elaped time [M]")
    cum_ax.set_ylabel("Time [S]")

    plt.tight_layout()
    plt.savefig('./haptic_trace_rnr_detail.pdf')

makeOverviewGraphs()
makeDetailGraphs()