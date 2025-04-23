import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
# import os
# import io

def read_table(path):
    try:
        return pd.read_table(path, sep=None, engine='python')
    except Exception as e:
        try:
            return pd.read_excel(path)
        except Exception as e_excel:
            st.error(f"Error reading file: {path}\n{e}\nExcel Error: {e_excel}")
            return None

def get_df_info_num(df):
    try:
        columns = df.columns
        df_info = []

        for col in columns:
            df_info.append({'Dtype': str(df[col].dtype),
                            'Null num': pd.isna(df[col]).sum(),
                            # 'Unique Count': df[col].nunique(),
                            })
        
        return pd.DataFrame({k: [dic[k] for dic in df_info] for k in df_info[0]}, index=columns)   

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None  

def get_df_info_cat(df):

    if len(df.columns) > 0:
        try:
            columns = df.columns
            df_info = []

            for col in columns:
                df_info.append({'Dtype': str(df[col].dtype),
                                'Null num': pd.isna(df[col]).sum(),
                                'Unique Count': df[col].nunique(),
                                'Count': len(df[col]),
                                'Most Frequent': df[col].mode().iloc[0],
                                'Frequency': df[col].value_counts().iloc[0],
                                })
            
            return pd.DataFrame({k: [dic[k] for dic in df_info] for k in df_info[0]}, index=columns)   

        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None 
        
    return None

def get_df_info(df):
    df_describe = df.describe().transpose()
    df_info = get_df_info_num(df)
    df_info_num = df_info.join(df_describe, how='inner')
    num_cols = [idx for idx in df_info.index if idx in df_info_num.index]
    cat_cols = [idx for idx in df_info.index if idx not in df_info_num.index]
    for column in df.columns:
        if df[column].nunique() < 6:
            if column not in cat_cols:
                cat_cols.append(column)

    df_info_cat = get_df_info_cat(df[cat_cols])

    target_cols = [col for col in cat_cols if df[col].nunique() < 6]

    return df_info_num, df_info_cat, num_cols, cat_cols, target_cols

def handle_duplicates(df, handling_method):
    if handling_method == 'Remove':
        dups = df.duplicated()
        return df[~dups]
    else:
        return df

def handle_missing(df, handling_method, num_cols, cat_cols):
    if handling_method == 'Keep':
        return df
    
    df_copy = df.copy()
    
    if len(num_cols) > 0:
        for col in num_cols:
            if handling_method == 'Drop':
                df_copy.dropna(inplace=True)
            elif handling_method == 'Mean Imputing':
                df_copy[col].fillna(df[col].mean(), inplace=True)
            elif handling_method == 'Median Imputing':
                df_copy[col].fillna(df[col].median(), inplace=True)
            elif handling_method == 'Mode Imputing':
                df_copy[col].fillna(df[col].mode().iloc[0], inplace=True)
            else:
                df_copy[col].fillna(0, inplace=True)
    
    if len(cat_cols) > 0:
        for col in cat_cols:
            df_copy[col].fillna(df[col].mode().iloc[0], inplace=True)
    
    return df_copy

def handle_outliers(df, detection_method, handling_method, num_cols, z_threshold=3):
    if handling_method == 'Keep':
        return df

    df_copy = df.copy()

    if len(num_cols) > 0:
        for col in num_cols:

            if detection_method == 'IQR-based':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                high = Q3 + 1.5 * IQR
                low = Q1 - 1.5 * IQR
            else:
                col_std = df[col].std()
                col_mean = df[col].mean()
                high = col_mean + z_threshold * col_std
                low = col_mean - z_threshold * col_std

            if handling_method == 'Drop':
                df_copy.drop(index=df_copy[(df_copy[col] < low) | (df_copy[col] > high)].index, inplace=True)
            elif handling_method == 'Replace with NaN':
                df_copy.loc[(df[col] < low) | (df[col] > high), col] = np.nan
            elif handling_method == 'Mean Imputing':
                df_copy.loc[(df[col] < low) | (df[col] > high), col] = df[col].mean()
            elif handling_method == 'Median Imputing':
                df_copy.loc[(df[col] < low) | (df[col] > high), col] = df[col].median()
            elif handling_method == 'Mode Imputing':
                df_copy.loc[(df[col] < low) | (df[col] > high), col] = df[col].mode().iloc[0]
            elif handling_method == 'High/low Imputing':
                df_copy.loc[df[col] < low, col] = low
                df_copy.loc[df[col] > high, col] = high
            else:
                df_copy.loc[(df[col] < low) | (df[col] > high), col] = 0.0
    
    return df_copy

def parse_z_score_string(z_score_string):
    try:
        z_scores = [float(z.strip()) for z in z_score_string.split(',')]
        return z_scores
    except ValueError:
        st.warning(f"Invalid z-score input: '{z_score_string}'. Please use comma-separated numbers (e.g., '1, 2.5, 3').")
        return []
    
def plot_line_with_shadow(ax, x, y, color, linestyle='solid', label=None):
    ax.plot(x, y, color='black', alpha=0.75, linewidth=3)
    line, = ax.plot(x, y, color=color, label=label, linestyle=linestyle)
    line_label = label
    return line, line_label


def _plot_mean_median(ax, df, x, y, plot_type, target, colors, method_name='Mean'):
    i = 0
    line_handles = []
    line_labels = []

    if target is None:
        labels = [method_name]
    else:
        target_values = pd.unique(df[target])
        labels = [f'{method_name} for {label}' for label in target_values]


    if method_name == 'Mean':
        linestyle = 'dashed'
    elif method_name == 'Median':
        linestyle = 'solid'

    
    for j in range(len(labels)):
        if target is not None:
            method_x = getattr(df[x][df[target] == target_values[j]], method_name.lower())()
            if plot_type == 'scatterplot':
                method_y = getattr(df[y][df[target] == target_values[j]], method_name.lower())()
        else:
            method_x = getattr(df[x], method_name.lower())()
            if plot_type == 'scatterplot':
                method_y = getattr(df[y], method_name.lower())()

        line_handle, line_label = plot_line_with_shadow(ax, [method_x] * 2, ax.get_ylim(), color=colors[i % len(colors)], label=labels[j], linestyle=linestyle)
        if plot_type == 'scatterplot':
            _, _ = plot_line_with_shadow(ax, ax.get_xlim(), [method_y] * 2, color=colors[i % len(colors)], linestyle=linestyle)

        line_handles.append(line_handle)
        line_labels.append(line_label)
        i += 1
    
    return line_handles, line_labels


def _plot_z_scores(ax, df, x, y, plot_type, target, colors, z_scores_to_plot):
    i = 0
    line_handles = []
    line_labels = []


    if target is None:
        target_values = [None]
    else:
        target_values = pd.unique(df[target])

    for target_value in target_values:
        for z_score in z_scores_to_plot:

            if len(z_scores_to_plot) == 1 and z_scores_to_plot[0] == 1.0:
                label = 'std' if target is None else f'std for {target_value}'
            else:
                label = f'z-score {z_score}' if target is None else f'z-score {z_score} for {target_value}'

            if target is None:
                df_temp = df
            else:
                df_temp = df[df[target] == target_value]

            x_mean = df_temp[x].mean()
            x_std = df_temp[x].std()
            xs = [x_mean - z_score * x_std, x_mean + z_score * x_std]
            
            if plot_type == 'scatterplot':
                y_mean = df_temp[y].mean()
                y_std = df_temp[y].std()
                ys = [y_mean - z_score * y_std, y_mean + z_score * y_std]
            else:
                ys = ax.get_ylim()

            line_handle, line_label = plot_line_with_shadow(ax, [xs[0]]*2, ys, color=colors[i % len(colors)], label=label, linestyle='dotted')
            _, _ = plot_line_with_shadow(ax, [xs[1]]*2, ys, color=colors[i % len(colors)], linestyle='dotted')

            if plot_type == 'scatterplot':
                _, _ = plot_line_with_shadow(ax, xs, [ys[0]]*2, color=colors[i % len(colors)], linestyle='dotted')
                _, _ = plot_line_with_shadow(ax, xs, [ys[1]]*2, color=colors[i % len(colors)], linestyle='dotted')

            line_handles.append(line_handle)
            line_labels.append(line_label)
            i += 1

    return line_handles, line_labels


def make_num_plot(df, plot_type, x, y, target, to_plot_z_scores, z_scores_to_plot, to_plot_mean, to_plot_median, to_plot_regression, x_label, y_label, title, regression_order=1, colormap='tab10'):
    
    colors = matplotlib.colormaps[colormap].colors

    fig, ax = plt.subplots()

    if plot_type == 'histogram':
        plot = sns.histplot(df, x=x, hue=target, palette=colormap, ax=ax)
    elif plot_type == 'kde':
        plot = sns.kdeplot(df, x=x, y=y, hue=target, palette=colormap, ax=ax)
    elif plot_type == 'histogram+kde':
        plot = sns.histplot(df, x=x, hue=target, palette=colormap, kde=True, ax=ax)
    elif plot_type == 'histogram (cumulative)':
        plot = sns.histplot(df, x=x, hue=target, palette=colormap, cumulative=True, ax=ax)
    elif plot_type == 'kde (cumulative)':
        plot = sns.kdeplot(df, x=x, hue=target, palette=colormap, cumulative=True, ax=ax)
    elif plot_type == 'histogram+kde (cumulative)':
        plot = sns.histplot(df, x=x, hue=target, palette=colormap, kde=True, cumulative=True, ax=ax)
    elif plot_type == 'scatterplot':
        if to_plot_regression:
            plot = sns.lmplot(df, x=x, y=y, hue=target, palette=colormap, order=regression_order)
            fig, ax = plot.figure, plot.ax
        else:
            plot = sns.scatterplot(df, x=x, y=y, hue=target, palette=colormap, ax=ax)
    elif plot_type == 'box plot':
        plot = sns.boxplot(df, x=x, hue=target, palette=colormap, ax=ax)
    elif plot_type == 'violin plot':
        plot = sns.violinplot(df, x=x, hue=target, palette=colormap, ax=ax)
    elif plot_type == 'jitter plot':
        plot = sns.stripplot(df, x=x, y=y, hue=target, palette=colormap, ax=ax)
    
    
    
    if x_label != '':
        ax.set_xlabel(x_label)
    if y_label != '':
        ax.set_ylabel(y_label)
    if title != '':
        ax.set_title(title)


    seaborn_handles, line_handles = [], []
    line_labels, seaborn_labels = [], []

    if target is not None:
            if not to_plot_regression and y != target:
                seaborn_handles = ax.get_children()[-2].legend_handles
                seaborn_labels = list(df[target].unique())

    if to_plot_mean:
        line_handle, line_label = _plot_mean_median(ax, df, x, y, plot_type, target, colors, 'Mean')
        line_labels.extend(line_label)
        line_handles.extend(line_handle)
    
    if to_plot_median:
        line_handle, line_label = _plot_mean_median(ax, df, x, y, plot_type, target, colors, 'Median')
        line_labels.extend(line_label)
        line_handles.extend(line_handle)

    if to_plot_z_scores:
        line_handle, line_label = _plot_z_scores(ax, df, x, y, plot_type, target, colors, z_scores_to_plot)
        line_labels.extend(line_label)
        line_handles.extend(line_handle)

    all_handles = seaborn_handles + line_handles
    all_labels = seaborn_labels + line_labels

    if all_labels:
        ax.legend(handles=all_handles, labels=all_labels)

    return fig