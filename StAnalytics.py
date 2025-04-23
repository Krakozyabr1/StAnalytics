import matplotlib.pyplot as plt
# from itertools import cycle
import streamlit as st
import functions as fn
import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib
# import wx
# import os
# import io

def make_cat_plot(df, plot_type, x, y, aggregation, target, x_label, y_label, title, colormap='tab10'):

    fig, ax = plt.subplots()

    if plot_type == 'bar plot':
        if y is None:
            sns.countplot(df, x=x, y=y, hue=target, palette=colormap, ax=ax)
        else:
            sns.barplot(df, x=x, y=y, hue=target, palette=colormap, ax=ax)
    elif plot_type == 'pie plot':
        if y is None:
            df[x].value_counts().plot.pie(autopct='%1.1f%%')
            ax.set_ylabel(f'Number of {x}')
        else:
            getattr(df.groupby([x]), aggregation)().plot(kind='pie', y=y, autopct='%1.1f%%', ax=ax)
            ax.set_ylabel(f'{aggregation.capitalize()} of {y}')

    if x_label != '':
        ax.set_xlabel(x_label)
    if y_label != '':
        ax.set_ylabel(y_label)
    if title != '':
        ax.set_title(title)

    return fig

@st.cache_resource
def read_table(path):
    return fn.read_table(path)


@st.cache_resource
def get_df_info_num(df):
    return fn.get_df_info(df)
 

@st.cache_resource
def get_df_info_cat(df):
    return fn.get_df_info(df)


@st.cache_resource
def get_df_info(df):
    return fn.get_df_info(df)


@st.cache_resource
def handle_duplicates(df, handling_method):
    return fn.handle_duplicates(df, handling_method)


@st.cache_resource
def handle_missing(df, handling_method, num_cols, cat_cols):
    return fn.handle_missing(df, handling_method, num_cols, cat_cols)


@st.cache_resource
def handle_outliers(df, detection_method, handling_method, num_cols, z_threshold=3):
    return fn.handle_outliers(df, detection_method, handling_method, num_cols, z_threshold)


st.set_page_config(layout="wide")

left, right = st.columns(2)
with left:
    file_selection_method = st.selectbox("Select file using", options=["Uploaded file", "File Path or URL"])
    with st.form("file_selector_form", clear_on_submit=False):
        if file_selection_method == "File Path or URL":
            file_path = st.text_input("Provide file path or URL:", value="").replace('"', "")
        elif file_selection_method == "Uploaded file":
            file_path = st.file_uploader("Upload a file", type=["csv", "xlsx", "txt"])
        select_file_b = st.form_submit_button("Confirm", type="primary")
    
    if select_file_b:
        st.session_state['df_prepared'] = None
        st.session_state['numerical_plot'] = None

if file_path != "" and file_path is not None:
    with right:
        df = read_table(file_path)

        df_info_num, df_info_cat, num_cols, cat_cols, target_cols = get_df_info(df) 
        if len(num_cols) > 0:
            st.text('Numerical columns info')
            st.dataframe(df_info_num)
        if len(cat_cols) > 0:
            st.text('Cathegorical columns info')
            st.dataframe(df_info_cat)

    if df is not None:
        with left:
            with st.container(border=True):
                with st.container(border=False):
                    st.text('Missing values handling')
                    missing_handling = st.selectbox('Missing values handling:', label_visibility='collapsed',
                                options=['Keep', 'Drop', 'Mean Imputing', 'Median Imputing', 'Mode Imputing', 'Zero Imputing'])

                with st.container(border=False):
                    st.text('Outliers handling')
                    outliers_left, outliers_right = st.columns(2)
                    with outliers_left:
                        outlier_detection = st.selectbox('Detection method', options=['IQR-based', 'z-score'])
                    with outliers_right:
                        outlier_handling = st.selectbox('Handling', options=['Keep', 'Drop', 'Replace with NaN', 'Mean Imputing', 'Median Imputing', 'Mode Imputing', 'High/low Imputing', 'Zero Imputing'])
                    z_threshold_input = st.text_input('Threshold (for z-score method)', value='3')
                    
                    try:
                        z_threshold = float(z_threshold_input)
                    except ValueError:
                        try:
                            z_threshold = float(z_threshold_input.replace(',', '.'))
                        except ValueError:
                            st.warning(f"Invalid input for threshold: '{z_threshold_input}'. Please use a period or comma as the decimal separator. Default value (3) will be used.")
                            z_threshold = 3
                        except Exception as e:
                            st.warning(f"An unexpected issue occurred with the threshold input: {e}\nDefault value (3) will be used.")
                            z_threshold = 3
                    except Exception as e:
                        st.warning(f"An unexpected issue occurred with the initial threshold input: {e}\nDefault value (3) will be used.")
                        z_threshold = 3
                
                with st.container(border=False):
                    dup_count = df.duplicated().sum()
                    if dup_count == 0:
                        st.text('Duplicate rows handling (no duplicates found)')
                    elif dup_count == 1:
                        st.text(f'Duplicate rows handling ({dup_count} duplicate found)')
                    else:
                        st.text(f'Duplicate rows handling ({dup_count} duplicates found)')
                        
                    duplicate_handling = st.selectbox('Duplicate rows handling:', label_visibility='collapsed',
                                options=['Keep', 'Remove'])
                
                prepare_data_button = st.button('Prepare data', type="primary")

            if 'df_prepared' not in st.session_state:
                st.session_state['df_prepared'] = None

            if prepare_data_button:
                df_prepared = handle_duplicates(df, duplicate_handling)
                df_prepared = handle_missing(df_prepared, missing_handling, num_cols, cat_cols)
                df_prepared = handle_outliers(df_prepared, outlier_detection, outlier_handling, num_cols, z_threshold)
                st.session_state['df_prepared'] = df_prepared
        
        plot_left, plot_right = st.columns(2)

        with plot_left:

            if len(num_cols) == 1:
                with st.container(border=True):
                    st.text("Numerical data plots")
                    if len(cat_cols) > 0:
                        plot_type_num = st.selectbox('Plot type', options=['histogram', 'kde', 'histogram+kde',
                                                                        'histogram (cumulative)', 'kde (cumulative)',
                                                                        'histogram+kde (cumulative)', 'box plot', 'violin plot', 'jitter plot'])
                    else:
                        plot_type_num = st.selectbox('Plot type', options=['histogram', 'kde', 'histogram+kde',
                                                                        'histogram (cumulative)', 'kde (cumulative)',
                                                                        'histogram+kde (cumulative)', 'box plot', 'violin plot'])
                    x_num_plot = st.selectbox('Variable', options=num_cols, index=0)
                    target_num_plot = st.selectbox('Target variable (optional)', options=[None] + cat_cols, index=0)
                    plot_num_button = st.button('Plot quantitative data', type='primary')
            elif len(num_cols) > 1:
                with st.container(border=True):
                    st.text("Numerical data plots")
                    if len(cat_cols) > 0:
                        plot_type_num = st.selectbox('Plot type', options=['histogram', 'kde', 'histogram+kde',
                                                                        'histogram (cumulative)', 'kde (cumulative)',
                                                                        'histogram+kde (cumulative)', 'scatterplot', 'box plot', 'violin plot', 'jitter plot'])
                    else:
                        plot_type_num = st.selectbox('Plot type', options=['histogram', 'kde', 'histogram+kde',
                                                                        'histogram (cumulative)', 'kde (cumulative)',
                                                                        'histogram+kde (cumulative)', 'scatterplot', 'box plot', 'violin plot'])
                    if plot_type_num == 'scatterplot':
                        x_num_plot = st.selectbox('X variable', options=num_cols, index=0)
                        y_num_plot = st.selectbox('Y variable', options=num_cols, index=1)
                        if len(target_cols) > 0:
                            target_num_plot = st.selectbox('Target variable (optional)', options=[None] + target_cols, index=0, key='target_num')
                        else:
                            target_num_plot = None

                    elif plot_type_num in ['kde', 'jitter plot']:
                        x_num_plot = st.selectbox('X variable', options=num_cols, index=0)
                        if plot_type_num == 'kde':
                            y_num_plot = st.selectbox('Y variable', options=[None] + num_cols, index=0)
                        else:
                            y_num_plot = st.selectbox('Y variable', options=[None] + cat_cols, index=0)
                        if len(target_cols) > 0:
                            target_num_plot = st.selectbox('Target variable (optional)', options=[None] + target_cols, index=0, key='target_num')
                        else:
                            target_num_plot = None
                    else:
                        x_num_plot = st.selectbox('Variable', options=num_cols, index=0)
                        y_num_plot = None
                        if len(target_cols) > 0:
                            target_num_plot = st.selectbox('Target variable (optional)', options=[None] + target_cols, index=0)
                        else:
                            target_num_plot = None

                    x_label_num = st.text_input('X label', value='')
                    y_label_num = st.text_input('Y label', value='')
                    title_num = st.text_input('Plot title', value='')


                    to_plot_mean = st.checkbox('Plot mean', False)

                    if plot_type_num != "boxplot":
                        to_plot_median = st.checkbox('Plot median', False)
                    else:
                        to_plot_median = False

                    to_plot_z_scores = st.checkbox('Plot z-scores', False)
                    z_scores_to_plot = fn.parse_z_score_string(st.text_input('Z-scores to plot (1 for std):', value='1, 2, 3'))

                    if plot_type_num == 'scatterplot':
                        to_plot_regression = st.checkbox('Plot regression curve', False)
                        regression_order_input = st.text_input('Regression curve order', value=1)
                        try:
                            regression_order = int(regression_order_input)
                        except ValueError:
                            st.warning(f"Invalid input for regression curve order: '{regression_order_input}'. Please use an integer. Default value (1) will be used.")
                            regression_order = 1
                        except Exception as e:
                            st.warning(f"An unexpected issue occurred with the regression curve order input: {e}\nDefault value (1) will be used.")
                            regression_order = 1

                    else:
                        to_plot_regression = False
                        regression_order = None

                    plot_num_button = st.button('Plot', type='primary', key='plot_num')                        

            if len(cat_cols) > 0:
                with st.container(border=True):
                    st.text("Categorical data plots")
                    plot_type_cat = st.selectbox('Plot type', options=['bar plot', 'pie plot'])

                    if plot_type_cat == 'bar plot':
                        x_cat_plot = st.selectbox('X variable', options=cat_cols, index=0, key='x_cat_var')
                        aggregation = None
                        if len(num_cols) > 0:
                            y_cat_plot = st.selectbox('Y variable', options=[None]+num_cols, index=0, key='y_cat_var')
                        else:
                            y_cat_plot = None   

                    elif plot_type_cat == 'pie plot':

                        x_cat_plot = st.selectbox('X variable', options=cat_cols, index=0, key='x_cat_var')

                        if len(num_cols) > 0:
                            y_cat_plot = st.selectbox('Y variable', options=[None]+num_cols, index=0, key='y_cat_var')
                            aggregation = st.selectbox('Aggregation (for Y)', options=['sum', 'mean', 'min', 'max', 'median'])
                        else:
                            y_cat_plot = None
                            aggregation = None

                    if len(target_cols) > 0 and len(cat_cols) > 1:
                        target_cat_plot = st.selectbox('Target variable (optional)', options=[None] + target_cols, index=0, key='target_cat')
                    else:
                        target_cat_plot = None
                    
                    x_label_cat = st.text_input('X label', value='', key='x_label_cat_input')
                    y_label_cat = st.text_input('Y label', value='', key='y_label_cat_input')
                    title_cat = st.text_input('Plot title', value='', key='title_cat_input')

                    plot_cat_button = st.button('Plot', type='primary', key='plot_cat')
        
        with plot_right:

            if len(num_cols) > 0:
                st.subheader("Numerical Plot")
                if plot_num_button:
                    if st.session_state['df_prepared'] is not None:
                        fig_num = fn.make_num_plot(st.session_state['df_prepared'],
                                                plot_type_num,
                                                x_num_plot,
                                                y_num_plot,
                                                target_num_plot,
                                                to_plot_z_scores,
                                                z_scores_to_plot,
                                                to_plot_mean,
                                                to_plot_median,
                                                to_plot_regression,
                                                x_label_num, 
                                                y_label_num,
                                                title_num,
                                                regression_order)
                        st.session_state['numerical_plot'] = fig_num
                    else:
                        st.warning("Please prepare the data first.")

                if 'numerical_plot' in st.session_state and st.session_state['numerical_plot'] is not None:
                    st.pyplot(st.session_state['numerical_plot'])
                else:
                    st.info("Click 'Plot' on the left to generate a numerical plot.")

            if len(cat_cols) > 0:
                st.subheader("Categorical Plot")
                if plot_cat_button:
                    if st.session_state['df_prepared'] is not None:
                        fig_num = make_cat_plot(st.session_state['df_prepared'],
                                                plot_type_cat,
                                                x_cat_plot,
                                                y_cat_plot,
                                                aggregation,
                                                target_cat_plot,
                                                x_label_cat, 
                                                y_label_cat,
                                                title_cat)
                        st.session_state['categorical_plot'] = fig_num
                    else:
                        st.warning("Please prepare the data first.")
                
                if 'categorical_plot' in st.session_state and st.session_state['categorical_plot'] is not None:
                    st.pyplot(st.session_state['categorical_plot'])
                else:
                    st.info("Click 'Plot' on the left to generate a categorical plot.")
                