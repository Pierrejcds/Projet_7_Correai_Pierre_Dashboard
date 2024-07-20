import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

import streamlit as st
import requests
import shap
import json
import joblib
import os

MODEL_URI = "https://api-flask-p7-45d9629e815b.herokuapp.com/prediction"
THRESHOLD = 0.6

@st.cache_data
def loading_df(directory_path):
    df = pd.read_csv(
        filepath_or_buffer=directory_path,
        sep=';',
        encoding='utf-8'
    )
    df['user_id'] = df.index
    X = df.drop(columns=['TARGET', 'user_id'])
    return df, X


@st.cache_data
def showing_waterfall(user_id):
    shap_values = joblib.load("shap_values_lgbm.joblib")
    plt.figure()
    shap.plots.waterfall(shap_values[user_id])
    st.pyplot(plt.gcf())


def request_prediction(MODEL_URI, df, user_id):
    client_data = df[df['user_id'] == user_id]
    client_data = client_data.drop(columns=['TARGET', 'user_id'])
    client_data_dict = client_data.iloc[0].to_dict()

    # Convertir NaN en None
    client_data_dict = {key: (None if pd.isna(value) else value) for key, value in client_data_dict.items()}

    headers = {"Content-Type": "application/json"}
    response = requests.post(MODEL_URI, json=client_data_dict, headers=headers)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(
                response.status_code,
                response.text
            )
        )
    else:
        prediction = response.json()['prediction']

    return prediction


def showing_prediction(pred):
    if pred[0] >= THRESHOLD:
        color = "green"
        text = f"Le crédit peut être accordé avec une probabilité de recouvrement de {int(pred[0] * 100)}%"
        emoji = '👌'
    else:
        color = "red"
        text = f"Le crédit ne devrait pas être accordé car il présente un risque de défaut de {int(pred[1] * 100)}%"
        emoji = '🖐'

    html_code = f"""
        <div style="display: flex; align-items: center;">
            <div style="
                width: 75px;
                height: 75px;
                background-color: {color};
                border-radius: 50%;
                margin-right: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 35px;">
                {emoji}
            </div>
            <div style="
                padding: 10px 20px;
                background-color: #f0f0f0;
                border-radius: 10px;">
                {text}
            </div>
        </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def showing_density(df, column_filter, data, users_filter):
    if users_filter == 'Eligibles':
        df_filter = df.loc[df['TARGET'] == 0]
    elif users_filter == 'Non éligibles':
        df_filter = df.loc[df['TARGET'] == 1]
    else:
        df_filter = df

    density_chart = alt.Chart(df_filter).transform_density(
            column_filter,
            as_=[column_filter, 'density'],
        ).mark_area(
            line={'color': 'blue'}
        ).encode(
            x=column_filter,
            y='density:Q'
        )

    dashed_line = alt.Chart(pd.DataFrame({
        column_filter: [data]
    })).mark_rule(
        color='red',
        strokeDash=[5, 5]
    ).encode(
        x=column_filter
    )
    density_chart += dashed_line

    st.altair_chart(density_chart, use_container_width=True)


def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {background-color: #fcf7c9;}
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title='Dashboard - Prêt à dépenser',
        layout='wide',
        page_icon='icon.png'
    )
    inject_custom_css()
    st.title("Dashboard")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df, X = loading_df(
        directory_path=os.path.join(current_dir, 'df_cleaned_reduced.csv'))

    col_user, col_general = st.columns([3, 7])

    with col_user:
        user_id = st.text_input("Entrez l'id de l'utilisateur : ", value=1)
        user_id = int(user_id)
        data = df.loc[df['user_id'] == user_id, X.columns]
        st.write(data.T)

        st.markdown('---')

        users_filter = st.selectbox(
            "Choisissez une population à afficher :",
            ("Tous", "Eligibles", "Non éligibles")
        )

        columns_filter = st.selectbox(
            "Choisissez une variable à afficher :",
            (X.columns)
        )

    with col_general:
        pred = request_prediction(MODEL_URI, df, user_id)
        showing_prediction(pred)
        showing_waterfall(user_id)
        showing_density(
            df,
            columns_filter,
            data[columns_filter].values[0],
            users_filter
        )


if __name__ == '__main__':
    main()
