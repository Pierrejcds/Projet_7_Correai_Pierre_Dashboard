import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import requests
import shap
import joblib
import os

from dashboard import (
    loading_df,
    showing_waterfall,
    request_prediction,
    showing_prediction,
    showing_density,
    inject_custom_css,
)

# unittest.TestCase permet d'hériter de méthode comme assertFalse, etc.
class FonctionsTestDashboard(unittest.TestCase):

    def setUp(self):
        """Créé un mock du dataframe utilisé pour les tests."""
        self.mock_df = pd.DataFrame({
            'EXT_SOURCE_1': [0.08, 3.11, None],
            'PAYMENT_RATE': [0.06, 2.75, 0.05],
            'EXT_SOURCE_3': [0.13, None, 0.72],
            'EXT_SOURCE_2': [0.26, 6.22, 0.55],
            'DAYS_BIRTH': [-9461, -1.67, -19046],
            'ACTIVE_DAYS_CREDIT_MAX': [-103, -6.06, None],
            'AMT_ANNUITY': [24700.50, 3.56, 6750],
            'APPROVED_CNT_PAYMENT_MEAN': [24, 1.00, 4],
            'INSTAL_DAYS_ENTRY_PAYMENT_MEAN': [-315.42, -1.38, -761.66],
            'INSTAL_DPD_MEAN': [0, 0, 0],
            'DAYS_EMPLOYED': [-637, -1.18, -225],
            'PREV_CNT_PAYMENT_MEAN': [24, 1.00, 4],
            'INSTAL_DAYS_ENTRY_PAYMENT_MAX': [-49, -5.44, -727],
            'POS_MONTHS_BALANCE_SIZE': [19, 2.80, 4],
            'BURO_AMT_CREDIT_MAX_OVERDUE_MEAN': [1681.02, 0.00, 0.00],
            'INSTAL_AMT_PAYMENT_SUM': [219625.69, 1.61, 21288.46],
            'CODE_GENDER': [0, 1, 0],
            'AMT_CREDIT': [406597.50, 1.29, 135000],
            'DAYS_ID_PUBLISH': [-2120, -2.91, -2531],
            'ANNUITY_INCOME_PERC': [0.12, 1.32, 0.10],
            'INSTAL_AMT_PAYMENT_MIN': [9251.77, 6.66, 5357.25],
            'AMT_GOODS_PRICE': [351000, 1.12, 135000],
            'ACTIVE_AMT_CREDIT_SUM_LIMIT_MEAN': [15994.28, 8.10, None],
            'ACTIVE_DAYS_CREDIT_ENDDATE_MAX': [780, 1.21, None],
            'DAYS_EMPLOYED_PERC': [0.06, 7.08, 0.01],
            'INCOME_CREDIT_PERC': [0.49, 2.08, 0.50],
            'POS_SK_DPD_DEF_MEAN': [0, 0, 0],
            'CLOSED_DAYS_CREDIT_MAX': [-476, -7.75, -408],
            'PREV_APP_CREDIT_PERC_MIN': [1, 8.68, 1.20],
            'TARGET': [1, 0, 0],
            'user_id': [0, 1, 2]
        })

    @patch('pandas.read_csv')
    def test_loading_df(self, mock_read_csv):
        """
        Test de la fonction loading_df en vérifiant le chargement du
        dataframe et la création de la colonne user_id.
        """
        mock_read_csv.return_value = self.mock_df # Définit dans le setUp.

        df, X = loading_df('dummy_path')

        expected_columns = list(self.mock_df.columns)
        # Vérifie que les noms de colonnes correspondent bien.
        self.assertEqual(list(df.columns), expected_columns)
        self.assertEqual(list(X.columns), list(self.mock_df.columns[:-2]))
        # Vérifie que la colonnes user_id a bien été créée.
        self.assertEqual(df['user_id'].tolist(), [0, 1, 2])

    @patch('joblib.load')
    @patch('shap.plots.waterfall')
    @patch('matplotlib.pyplot.figure')
    @patch('streamlit.pyplot')
    def test_showing_waterfall(self, mock_st_pyplot, mock_plt_figure, mock_shap_waterfall, mock_joblib_load):
        """
        Test de la fonction showing_waterfall en simulant le chargement
        et l'affichage des valeurs SHAP en un graphique waterfall affiché sur
        le dashboard Streamlit.
        """
        shap_values = {0: 'shap_value'}
        mock_joblib_load.return_value = shap_values

        showing_waterfall(0)

        # Confirme l'appel des bons arguments.
        mock_joblib_load.assert_called_once_with("shap_values_lgbm.joblib")
        # Simule la valeur retournée par le dict shap.
        mock_shap_waterfall.assert_called_once_with('shap_value')
        mock_st_pyplot.assert_called_once()

    @patch('requests.post')
    def test_request_prediction(self, mock_post):
        """
        Test de la fonction request_prediction en simulant une demande
        API et vérifiant le traitement de la réponse pour retourner
        la prédiction du modèle.
        """
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'prediction': [0.7, 0.3]}
        mock_post.return_value = mock_response

        prediction = request_prediction('dummy_uri', self.mock_df, 1)

        self.assertEqual(prediction, [0.7, 0.3])

    @patch('streamlit.markdown')
    def test_showing_prediction(self, mock_st_markdown):
        """
        Test de la fonction showing_prediction en vérifiant
        l'affichage des résultats de prédiction en fonctoin du seuil.
        """
        pred = [0.7, 0.3]
        showing_prediction(pred)
        mock_st_markdown.assert_called_once()

        pred = [0.5, 0.5]
        showing_prediction(pred)
        self.assertEqual(mock_st_markdown.call_count, 2)

    @patch('streamlit.altair_chart')
    def test_showing_density(self, mock_st_altair_chart):
        """
        Test de la fonction showing_density en vérifiant l'affichage du
        graphique de densité en fonction de la colonne sélectionnée
        et des données filtrées,
        que la ligne en pointillée s'affiche,
        et enfin si ce dernier s'affiche correctement dans le dashboard Streamlit.
        """
        column_filter = 'EXT_SOURCE_1'
        data = 0.08
        users_filter = 'Eligibles'

        showing_density(self.mock_df, column_filter, data, users_filter)
        mock_st_altair_chart.assert_called_once()

    @patch('streamlit.markdown')
    def test_inject_custom_css(self, mock_st_markdown):
        """
        Test de la fonction inject_custom_css en s'assurant que le code CSS
        soit correctement injecté dans le dashboard Streamlit
        et modifie la couleur de fond.
        """
        inject_custom_css()
        mock_st_markdown.assert_called_once()

if __name__ == '__main__':
    unittest.main()
