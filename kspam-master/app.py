import streamlit as st
from catboost import CatBoostClassifier

# import train_catboost
import train_logreg
import train_naive_bayes


cb_model = CatBoostClassifier()
cb_model.load_model("cb.model.bin")

txt = st.text_area('Text to analyze', "")
st.write('Result by catboost:', cb_model.predict([txt]))
st.write('Result by logreg:', train_logreg.predict([txt]))
st.write('Result by naive bayes:', train_naive_bayes.pipeline.predict([txt])[0])
