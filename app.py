import streamlit as st
import pickle

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.components.data_ingestion import *
from src.components.data_transformation import *
from src.components.model_trainer import *

def predict(gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score):
    data = CustomData(
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score
    )
    pred_data = data.get_data_as_dataframe()
    pipeline = PredictPipeline()
    results = pipeline.predict(pred_data)
    return results

def main():
    st.markdown('# End to end Machine Learning Project Template')
    st.markdown('##### This project has the goal to show a demo of a generic machine learning project')
    st.markdown('---')
    st.markdown('### Please input the values of the features to generate the prediction:\n')

    df = {
        'gender' : ['female', 'male'],
        'race_ethnicity' : ['group A', 'group B', 'group C', 'group D','group E'],
        'parental_level_of_education' : ["bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school', 'some high school'],
        'lunch' : ['standard', 'free/reduced'],
        'test_preparation_course' : ['none', 'completed'],
        'reading_score' : [],
        'writing_score' : []
        }

    col1,col2,_= st.columns(3)
    with col1:
        gender_new = st.selectbox('Select your gender:', df['gender'])
        race_ethnicity_new = st.selectbox('Select your race/ethnicity:', df['race_ethnicity'])               
        parental_level_of_education_new = st.selectbox('Select the level of education of the parents:', df['parental_level_of_education'])  
        lunch_new = st.selectbox('Select the type of lunch:', df['lunch'])                        
        test_preparation_course_new = st.selectbox('What test preparation you had?', df['test_preparation_course'])      
    with col2:
        reading_score_new = st.number_input('Please input your reading score:', 0,100)                   
        writing_score_new = st.number_input('Please input your writing score:', 0,100)
    
    if st.button('Predict!'):
        results = predict(gender_new,
                          race_ethnicity_new,
                          parental_level_of_education_new,
                          lunch_new,
                          test_preparation_course_new,
                          reading_score_new,
                          writing_score_new
                          )               
        st.success(f'Your math test score is: {results}')
if __name__ == "__main__":
    main()