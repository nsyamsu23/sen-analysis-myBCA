import streamlit as st
from streamlit_option_menu import option_menu
from dashboard import main as dashboard_main
from data_understanding import main as data_understanding_main
from data_preparation import main as data_preparation_main
from modeling import main as modeling_main
from evaluation import main as evaluasi_main
import streamlit as st

def main():
    with st.sidebar:
        selected = option_menu("Sentiment Analysis : My BCA", ["Dashboard", "Data Understanding", "Data Preparation", "Modeling", "Evaluasi"], 
                               icons=['house', 'book', 'database', 'activity', 'bar-chart'], menu_icon="cast", default_index=0)
    
    if selected == "Dashboard":
        dashboard_main()
    elif selected == "Data Understanding":
        data_understanding_main()
    elif selected == "Data Preparation":
        data_preparation_main()
    elif selected == "Modeling":
        modeling_main()
    elif selected == "Evaluasi":
        evaluasi_main()

if __name__ == "__main__":
    main()
