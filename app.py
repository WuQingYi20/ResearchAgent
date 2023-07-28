import streamlit as st

from agent import initialize_research_agent

def main():
    st.set_page_config(page_title="Research Agent", page_icon=":books:")
    st.header("Research Agent")

    query = st.text_input("Enter your research objective")
    if query:
        st.write("doing research for", query)

        agent = initialize_research_agent()
        result = agent({"input": query})

        st.info(result['output'])

if __name__ == '__main__':
    main()