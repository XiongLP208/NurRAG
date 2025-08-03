import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages import *
import os
from configs import VERSION
from server.utils import api_address

api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    st.set_page_config(
        "Shenzhen People's Hospital Large Model Nursing Decision Platform",
        # os.path.join("img", "chatchat_icon_blue_square_v2.png"),
        os.path.join("img", "logo02.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
            'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
            'About':  f"""Welcome to use Shenzhen People's Hospital Nursing Clinical Decision Platform V1.0!!"""
        }
    )

    if not chat_box.chat_inited:
        st.toast(
             f"Welcome to the Shenzhen People's Hospital Nursing Intelligent Decision—making Platform V1.0\n\n"
            f"Current language model used is `{LLM_MODEL}`, you can start asking questions."
        )

    pages = {
        "Nursing Decision Mode": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "Nursing Knowledge Base Management": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                # "logo-long-chatchat-trans-v2.png"
                "logo03.png"
            ),
            use_column_width=True
        )
        st.caption(
            f"""<p align="left">Welcome to the Shenzhen People's Hospital Nursing Intelligent Decision—making Platform V1.0! I can conduct 1. Nursing Knowledge Training; 2. Assistance in Understanding Nursing Records; 3. Patient Condition Prediction; 4. Scientific Management of Nursing Knowledge....\n\n</p>"""
            f"""<p align="right">Current Vision: V1.0</p>""",
            unsafe_allow_html=True,
        )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api)
