import pandas as pd
import streamlit as st
from streamlit_extras.let_it_rain import rain

from csv_upload import build_batch_results, find_text_column
from sentiment_utils import analyze_text


st.title("A Sentiment Analysis WebApp .")
st.text("developed by Anop Singh")

message = st.text_area("Please Enter your text")

if st.button("Analyze the Sentiment"):
    if not message.strip():
        st.warning("Please enter text before analyzing.")
    else:
        result = analyze_text(message)
        polarity = result["polarity"]

        if result["label"] == "Negative":
            st.warning("The entered text has negative sentiments associated with it: " + str(polarity))
            rain(
                emoji="😥",
                font_size=25,
                falling_speed=3,
                animation_length="infinite",
            )
        elif result["label"] == "Positive":
            st.success("The entered text has positive sentiments associated with it: " + str(polarity))
            rain(
                emoji="😍",
                font_size=25,
                falling_speed=3,
                animation_length="infinite",
            )
        else:
            st.info("The entered text has neutral sentiments associated with it: " + str(polarity))

        st.success(
            f"Polarity: {result['polarity']}; Subjectivity: {result['subjectivity']}"
        )

st.divider()
st.subheader("Batch analyze CSV")
st.caption("Upload a CSV with review text, text, or Xquik Tweet Text columns.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read CSV: {error}")
    else:
        text_column = find_text_column(uploaded_df)
        if text_column is None:
            st.error("No text column found. Use review, text, tweet, feedback, or Tweet Text.")
        else:
            results_df = build_batch_results(uploaded_df)
            if results_df.empty:
                st.warning("No non-empty text rows found in the selected column.")
            else:
                st.caption(f"Analyzed {len(results_df)} rows from {text_column}.")
                st.dataframe(results_df, use_container_width=True)
                st.download_button(
                    "Download results as CSV",
                    results_df.to_csv(index=False),
                    "sentiment_results.csv",
                    "text/csv",
                )

