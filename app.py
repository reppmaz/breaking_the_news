import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from ast import literal_eval
import ast
import plotly.graph_objects as go
import plotly.express as px

#--------------
# GENERAL SETUP
#--------------
st.set_page_config(layout="wide")

# load data
@st.cache_data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = df['datetime'].astype(str)
    df['entities'] = df['entities'].apply(literal_eval)
    df = df[df['datetime'].str[:6].apply(lambda x: x.isnumeric() and int(x) >= 202311 or x == 'nan')]
    return df

data_filepath = "hackshow_dataframe_news_labeled.csv"
df = load_data(data_filepath)
df = df[df['topic'] != 'Börse'] # drop boerse

# color mappings
pol_leaning_colors = {
    'links': '#fe292b',
    'mitte_links': '#fed26c',
    'mitte': '#feaaac',
    'mitte_rechts': '#2bb19d',
    'rechts': '#036acc'}

sentiment_colors = {
    'positive': '#68b25d',
    'neutral': '#676269',
    'negative': '#d94eda'}

# create tabs for analysis and about sections
tab1, tab2 = st.tabs(["Politische Analyse", "Über das Projekt"])

# Political Analysis Tab
with tab1:
    #--------------
    # PROJECT TITLE
    #--------------
    st.title("WHAT'S NEW(S)")
    st.markdown("<hr style='border:2px solid #FFF'>", unsafe_allow_html=True)

    st.markdown(
    "<p style='font-size:20px; color:#5b5b5c;'><strong>Für die beste Darstellung schalte bitte in den Wide-Modus:</strong><br> "
    "Klicken Sie auf die drei Punkte oben rechts (⋮) → <em>Settings</em> → <em>Wide mode</em>.</p>",
    unsafe_allow_html=True)

    # ----------------------------------
    # TOPIC SELECTION AND RELATED TOPICS
    # ----------------------------------
    # get 300 most recent articles and calculate the most frequent topic
    df_recent = df.dropna(subset=['datetime']).sort_values(by='datetime', ascending=False).head(300)
    top_topic = df_recent['topic'].value_counts().idxmax()

    # get the top 10 topics overall
    top_10_topics = df['topic'].value_counts().nlargest(10).index.tolist()

    # handle cases where top_topic is not in top_10_topics
    if top_topic not in top_10_topics:
        top_10_topics.insert(0, top_topic)

    col1, col2 = st.columns(2)
    style = """
    <style>
        .stSelectbox > div {font-size: 18px;}
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

    # initialize session state for selected topic
    if "selected_topic" not in st.session_state:
        st.session_state.selected_topic = top_topic

    # dropdown for main topic
    with col1:
        st.subheader(f"Wähle ein Thema für die Analyse:")

        selected_main_topic = st.selectbox(
            "Hauptthema",
            options=sorted(df['topic'].unique()),  # Sorted list of all topics
            index=sorted(df['topic'].unique()).index(st.session_state.selected_topic)
            if st.session_state.selected_topic in sorted(df['topic'].unique()) else 0)
        # update session state if new main topic is selected
        if st.session_state.selected_topic != selected_main_topic:
            st.session_state.selected_topic = selected_main_topic

    # dropdown for related topics
    with col2:
        st.subheader(f"Verwandte Themen:")

        # find related topics based on shared entities
        if 'related_topics' in df.columns:
            related_topics = df.loc[df['topic'] == st.session_state.selected_topic, 'related_topics'].dropna().values
            if len(related_topics) > 0:
                related_topics = [
                    topic[0] if isinstance(topic, tuple) else topic  # extract topic name
                    for sublist in related_topics
                    for topic in eval(sublist)
                    if topic != st.session_state.selected_topic]
                #get unique related topics
                related_topics = list(pd.Series(related_topics).value_counts().head(5).index)
            else:
                related_topics = []
        else:
            related_topics = []

        if len(related_topics) > 0:
            selected_related_topic = st.selectbox(
                "Verwandtes Thema",
                options=related_topics,
                index=0,)
            # ppdate session state if new related topic is selected
            if st.session_state.selected_topic != selected_related_topic:
                st.session_state.selected_topic = selected_related_topic
        else:
            st.write("Keine verwandten Themen verfügbar.")

    # Define selected topic
    selected_topic = st.session_state.selected_topic

    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True) # boarder

    # ----------------------------------------
    # SELECTED ARTICLES DISPLAY
    # ----------------------------------------
    col1, col2 = st.columns(2)

    # select articles for selected topic
    selected_articles = []
    for leaning in pol_leaning_colors.keys():  # iterate over each political leaning
        # filter articles for selected topic and political leaning
        articles = df[(df['topic'] == selected_topic) & (df['pol_leaning'] == leaning)]
        if not articles.empty:
            # sort articles by datetime in descending order and select most recent
            most_recent_article = articles.sort_values(by='datetime', ascending=False).iloc[0]
            selected_articles.append(most_recent_article)

    with col1:
        st.subheader(f"Das wird über {selected_topic} gesagt:")

        # show selected articles
        formatted_articles = ""
        for article in selected_articles:
            formatted_articles += f"""
                <p style='font-size:20px'>- <a href='{article["url"]}' target='_blank' style='color:white;'>{article["source"]}</a> (eher {article["pol_leaning"]})</p>
            """

        # sisplay formatted articles
        st.markdown(formatted_articles, unsafe_allow_html=True)

        # percentage of articles by political leaning
        pol_leaning_counts = df[df['topic'] == selected_topic]['pol_leaning'].value_counts()
        pol_leaning_counts = pol_leaning_counts.reindex(pol_leaning_colors.keys(), fill_value=0)  # Ensure all leanings are included
        total_articles = pol_leaning_counts.sum()
        pol_leaning_percentages = (pol_leaning_counts / total_articles * 100).round(2)

        # blind spots (pol_leaning <= 10%)
        blind_spots = pol_leaning_percentages[pol_leaning_percentages <= 10]
        if not blind_spots.empty:
            st.markdown("<br>", unsafe_allow_html=True)  # Empty line
            st.markdown("### Blinder Fleck")
            st.markdown("""<p style="font-size:20px;">Die folgenden politischen Seiten sprechen wenig über dieses Thema:</p>""", unsafe_allow_html=True)

            formatted_blind_spots = ""
            for leaning, pct in blind_spots.items():
                if pct == 0.0:
                    formatted_blind_spots += f"""
                        <p style="font-size:20px; line-height:1.5;">- <b>{leaning}</b> mit gar keiner Berichterstattung</p>
                    """
                else:
                    formatted_blind_spots += f"""
                        <p style="font-size:20px; line-height:1.5;">- <b>{leaning}</b> mit nur <b>{pct}%</b> der Berichterstattung</p>
                    """
            st.markdown(formatted_blind_spots, unsafe_allow_html=True)

    with col2:
        st.subheader(f"Wer berichtet wie viel über {selected_topic}:")

        # filter out 0 values
        pol_leaning_counts = df[df['topic'] == selected_topic]['pol_leaning'].value_counts()
        pol_leaning_counts = pol_leaning_counts[pol_leaning_counts > 0]  # Exclude 0 values
        labels, values = pol_leaning_counts.index, pol_leaning_counts.values
        colors = [pol_leaning_colors[leaning] for leaning in labels]

        # donut chart
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                marker=dict(colors=colors),
                textinfo='percent',
                textfont=dict(size=19))])

        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1.2,
                font=dict(size=19)),
            margin=dict(t=40, b=40, l=40, r=40),
            paper_bgcolor="#0e1214",
            font=dict(color="#f8f8fa"))

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # -------------------
    # REPORTING OVER TIME
    # -------------------
    st.subheader(f"So oft wurde über {selected_topic} berichtet:")

    # filter out rows with NaN, invalid 'datetime' values, or years before 2023
    df_filtered = df.dropna(subset=['datetime'])
    df_filtered = df_filtered[df_filtered['datetime'].str.match(r'^\d{8,}')]
    df_filtered['Datum'] = df_filtered['datetime'].str[:6]
    df_filtered['Datum'] = pd.to_datetime(df_filtered['Datum'], format='%Y%m')

    if df_filtered.empty:
        st.write("Keine Daten verfügbar.")
    else:
        #determine min and max dates for the slider
        min_date = df_filtered['Datum'].min()
        max_date = df_filtered['Datum'].max()

        # add a date range slider using formatted strings
        date_range = st.slider(
            "Wähle den Zeitraum für die Analyse:",
            min_value=min_date.to_pydatetime(),  # convert to Python datetime
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
            format="MM/YYYY")

        # filter data based on selected date range
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_filtered = df_filtered[(df_filtered['Datum'] >= start_date) & (df_filtered['Datum'] <= end_date)]

        # normalize by total articles per source
        total_articles_per_source = df_filtered[df_filtered['topic'] == selected_topic].groupby('source').size()

        monthly_topic_data = (
            df_filtered[df_filtered['topic'] == selected_topic]
            .groupby(['Datum', 'source'])
            .size()
            .div(total_articles_per_source)
            .unstack(fill_value=0))

        if monthly_topic_data.empty:
            st.write("Keine Daten Verfügbar.")
        else:
            # prep data for plotting
            monthly_topic_data = monthly_topic_data.reset_index().melt(
                id_vars="Datum", var_name="Source", value_name="Relative Häufigkeit")

            # create line chart
            fig = px.line(monthly_topic_data, x="Datum", y="Relative Häufigkeit", color="Source")

            # update axes and layout
            fig.update_xaxes(
                title=dict(text="Datum", font=dict(size=20)),
                tickfont=dict(size=20),
                tickformat="%b %Y",
                dtick="M1")
            fig.update_yaxes(
                title=dict(text="Relative Häufigkeit", font=dict(size=20)),
                tickfont=dict(size=20))
            fig.update_layout(
                legend=dict(title=dict(text="Source", font=dict(size=20)), font=dict(size=20)),
                paper_bgcolor="#0e1214",
                plot_bgcolor="#0e1214",
                font=dict(color="#f8f8fa"))

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True) # boarder

    # ---------
    # SENTIMENT
    # ---------
    st.subheader(f"So ist die Stimmung zu {selected_topic}:")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<p style='font-size:20px;'>Gesamtstimmung über alle Medien:</p>", unsafe_allow_html=True)
        # sentiment counts
        total_sentiment_counts = df[df['topic'] == selected_topic]['sentiment'].value_counts()
        
        for sentiment in ['negative', 'neutral', 'positive']:
            if sentiment not in total_sentiment_counts:
                total_sentiment_counts[sentiment] = 0
        total_sentiment_counts = total_sentiment_counts[['negative', 'neutral', 'positive']]

        # bar chart for total sentiment
        fig = go.Figure(data=[
            go.Bar(
                x=total_sentiment_counts.index,
                y=total_sentiment_counts.values,
                marker_color=[sentiment_colors[s] for s in total_sentiment_counts.index])])
        fig.update_layout(
            xaxis=dict(
                title="",
                titlefont=dict(size=20),
                tickfont=dict(size=20)),
                yaxis=dict(
                title="Anzahl der Artikel",
                titlefont=dict(size=20)),
            paper_bgcolor="#0e1214",
            plot_bgcolor="#0e1214",
            font=dict(color="#f8f8fa"),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown(f"<p style='font-size:20px;'>Stimmung nach Medium:</p>", unsafe_allow_html=True)
        
        # sentiment counts by source
        sentiment_counts = df[df['topic'] == selected_topic].groupby('source')['sentiment'].value_counts().unstack(fill_value=0)
        
        # Ensure all columns (sentiment categories) are included
        for sentiment in ['negative', 'neutral', 'positive']:
            if sentiment not in sentiment_counts.columns:
                sentiment_counts[sentiment] = 0
        sentiment_counts = sentiment_counts[['negative', 'neutral', 'positive']]
        
       # stacked bar chart for sentiment by source
        fig2 = go.Figure()
        for sentiment in sentiment_counts.columns:
            fig2.add_trace(
                go.Bar(
                    name=sentiment,
                    x=sentiment_counts.index,
                    y=sentiment_counts[sentiment],
                    marker_color=sentiment_colors[sentiment]))

        fig2.update_layout(xaxis=dict(title="",
                                      tickfont=dict(size=20)),
                                      yaxis=dict(title="Anzahl der Artikel",
                                                 titlefont=dict(size=20),
                                                 tickfont=dict(size=20)),
                                                 barmode="stack",
                                                 paper_bgcolor="#0e1214",
                                                 plot_bgcolor="#0e1214",
                                                 font=dict(color="#f8f8fa"),
                                                 legend=dict(title="",
                                                             font=dict(size=20)))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # -------------------------
    # SENTIMENT ANALYSIS OVER TIME
    # -------------------------
    st.subheader(f"So verändert sich die Stimmung zu {selected_topic} über die Zeit?")

    # filter out rows with NaN, invalid 'datetime' values, or years before 2023
    df_filtered = df.dropna(subset=['datetime'])
    df_filtered = df_filtered[df_filtered['datetime'].str.match(r'^\d{8,}')]
    df_filtered['Datum'] = df_filtered['datetime'].str[:6]
    df_filtered['Datum'] = pd.to_datetime(df_filtered['Datum'], format='%Y%m')

    if df_filtered.empty:
        st.write("Keine Daten verfügbar.")
    else:
        # determine min and max dates for the slider
        min_date = df_filtered['Datum'].min()
        max_date = df_filtered['Datum'].max()

        # ddd a date range slider with a unique key
        date_range = st.slider(
            "Wähle den Zeitraum für die Analyse:",
            min_value=min_date.to_pydatetime(),  # Convert to Python datetime
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
            format="MM/YYYY",
            key="sentiment_date_slider")

        # dilter data based on selected date range
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_filtered = df_filtered[(df_filtered['Datum'] >= start_date) & (df_filtered['Datum'] <= end_date)]

        # group by 'Datum' and 'sentiment' for trend analysis
        sentiment_over_time = (
            df_filtered[df_filtered['topic'] == selected_topic]
            .groupby(['Datum', 'sentiment'])
            .size()
            .unstack(fill_value=0))

        if sentiment_over_time.empty:
            st.write("Keine Daten Verfügbar.")
        else:
            # calculate relative numbers (percentages)
            sentiment_over_time_relative = sentiment_over_time.div(sentiment_over_time.sum(axis=1), axis=0) * 100

            # prepare data for plotting
            sentiment_over_time_relative = sentiment_over_time_relative.reset_index().melt(
                id_vars="Datum", var_name="Sentiment", value_name="Prozentsatz")

            #line chart
            fig = px.line(
                sentiment_over_time_relative,
                x="Datum",
                y="Prozentsatz",
                color="Sentiment",
                color_discrete_map=sentiment_colors,  # Use sentiment-specific colors
                title=f"")

            fig.update_xaxes(
                title=dict(text="Datum", font=dict(size=20)),
                tickfont=dict(size=20),
                tickformat="%b %Y",
                dtick="M1")
            fig.update_yaxes(
                title=dict(text="Prozentsatz der Artikel", font=dict(size=20)),
                tickfont=dict(size=20))
            fig.update_layout(
                legend=dict(title=dict(text="Sentiment", font=dict(size=20)), font=dict(size=20)),
                paper_bgcolor="#0e1214",
                plot_bgcolor="#0e1214",
                font=dict(color="#f8f8fa"))
            
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True) # boarder

    # -----------------------------------------------
    # SENTIMENT SCORES OVER TIME BY POLITICAL LEANING
    # -----------------------------------------------
    st.subheader(f"Stimmungs-Scores im Zeitverlauf, gruppiert nach politischer Ausrichtung für {selected_topic}")

    # Filter out rows with NaN, invalid 'datetime' values, or years before 2023
    df_filtered = df.dropna(subset=['datetime'])
    df_filtered = df_filtered[df_filtered['datetime'].str.match(r'^\d{8,}')]
    df_filtered['Datum'] = df_filtered['datetime'].str[:6]
    df_filtered['Datum'] = pd.to_datetime(df_filtered['Datum'], format='%Y%m')

    # Filter for 'links', 'mitte', and 'rechts'
    selected_political_leanings = ['links', 'mitte', 'rechts']
    pol_leaning_colors = {
        'links': '#fe292b',
        'mitte': '#feaaac',
        'rechts': '#036acc'}

    if df_filtered.empty:
        st.write("Keine Daten verfügbar.")
    else:
        min_date = df_filtered['Datum'].min()
        max_date = df_filtered['Datum'].max()

        date_range = st.slider(
            "Wähle den Zeitraum für die Analyse:",
            min_value=min_date.to_pydatetime(),  # Convert to Python datetime
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
            format="MM/YYYY",
            key="scatter_date_slider")

        # Filter data based on the selected date range
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_filtered = df_filtered[(df_filtered['Datum'] >= start_date) & (df_filtered['Datum'] <= end_date)]

        # Map sentiment categories to numerical scores
        sentiment_mapping = {
            "positive": 1,
            "neutral": 0,
            "negative": -1}
        df_filtered['sentiment_score'] = df_filtered['sentiment'].map(sentiment_mapping)

        # Filter for specific political leanings
        df_filtered = df_filtered[df_filtered['pol_leaning'].isin(selected_political_leanings)]

        # Group by Datum and pol_leaning
        sentiment_trends = (
            df_filtered[df_filtered['topic'] == selected_topic]
            .groupby(['Datum', 'pol_leaning'])
            .agg({'sentiment_score': 'mean'})  # Average sentiment score
            .reset_index())

        if sentiment_trends.empty:
            st.write("Keine Daten Verfügbar.")
        else:
            st.markdown(
                "<p style='font-size:18px;'>"
                "-1 = negativ, 0 = neutral, 1 = positiv</p>",
                unsafe_allow_html=True)

            # Prepare a plotly scatter plot
            fig = go.Figure()

            # Add scatter points and regression line for each political leaning
            for pol_leaning in selected_political_leanings:
                filtered = sentiment_trends[sentiment_trends['pol_leaning'] == pol_leaning]
                
                if not filtered.empty:
                    # Add scatter points
                    fig.add_trace(go.Scatter(
                        x=filtered['Datum'],
                        y=filtered['sentiment_score'],
                        mode='markers',
                        name=pol_leaning,
                        marker=dict(color=pol_leaning_colors[pol_leaning])
                    ))

                    # Calculate regression line
                    x_numeric = (filtered['Datum'] - filtered['Datum'].min()).dt.days
                    coefficients = np.polyfit(x_numeric, filtered['sentiment_score'], 1)  # Linear fit
                    regression_line = coefficients[0] * x_numeric + coefficients[1]

                    # Add regression line
                    fig.add_trace(go.Scatter(
                        x=filtered['Datum'],
                        y=regression_line,
                        mode='lines',
                        name=f"{pol_leaning} (Trend)",
                        line=dict(color=pol_leaning_colors[pol_leaning], dash='dash')
                    ))

            # Update layout
            fig.update_layout(
                title="",
                xaxis=dict(
                    title="Datum",
                    titlefont=dict(size=20),
                    tickfont=dict(size=20),
                    tickformat="%b %Y"),
                yaxis=dict(
                    title="Durchschnittlicher Stimmungs-Score",
                    titlefont=dict(size=20),
                    tickfont=dict(size=20),
                    range=[-1.2, 1.2]),
                legend=dict(title=dict(text="Politische Ausrichtung", font=dict(size=20)), font=dict(size=20)),
                paper_bgcolor="#0e1214",
                plot_bgcolor="#0e1214",
                font=dict(color="#f8f8fa")
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)


with tab2:
    # --------------
    # ABOUT SECTION
    # --------------
    # add main section header
    st.title("WHAT'S NEW(S)")
    st.markdown("<hr style='border:2px solid #FFF'>", unsafe_allow_html=True)

    # Brief Introduction
    st.markdown("""
        <p style='font-size:30px'>Diese Informationen dienen der Transparenz und Nachvollziehbarkeit unserer Analyseprozesse.</p>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # Section: Datenquellen
    st.header("Datenquellen")
    st.markdown("<hr style='border:2px solid #5b5b5c'>", unsafe_allow_html=True)  # Section divider
    st.markdown("""
        <p style='font-size:20px'>Diese Anwendung verwendet deutsche Nachrichtenartikel der letzten 12 Monate:</p>
        <p style='font-size:20px'>- <a href='https://www.gdeltproject.org' target='_blank' style='color:white;'>Gdelt Project</a></p>
        <p style='font-size:20px'>- News APIs: NewsCatcher, NewsAPI, MediaStack</p>

    """, unsafe_allow_html=True)

    # Divider
    st.markdown("<hr style='border:1px solid #333'>", unsafe_allow_html=True)

    # Section: Methodik
    st.header("Methodik")
    st.markdown("<hr style='border:2px solid #5b5b5c'>", unsafe_allow_html=True)  # Section divider

    # Subsection: Themen Klassifizierung
    st.subheader("Themen Klassifizierung")
    st.markdown("""
        <p style='font-size:20px'>Modell: <a href='https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2' target='_blank' style='color:white;'>paraphrase-MiniLM-L6-v2</a></p>
    """, unsafe_allow_html=True)

    # Divider for subsections
    st.markdown("<hr style='border:1px solid #5b5b5c'>", unsafe_allow_html=True)

    # Subsection: Politische Einordnung
    st.subheader("Politische Einordnung")
    st.markdown("""
        <p style='font-size:20px'>Medien wurden basierend auf 4 Quellen auf dem politischen Spektrum verordnet:</p>
        <p style='font-size:20px'>- <a href='https://interaktiv.tagesspiegel.de/lab/die-lieblingsmedien-der-parteien/' target='_blank' style='color:white;'>Tagesspiegel</a></p>
        <p style='font-size:20px'>- <a href='https://uebermedien.de/93000/wieso-haben-zeitungen-eine-politische-ausrichtung/' target='_blank' style='color:white;'>Übermedien</a></p>
        <p style='font-size:20px'>- <a href='https://www.pewresearch.org/global/fact-sheet/datenblatt-nachrichtenmedien-und-politische-haltungen-in-deutschland/' target='_blank' style='color:white;'>Pew Research</a></p>
        <p style='font-size:20px'>- <a href='https://www.polkom.ifp.uni-mainz.de/files/2024/01/pm_perspektivenvielfalt.pdf' target='_blank' style='color:white;'>Mainz Studie</a></p>
    """, unsafe_allow_html=True)

    # table of sources and their political leanings
    st.markdown("<h3 style='font-size:20px; color:#f8f8fa'>Liste der Medien und ihre politische Ausrichtung</h3>", unsafe_allow_html=True)

    sources_and_leanings = (
        df[['source', 'pol_leaning']]
        .drop_duplicates()
        .sort_values(by='pol_leaning'))

    st.markdown(
        sources_and_leanings.rename(columns={
            'source': 'Medienquelle',
            'pol_leaning': 'Politische Ausrichtung'
        }).to_html(index=False, escape=False),
        unsafe_allow_html=True)

    # Divider for subsections
    st.markdown("<hr style='border:1px solid #5b5b5c'>", unsafe_allow_html=True)

    # Subsection: Blinderfleck-Analyse
    st.subheader("Blinderfleck-Analyse")
    st.markdown("""
        <p style='font-size:20px'>Definiert als politische Ausrichtungen mit ≤10% Anteil an der Berichterstattung zum ausgewählten Thema.</p>
    """, unsafe_allow_html=True)

    # Divider for subsections
    st.markdown("<hr style='border:1px solid #5b5b5c'>", unsafe_allow_html=True)

    # Subsection: Stimmungs-Analyse
    st.subheader("Stimmungs-Analyse")
    st.markdown("""
        <p style='font-size:20px'>- <a href='https://huggingface.co/ssary/XLM-RoBERTa-German-sentiment' target='_blank' style='color:white;'>XLM-RoBERTa-German-sentiment</a></p>
    """, unsafe_allow_html=True)

    # Divider for subsections
    st.markdown("<hr style='border:1px solid #5b5b5c'>", unsafe_allow_html=True)

    # Subsection: Verwandte Themen
    st.subheader("Verwandte Themen")
    st.markdown("""
        <p style='font-size:20px'>Zeigt die 5 Themen mit den meisten (mind. 2) gemeinsamen Entitäten.</p>
    """, unsafe_allow_html=True)

    # Final Divider
    st.markdown("<hr style='border:2px solid #333'>", unsafe_allow_html=True)

    # Section: Datenbank
    st.header("Datenbank")
    st.markdown("<hr style='border:2px solid #5b5b5c'>", unsafe_allow_html=True)  # Section divider
    st.markdown(f"""
        <p style='font-size:20px; color:#f8f8fa'>
        Unsere Datenbank umfasst <b>{df.shape[0]:,} Artikel</b> aus <b>{df['source'].nunique()} Medien</b>, die <b>{df['topic'].nunique()} Themen</b> abdecken.
        </p>
    """, unsafe_allow_html=True)

    # basic visualizations
    st.markdown("""
        <p style='font-size:20px; color:#f8f8fa'>Hier sind einige Eckdaten visualisiert:</p>
    """, unsafe_allow_html=True)

    def create_colored_bar_chart(x, y, title, xaxis_title, yaxis_title):
        fig = go.Figure(data=[go.Bar(
            x=x,
            y=y,
            marker=dict(color=y, colorscale='dense'))])
        fig.update_layout(
            title=dict(text=title, font=dict(size=20)),
            xaxis=dict(title=xaxis_title, titlefont=dict(size=20), tickfont=dict(size=20)),
            yaxis=dict(title=yaxis_title, titlefont=dict(size=20), tickfont=dict(size=20)),
            paper_bgcolor="#0e1214",
            plot_bgcolor="#0e1214",
            font=dict(color="#f8f8fa"))
        return fig

    # Plot 1: Number of articles per political leaning
    st.markdown("<h3 style='font-size:20px; color:#5b5b5c'>Artikel nach politischer Ausrichtung</h3>", unsafe_allow_html=True)
    pol_leaning_counts = df['pol_leaning'].value_counts()
    fig1 = create_colored_bar_chart(
        x=pol_leaning_counts.index,
        y=pol_leaning_counts.values,
        title="",
        xaxis_title="",
        yaxis_title="Anzahl der Artikel")
    st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Number of articles per news outlet
    st.markdown("<h3 style='font-size:20px; color:#5b5b5c'>Artikel nach Nachrichtenquelle</h3>", unsafe_allow_html=True)
    source_counts = df['source'].value_counts()
    fig2 = create_colored_bar_chart(
        x=source_counts.index,
        y=source_counts.values,
        title="",
        xaxis_title="",
        yaxis_title="Anzahl der Artikel")
    st.plotly_chart(fig2, use_container_width=True)

    # Plot 3: Number of articles per topic
    st.markdown("<h3 style='font-size:20px; color:#5b5b5c'>Artikel nach Thema</h3>", unsafe_allow_html=True)
    topic_counts = df['topic'].value_counts()
    fig3 = create_colored_bar_chart(
        x=topic_counts.index,
        y=topic_counts.values,
        title="",
        xaxis_title="",
        yaxis_title="Anzahl der Artikel")
    st.plotly_chart(fig3, use_container_width=True)
