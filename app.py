import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import preprocessor, helper
from models import User_prediction, summarizer, sentiment_analysis, personality_model, emotion_model
from summarizer import clean_messages, detect_topics, get_topic_keywords
from User_prediction import train_user_identification_model
from sentiment_analysis import analyze_sentiment
from emotion_model import detect_emotion
import seaborn as sns

st.sidebar.title('Whatsapp Chat Analyzer')
st.set_page_config(layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = None

if 'ml_feature' not in st.session_state:
    st.session_state.ml_feature = None

if 'personality' not in st.session_state:
    st.session_state.personality = None

if 'summary' not in st.session_state:
    st.session_state.summary = None

st.markdown(
    """
    <h1 style = "text-align: center; margin-bottom: 0;">
       WhatsApp Chat Analyzer
    </h1>
    <hr style = "width: 100%; magin: auto;">
    """,
    unsafe_allow_html = True
)

uploaded_file = st.sidebar.file_uploader('Choose a file')

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")

    df, chat_type = preprocessor.preprocess(data)

    # st.write(df)

    # for getting unique users
    Users = df['user'].unique().tolist()

    Users.sort()
    Users.insert(0,"All Users")

    if st.sidebar.button("Show Analysis"):
        st.session_state.page = "analysis"

    if st.sidebar.button("Fun ML Predictions"):
        st.session_state.page = "ml"

    if st.sidebar.button("User Personality Analysis"):
        st.session_state.page = 'personality'

    if st.sidebar.button("Chat Summary"):
        st.session_state.page = 'summary'


    if st.session_state.page == "analysis":
    
        st.title("Top Statistics")

        selected_user = st.selectbox(
            "Show Analysis with respect to",
            Users
        )
    
        st.divider()
    
        st.header(
            f"Analysis of {selected_user}"
        )


        messages_count, word_count, media_count = helper.fetch_stats(selected_user, df)

        col1, col2, col3= st.columns(3)

        with col1:
            st.header("Total Messages")
            st.title(messages_count)

        with col2:
            st.header("Total Words")
            st.title(word_count)

        with col3:
            st.header("Total Media Shared")
            st.title(media_count) 


        # monthly Timeline

        st.title('Monthly Timeline')
        timeline = helper.monthly_timeline(selected_user, df)

        fig = px.line(
            timeline,
            x = 'times',
            y = 'messages',
            color_discrete_sequence=['orange']
        )
        st.plotly_chart(fig)
        
        # daily tiemline

        st.title("Daily Timeline")

        daily_timeline = helper.daily_timeline(selected_user, df)

        fig = px.line(
            daily_timeline,
            x='date',
            y='messages',
            color_discrete_sequence=['green']
        )

        st.plotly_chart(fig, use_container_width=True)

        # activity map

        st.title('Activity Map')

        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color = 'purple')
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy Month")
            busy_month = helper.month_activity(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color = 'purple')
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)


        # Activity heat map

        st.title("Weekly activity map")

        activity_table = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(activity_table)
        st.pyplot(fig)
        
        
        # Finding the busy users in the group chats
        if selected_user == 'All Users':
            
            st.title('Most Busy')
            col1 , col2 = st.columns(2)
            x, y = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            
            with col1:
                ax.bar(x.index, x.values, color = 'orange')
                plt.xticks(rotation = 'vertical')
                st.pyplot(fig)

            with col2:
                st.dataframe(y)

        # WordCloud
        
        st.title("Word Cloud")
        wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        plt.imshow(wc)
        st.pyplot(fig)

        # most common words

        most_common_df = helper.most_common_words(selected_user, df)
        st.title("Most Common Word Analysis")

        col1 , col2 = st.columns(2)
        

        with col2:
            st.dataframe(most_common_df)

        with col1:

            fig, ay = plt.subplots()

            ay.barh(most_common_df[0], most_common_df[1], color = 'green')
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)

        # most_commom emojis

        st.title("Emoji Analysis")
        emojis_df = helper.emojis(selected_user, df)
        col1, col2 = st.columns(2)

        with col1:
            
            fig, ax = plt.subplots()
            plt.rcParams['font.family'] = 'Segoe UI Emoji'
            ax.pie(emojis_df[1].head(), labels = emojis_df[0].head(), autopct = "%0.2f")
            plt.xticks(rotation = 'vertical')
            st.pyplot(fig)
            

        with col2:
            
            st.dataframe(emojis_df)

    
    if st.session_state.page == "ml":


        st.title("Fun ML Predictions")
        
        message = st.text_input(
                        "Enter a message"
                )
    
        col1, col2, col3 = st.columns(3)

        with col1:
        
            if st.button(
                "User Identification",
                use_container_width=True
            ):
        
                st.session_state.ml_feature = "user_identification"
        
        
        with col2:
        
            if st.button(
                "Sentiment Analysis",
                use_container_width=True
            ):
        
                st.session_state.ml_feature = "sentiment"
        
        
        with col3:
        
            if st.button(
                "Emotion Detection",
                use_container_width=True
            ):
        
                st.session_state.ml_feature = "emotion"

    
        if st.session_state.ml_feature == "user_identification":

            st.subheader("User Identification")
    
            if chat_type == "private":
    
                with st.spinner("Training User Identification model..."):
    
                    model, tfidf, accuracy = train_user_identification_model(df)
    
                private_users = df['user'].unique().tolist()

                col1, col2 = st.columns(2)

                with col1:
                    
                    st.subheader("Detected Users")
        
                    for user in private_users:
                        st.write(user)
                    

                with col2:

                    st.subheader("Model Performance")
                    st.metric(
                        "Model Accuracy",
                        f"{accuracy * 100:.2f}%"
                    )

                st.subheader("User Prediction")
                    
                if message:
        
                    message_tfidf = tfidf.transform([message])
    
                    prediction = model.predict(message_tfidf)[0]
        
                    st.success(
                        f"Predicted User: {prediction}"
                    )
    
            else:
    
                st.warning(
                    "User Identification is currently available only for private 2-person chats."
                )


        
        elif st.session_state.ml_feature == "sentiment":

            st.subheader("Sentiment Analysis")

            if chat_type == 'private':
        
                if message:
            
                    result = analyze_sentiment(message)
            
                    sentiment = result[0]['label']
                    confidence = result[0]['score']
            
                    st.success(
                        f"Sentiment: {sentiment}"
                    )
    
                    st.subheader("Model Performance")
            
                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

            else:
    
                st.warning(
                    "Sentiment Analysis is currently available only for private 2-person chats."
                )

        elif st.session_state.ml_feature == "emotion":

            st.subheader("Emotion Detection")

            if chat_type == 'private':
            
                if message:
            
                    result = detect_emotion(message)
            
                    emotion = result[0]['label']
                    confidence = result[0]['score']
            
                    st.success(
                        f"Detected Emotion: {emotion}"
                    )
            
                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

            else:
    
                st.warning(
                    "Emotion Detecion is currently available only for private 2-person chats."
                )


    if st.session_state.page == "personality":

        st.title("Chat Personality Analysis")
    
        personality_df = (
            personality_model
            .generate_personality_analysis(df)
        )
    
        # User selection
        selected_person = st.selectbox(
            "Select a person",
            personality_df['user'].tolist()
        )
    
        # Selected user's data
        person = personality_df[
            personality_df['user'] == selected_person
        ].iloc[0]
    
        st.markdown("---")
    
        st.subheader(
            f"{selected_person}'s Chat Personality"
        )
    
        # Persona description
        persona_descriptions = {
    
            "Talkative":
            "This person is highly active and frequently participates in conversations.",
    
            "Expressive":
            "This person expresses themselves openly and uses emojis and expressive messages frequently.",
    
            "Questioner":
            "This person is curious and often engages in conversations by asking questions.",
    
            "Minimalist":
            "This person prefers short and concise messages."
        }
    
        persona = person['chat_persona']

        st.subheader(
            f"{persona}"
        )
        
        st.info(
            persona_descriptions[persona]
        )
        
        # Statistics
        st.subheader("Chat Behaviour")
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
    
            st.metric(
                "Messages",
                int(person['message_count'])
            )
    
        with col2:
    
            st.metric(
                "Avg Message Length",
                f"{person['avg_message_length']:.1f}"
            )
    
        with col3:
    
            st.metric(
                "Avg Words",
                f"{person['avg_word_count']:.1f}"
            )
    
        col4, col5, col6 = st.columns(3)
    
        with col4:
    
            st.metric(
                "Questions",
                int(person['question_count'])
            )
    
        with col5:
    
            st.metric(
                "Exclamations",
                int(person['exclamation_count'])
            )
    
        with col6:
    
            st.metric(
                "Emoji Usage",
                int(person['emoji_count'])
            )


        st.subheader("Communication Style")

        features = [
            "Message Activity",
            "Message Length",
            "Word Usage",
            "Questions",
            "Exclamations",
            "Emoji Usage"
        ]
        
        values = [
            person['message_count'],
            person['avg_message_length'],
            person['avg_word_count'],
            person['question_count'],
            person['exclamation_count'],
            person['emoji_count']
        ]

        scaler = MinMaxScaler()

        normalized_values = scaler.fit_transform(
            np.array(values).reshape(-1, 1)
        ).flatten()

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=normalized_values,
                theta=features,
                fill='toself',
                name=selected_person
            )
        )
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False
        )
        
        st.plotly_chart(
            fig,
            use_container_width=True
        )


    if st.session_state.page == "summary":

        st.title("Chat Summary")
    
        st.write(
            "Discover the main topics discussed in your WhatsApp chat."
        )
    
        if st.button("Generate Chat Summary"):
    
            with st.spinner("Analyzing chat topics..."):
    
                cleaned_messages = clean_messages(
                    df['messages']
                )
    
                clusters, vectorizer, kmeans = detect_topics(
                    cleaned_messages,
                    n_topics=5
                )
    
                topic_keywords = get_topic_keywords(
                    vectorizer,
                    kmeans
                )
    
            st.success("Chat analysis completed!")
    
            st.subheader("Main Topics Discussed")
    
            for topic in range(5):
    
                topic_messages = [
                    cleaned_messages[i]
                    for i in range(len(cleaned_messages))
                    if clusters[i] == topic
                ]
    
                keywords = topic_keywords[topic]
    
                st.markdown(
                    f" Topic {topic + 1}"
                )
    
                st.write(
                    "**Keywords:** "
                    + ", ".join(keywords)
                )
    
                st.write(
                    f" {len(topic_messages)} messages"
                )
    
                with st.expander("View conversation examples"):
    
                    for message in topic_messages:
                        st.write("•", message)
    
                st.divider()
