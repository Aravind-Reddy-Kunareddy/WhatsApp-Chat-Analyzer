import pandas as pd
import re
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

def create_personality_features(df):

    user_features = df.groupby('user').agg(

        message_count=('messages', 'count'),

        avg_message_length=('messages', lambda x: x.str.len().mean()),

        avg_word_count=(
            'messages',
            lambda x: x.str.split().str.len().mean()
        ),

        question_count=(
            'messages',
            lambda x: x.str.contains(r'\?', regex=True).sum()
        ),

        exclamation_count=(
            'messages',
            lambda x: x.str.contains(r'!', regex=True).sum()
        )

    ).reset_index()

    df = df.copy()

    df['emoji_count'] = df['messages'].apply(
        lambda x: len(
            re.findall(
                r'[\U0001F300-\U0001FAFF]',
                x
            )
        )
    )

    emoji_features = df.groupby('user')['emoji_count'].sum().reset_index()

    user_features = user_features.merge(
        emoji_features,
        on='user'
    )


    return user_features

def assign_chat_persona(row):

    scores = {
        "Talkative": 0,
        "Expressive": 0,
        "Questioner": 0,
        "Minimalist": 0
    }

    # Message behaviour
    if row['message_count'] > 500:
        scores["Talkative"] += 2

    if row['avg_message_length'] > 30:
        scores["Expressive"] += 2

    if row['avg_message_length'] < 15:
        scores["Minimalist"] += 2

    # Question behaviour
    if row['question_count'] > 50:
        scores["Questioner"] += 2

    # Emoji behaviour
    if row['emoji_count'] > 50:
        scores["Expressive"] += 2

    # Return highest scoring persona
    return max(
        scores,
        key=scores.get
    )

def generate_personality_analysis(df):

    user_features = create_personality_features(df)

    user_features['chat_persona'] = (
        user_features.apply(
            assign_chat_persona,
            axis=1
        )
    )

    return user_features


if __name__ == "__main__":

    import preprocessor

    with open("chat.txt", "r", encoding="utf-8") as file:
        data = file.read()

    df, chat_type = preprocessor.preprocess(data)

    personality_df = generate_personality_analysis(df)

    print(personality_df)