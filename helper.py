from wordcloud import WordCloud
import pandas as pd
import emoji
from collections import Counter

def fetch_stats(selected_user, df):

    if selected_user == 'All Users':
        # No.of messages
        num_messages = df.shape[0]
        
        # No.of words
        words = []
        for i in df['messages']:
            words.extend(i.split())

        # No.of media shared
        count = df[df['messages'] == '<Media omitted>\n'].shape[0]
        
        return num_messages, len(words), count

    else:
        new_df = df[df['user'] == selected_user] 
        num_messages = new_df.shape[0]
        
        words = []
        for i in new_df['messages']:
            words.extend(i.split())

        count = new_df[new_df['messages'] == '<Media omitted>\n'].shape[0]

        return num_messages, len(words), count


def most_busy_users(df):
    
    x = df['user'].value_counts().head()
    y = round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns = {'user' : 'name', 'count' : 'percent'})
    
    return x, y


def create_wordcloud(selected_user, df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
    
    df = df[df['messages'] != '<Media omitted>\n']
        
    wc = WordCloud(
        width=500,
        height=500,
        min_font_size=10,
        background_color='white'
)

    wordcloud = wc.generate(df['messages'].str.cat(sep=" "))

    return wordcloud


def most_common_words(selected_user, df):

    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]
    
    df = df[df['messages'] != '<Media omitted>\n']

    words = []
    common = ['this', 'message', 'was', 'deleted', '<this', 'edited>']

    for message in df['messages']:
        for word in message.lower().split():
            if word not in common:
                words.append(word)

    return_df = pd.DataFrame(Counter(words).most_common(25))

    return return_df


def emojis(selected_user, df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]

    emojis = []

    for i in df['messages']:
        emojis.extend([c for c in i if emoji.is_emoji(c)])
    
    emojis_df = pd.DataFrame(
        Counter(emojis).most_common(len(Counter(emojis)))
    )

    return emojis_df


def monthly_timeline(selected_user, df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month']).count()['messages'].reset_index()
    timeline['times'] = (timeline['month'] + '-' + timeline['year'].astype(str))

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('date') .count()['messages'].reset_index()

    return daily_timeline
    

def week_activity_map(selected_user, df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]

    return df['day'].value_counts()


def month_activity(selected_user, df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]


    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'All Users':
        df = df[df['user'] == selected_user]

    activity_table = df.pivot_table(index = 'day', columns = 'period', values = 'messages', aggfunc ='count').fillna(0)

    return activity_table

    