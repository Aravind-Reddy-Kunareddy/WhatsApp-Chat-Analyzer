import re
import pandas as pd

def categorize_reply_time(seconds):

    if seconds <= 60:
        return "Instant"

    elif seconds <= 1800:
        return "Normal"

    else:
        return "Late"

def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}\s*[AaPp][Mm]\s*-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({
        'message_date': dates,
        'User_message': messages
    })
    df['message_date'] = df['message_date'].str.rstrip(' -')
    df['message_date'] = pd.to_datetime(
        df['message_date'],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )
    users = []
    messages = []
    period = []

    for message in df['User_message']:
        entry = re.split(r'([^:]+):\s', message, maxsplit=1)
        if entry[1:]:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['messages'] = messages

    
    df.drop(columns = ['User_message'], inplace = True)
    df['year'] = df['message_date'].dt.year
    df['month'] = df['message_date'].dt.month_name()
    df['day'] = df['message_date'].dt.day_name()
    df['hour'] = df['message_date'].dt.hour
    df['min'] = df['message_date'].dt.minute
    df['date'] = df['message_date'].dt.date

    df = df[df['user'] != 'group_notification']

    for hour in df[['day', 'hour']]['hour']:
        if hour == 2:
            period.append(str(hour) + '-' + str('00'))
    
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
    
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    df['reply_time'] = (
        df['message_date']
        .shift(-1)
        - df['message_date']
    ).dt.total_seconds()

    df['reply_category'] = df['reply_time'].apply(
        categorize_reply_time
    )

    unique_users = df['user'].nunique()

    if unique_users == 2:
        chat_type = 'private'
    else:
        chat_type = 'group'

    
    return df, chat_type



