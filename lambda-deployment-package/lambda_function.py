import json
import boto3
import urllib.parse
import pandas as pd
from trp import Document


# boto3 handles for services of interest and other variables
sns = boto3.resource('sns')
s3 = boto3.client('s3')
dynamodb = boto3.client('dynamodb')
comprehend = boto3.client('comprehend')
ssm = boto3.client('ssm')
textract = boto3.client('textract')
overall_sentiment = ''
targeted_sentiment_response = ''
# Call Textract and get the text from our email image
def get_document(bucket, key):
    response = textract.detect_document_text(
    Document={
            'S3Object': {
                'Bucket': bucket,
                'Name': key
                }
            }
        )

    doc = Document(response)
    in_text = ''
    # Iterate over elements in the document
    for page in doc.pages:
        # Print lines and words
        for line in page.lines:
            in_text += line.text + '\n'
            # Let us fix a small typo where Textract detected I as a pipe character
    in_text = in_text.replace('|\n','I ')
    print("get document is complete: " + in_text)
    return in_text

def get_targeted_sentiment(in_text):
    # Generate targeted sentiment text for the input text
    targeted_sentiment = comprehend.detect_targeted_sentiment(Text=in_text, LanguageCode='en')
    # Load insights from targeted sentiment onto a dataframe
    insight_df = pd.DataFrame()
    j = 0
    for entity in targeted_sentiment['Entities']:
        for mention in entity['Mentions']:
            #print(mention)
            j+=1
            begin = mention['BeginOffset']
            end = mention['EndOffset']
            text = mention['Text']
            typ = mention['Type']
            sentiment = mention['MentionSentiment']['Sentiment']
            insight = in_text[begin:end]+'['+str(typ).lower()+'-'+str(sentiment).lower()+']'
            insight_df.at[j,'begin'] = begin
            insight_df.at[j,'end'] = end
            insight_df.at[j,'text'] = text
            insight_df.at[j,'type'] = typ
            insight_df.at[j,'sentiment'] = sentiment
            insight_df.at[j,'insight'] = insight
    insight_df = insight_df.sort_values(['begin'],ascending=True)
    # Now take the insights and overlay on the original text from the email
    off_len = 0
    k = 0
    temp = ''
    off_len = 0
    for idx, row in insight_df.iterrows():
        k += 1
        if k <= 1:
            temp=''.join((in_text[:int(row['begin'])],row['insight'],in_text[int(row['end']):]))
        else:
            temp=''.join((temp[:int(row['begin'])+off_len],row['insight'],temp[int(row['end'])+off_len:]))
        off_len += len(row['insight'])-len(row['text'])
    return temp

def put_ddb_entry(key,in_text, overall_sentiment, targeted_sentiment_response):
    company_addr = ''
    incoming_addr = ''
    owed_by = ''
    incoming_contact = ''
    company_contact = ''
    incoming_org = ''
    company_org = ''

    # Get the entities to store in our DynamoDB table
    entities = comprehend.detect_entities(Text=in_text, LanguageCode='en')

    for entity in entities['Entities']:
        if entity['BeginOffset'] < 75:
            if entity['Type'] == 'PERSON':
                incoming_contact = entity['Text']
            if entity['Type'] == 'ORGANIZATION':
                incoming_org = entity['Text']
            if entity['Type'] == 'LOCATION':
                incoming_addr += entity['Text'] + ' '
        else:
            if entity['Type'] == 'PERSON':
                company_contact = entity['Text']
            if entity['Type'] == 'ORGANIZATION':
                company_org = entity['Text']
            if entity['Type'] == 'LOCATION':
                company_addr += entity['Text'] + ' '

        if entity['Type'] == 'QUANTITY':
            amount_owed = entity['Text']
        if entity['Type'] == 'DATE':
            owed_by = entity['Text']

    # Determine status for the communication
    # if sentiment is not positive, it will be alerted, or it will be new
    status = 'NEW'
    if overall_sentiment != 'POSITIVE':
        status = 'ALERTED'
    # Insert into DynamoDB table including sentiment and targeted sentiment
    # key is the email name or number
    # Get the table name from SSM parameter
    param_response = ssm.get_parameter(Name='ddb_table_name', WithDecryption=True)
    # S3 prefix will be part of the Lambda input which is the ID to the DynamoDB table
    s3_prefix = 'email1' # dummy assignment in notebook but Lambda will have correct value

    dynamodb.put_item(
            TableName=str(param_response['Parameter']['Value']),
            Item={
            'id': {'S': str(key)},
            'IncomingContact': {'S': str(incoming_contact)},
            'IncomingOrg': {'S': str(incoming_org)},
            'IncomingAddress': {'S': str(incoming_addr)},
            'CompanyContact': {'S': str(company_contact)},
            'CompanyOrg': {'S': str(company_org)},
            'CompanyAddress': {'S': str(company_addr)},
            'AmountOwed': {'S': str(amount_owed)},
            'OwedBy': {'S': str(owed_by)},
            'EmailSentiment': {'S': str(overall_sentiment)},
            'EmailTargetedSentiment': {'S': str(targeted_sentiment_response)},
            'Status': {'S': status}
            }
        )
    print("Email comms added to DynamoDB")

def send_email(overall_sentiment, targeted_sentiment_response):
    # Now we will send SNS email with targeted sentiment embedded
    # first get topic name from SSM (this was created by our CloudFormation template)
    if overall_sentiment != 'POSITIVE':
        temp_topic = ssm.get_parameter(Name='sns_topic_name', WithDecryption=True)
        topic_name = temp_topic['Parameter']['Value']
        # workaround to get the topic ARN for publish, this will not create a new topic but just return the ARN
        topic = sns.create_topic(Name=topic_name)
        subject = "A " + overall_sentiment + " inbound communication was received - please take action"
        message = "Amazon Comprehend analyzed the inbound communication and detected contextual sentiment as follows. \n" + targeted_sentiment_response
        response = topic.publish(Subject=subject, Message=message)
        print("Email alert sent to adminstrator")

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    response = None
    # Let us first get the S3 prefix
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    # call textract and get the text from our mail image
    in_text = get_document(bucket, key)
    # call Comprehend to first get overall sentiment of our mail
    sentiment = comprehend.detect_sentiment(Text=in_text, LanguageCode='en')
    overall_sentiment = sentiment['Sentiment']
    # Now get targeted sentiment from Comprehend
    targeted_sentiment_response = get_targeted_sentiment(in_text)
    # Now detect key entities using Comprehend for sending our mail into a DynamoDB table for downstream consumption
    put_ddb_entry(key, in_text, overall_sentiment, targeted_sentiment_response)
    # Finally let us send an email to an adminstrator if the overall_sentiment is not POSITIVE
    send_email(overall_sentiment, targeted_sentiment_response)

    return response
