# Subject_Classification_and_AI_Evaluation_Paper_Development_through_Voice_Data_Analysis

## Goal
Matching the Korean evaluation paper with the appropriate topic based on the audio conversation data between two people

## Detailed Goal
1. Language identification of audio data
- Recognize language of audio data
- Multilingual model
- Language recognition rate above 95%

2. Perform Speech-To-Text (STT) according to identified language
- Extract the conversation to text
- Multilingual model
- WER is 5.5% or less

3. Topic modeling based on extracted conversations
- Identify what topics they are talking about
- Extract more than 20 topics

4. Classifying conversations
- Matching Korean evaluation paper that matches the topic of audio data conversation
- Achieve 75% or more accuracy

## Data
1. Genius data(speech data between teachers and students)
2. AIHub “주제별 텍스트 일상 대화” data

## Process
1. Language identification of audio data
- Use VoxLingua107 ECAPA-TDNN model
- 100% language recognition rate achieved for Genius data (speech data between teachers and students)
  
2. Perform with Speech-to-Text according to identified language
- Use Whisper from OpenAI
- 5.2% WER

3. Topic modeling based on extracted conversations
- Use BERTopic
![topic](https://github.com/kimchaeri/Subject_Classification_and_AI_Evaluation_Paper_Development_through_Voice_Data_Analysis/assets/74261590/95d32170-27b8-464c-8859-39d0e16695f1)

4. Labeling according to topics and train the model





