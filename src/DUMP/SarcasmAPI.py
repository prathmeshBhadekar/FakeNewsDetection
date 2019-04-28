def Fun( line ):
    p_array = []
    n_array = []
    word_array = line.split(" ")
    p_count=0
    n_count=0
    for i in range(len(word_array)):
        blob = TextBlob(word_array[i])
        p = round(blob.sentiment.polarity, 5)
        if p>=0.5 : 
            p_count+=1
        else:
            n_count+=1
    p_array.append(p_count)
    n_array.append(n_count)
    return p_array,n_array
    
def Processing(text):
    clean_data = []    
    p_array=[]
    n_array=[]   
    s_array=[]
    po_array=[]  
    words = [w for w in text.split() if not w in stopwords.words('english')]
    clean_data.append("".join(words))
    p_array,n_array = Fun(text)
    blob=TextBlob(text)
    s_array.append(round(blob.sentiment.subjectivity, 5))
    po_array.append(round(blob.sentiment.polarity, 5)+1)
    return p_array,n_array,s_array,po_array

#API
import pickle

with open('tokenizerCNN-28-04.pkl', 'rb') as f:
    tokenizer_sarcasm = pickle.load(f)   
        
with open('sarcasmCNN-28-04.pkl', 'rb') as f:
    cnn_sarcasm = pickle.load(f)   

text = "Mirrors can't talk, lucky for you they can't laugh either."

def Sarcasm(text, tokenizer_sarcasm, cnn_sarcasm):
    tokenize_string = tokenizer_sarcasm.texts_to_sequences([text])
    string_sequence = pad_sequences(tokenize_string, maxlen=120)
    p_array,n_array,s_array,po_array = Processing(""+text)
    p=pd.DataFrame(np.column_stack((string_sequence, p_array)))
    p=pd.DataFrame(np.column_stack((p,n_array)))
    p=pd.DataFrame(np.column_stack((p,s_array)))
    p=pd.DataFrame(np.column_stack((p,po_array)))
    return model.predict(p)[0][0]*100

#call as
print(Sarcasm(text, tokenizer_sarcasm, cnn_sarcasm))    
