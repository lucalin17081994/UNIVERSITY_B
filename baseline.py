import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
class Baseline():
    
    def __init__(self):
        self.X, self.y= [],[]
        self.X_train, self.X_test, self.y_train, self.y_test= [],[],[],[]
        self.highest_label=""
        self.correct, self.incorrect=0,0
        
        keywords_m = {"request": ["where", "what", "whats", "type", "phone", "number",  #meaningful categories
                "address", "postcode", "post code"],
                "inform": ["restaurant", "food"],
                "confirm": ["is it", "does it", "do they", "is there", "is that"]
                 }
                    

    keywords_ts = {"ack": ["okay", "kay", "ok", "fine"],                       #turn-service categories
                "deny": ["wrong", "dont want", "not"],
                "reqmore": ["more"],
                "affirm": ["yes", "right", "correct", "yeah", "uh huh"],
                "negate": ["no"]
                  }
                
    keywords_ds = {"hello": ["hi", "hello"],                                #dialogue-service categories
                "goodbye": ["goodbye", "bye", "stop"],
                "thankyou": ["thank", "thanks"],
                "repeat": ["repeat", "back", "again"],
                "restart": ["start over", "reset", "restart"],
                "reqalts": ["how about", "what about", "anything else"],
                "null": ["cough", "unintelligible", "tv_noise", "noise", 
                         "sil", "sigh", "um"]
                  }
        self.wrong_predictions=[]
#%%
    def open_dataset(self, filename):
        """
        open dataset

        Parameters
        ----------
        filename : str
            To import dataset. Format should be [label utterance].


        """
        X = list()
        y = list()
        with open(filename, "r") as infile:
            for line in infile:
                label_and_utterance = line.lower().split(" ", 1)
                if label_and_utterance[0]!="null":
                    X.append(label_and_utterance[1])
                    y.append(label_and_utterance[0])
                
            self.X=X
            self.y=y
    
    
    #%%
    def split_dataset(self):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=0.15)
        
    #%%
    def convert_to_dict_freq(self): 
        counts_dictionary = {}
        for i in self.y:
          counts_dictionary[i] = counts_dictionary.get(i, 0) + 1
        return counts_dictionary
    
    #%%
    def get_highest_in_dict(self,d):
        count=0
        k=""
        for key,value in d.items():
            if value>count:
                count=value
                k=key
        return k,count
    #%%
    def get_highest_label(self):
        """
        make a dictionary out of y with frequencies and find out which label is most frequent

        """
        d=self.convert_to_dict_freq()
        k,count=self.get_highest_in_dict(d)
        self.highest_label=k
    
#%%
                
    def predict_highest_label_rule(self):
        #just return single instance as highest label
        return self.highest_label
    #%%
    def test_baseline_one(self):
        """
        classify all cases in testset as highest label

        """
        self.correct,self.incorrect=0,0 #reset in case tested already
        for y in self.y_test:
            if y==self.highest_label:
                self.correct+=1
            else:
                self.incorrect+=1
    #%%
    def predict_keyword_rule(self,x):
        """
        Given a phrase, predict the label based on key-word matching rules
    
        Parameters
        ----------
        x : str
            user input.
    
        Returns
        -------
        y_pred : str
            prediction of this classifier.
    
        """
        y_pred = "inform"
        
        for key_dict in [self.KEYWORDS_M, self.KEYWORDS_TS, self.KEYWORDS_DS]:
            for k,v in key_dict.items():
                for keyword in v:
                    if len(keyword.split(' ')) > 1:
                        if keyword in x:
                            y_pred = k
                            break
                    else:
                        if keyword in word_tokenize(x):
                            y_pred = k
                            break
        return y_pred
    
    
    #%%
    def test_baseline_two(self):
        """
        Test keyword-matching classification using whole dataset

        """
        self.correct,self.incorrect=0,0#reset in case tested already
        for n in range(len(self.X)):
            y_pred=self.predict_keyword_rule(self.X[n])
            if y_pred == self.y[n]:
                self.correct+=1
            else:
                self.incorrect+=1
                self.wrong_predictions.append((self.X[n],y_pred,self.y[n]))
        
          
            #%%
    def score(self):
        return self.correct/(self.correct+self.incorrect)
    #%%
    def get_wrong_predictions(self):
        return self.wrong_predictions
    #%%
    def user_input(self):
        while True:
            classifier=input("enter 1 for highest-label classification, enter 2 for rule-based classification and enter stop to quit:\n")
            if (classifier != "stop"):
                if (classifier=="1"):
                    while True:
                        
                        x=input("You have chosen for highest-label classification. Enter utterance for me to classify or enter stop to quit:\n")
                        if (x == "stop"):
                            break
                        y_pred=self.predict_highest_label_rule()
                        print("You have entered a {} utterance".format(y_pred))
                if (classifier=="2"):
                    while True:
                        
                        x=input("You have chosen for rule-based classification. Enter utterance for me to classify or enter stop to quit:\n")
                        if (x == "stop"):
                            break
                        y_pred=self.predict_keyword_rule(x)
                        print("You have entered a {} utterance".format(y_pred))
                        

            else:
                break

