#!/usr/bin/env python
# coding: utf-8


#imported librairies:
import pandas as pd
from Levenshtein import distance as dt
from nltk.corpus import stopwords as s_w
from nltk.tokenize import word_tokenize as w_t
from classification import Classification
import re
import random
import time
#%%
class Dialogue_Agent():
    
    def __init__(self,dialog_acts_filename, restaurant_matrix_filename, machine_learning_model=""):
        """
        initialize the dialog agent

        Parameters
        ----------
        dialog_acts_filename : str
            filename of dialog acts. Dialog acts should be in format of [label utterance] with whitespace as separator.
        restaurant_matrix_filename : str
            csv file containing restaurants, their contacts and information.
        machine_learning_model: str
            nn for neural net, standard LR
        Returns
        -------
        None.

        """
        self.statelog=[]
        
        self.clf_agent= Classification()
        self.clf_agent.initialize_data(dialog_acts_filename)
        if machine_learning_model=="nn":
            self.clf_agent.train_nn()
        else:
            self.clf_agent.train_lr()
        #preparation for preference extraction
        file=pd.read_csv(restaurant_matrix_filename)
        
        #extracting columns
        self.restaurant_names=list(file['restaurantname'])
        self.price_range=list(file['pricerange'])
        self.area=list(file['area'])
        self.food_types=list(file['food'])
        self.phone=list(file['phone'])
        self.address=list(file['addr'])
        self.post_code=list(file['postcode'])
        self.good_food=list(file['good food'])
        self.open_kitchen=list(file['open kitchen'])
        self.hygiene=list(file['hygiene'])
        self.suggestions=[]
        self.delay=1
        self.responses_formal={"Welcome": 
            ["Hello! I can recommend restaurants to you. To start, please enter your request. You can ask for restaurants by area, price range, or food type\n",
             "Hello and welcome to our restaurant system. You can ask for restaurants by area, price range, or food type. To start, please enter your request\n",
             "Hello! You can ask for restaurants by area, price range, or food type. How may I help you?\n"],
        
            "Area":
                ['What part of town do you have in mind?\n'],
            'Price':
                ['What is your desired price range? Cheap, moderate, or expensive?\n',
                 'Would you prefer a cheap, moderate, or expensive restaurant?\n'],
            'Food':
                ["What kind of food would you like? \n",
                "What type of food do you prefer?\n" ],
            
            "AffirmPreferences":
                ['So, you are looking for a restaurant in the {0} part of town, with {1} price range, serving {2} food, correct?\n'], 
            
            "Answer":
                ["Okay, here is your recommendation: '{}'. Is it fine? \n",
                "I have found a nice restaurant matching your preferences: ‘{}’. Do you like it? \n",
                "I can recommend a good place that fits your criteria: ‘{}’. Is it fine? \n"],
            
            "NoOptions":
                ["Sorry, there are no recommendations matching your demands.Let's try finding something else\n",
                "Unfortunately, I couldn’t find a restaurant that matches your expectations. Let’s try to find something else \n"],
                
            'Goodbye':
                ["Thank you for using our restaurant system. Come back! \n",
                "Thank you, I hope I was useful. Do come back! \n"]
                }

        self.responses_informal={
                "Welcome": 
                    ["Hi, let’s choose a restaurant! Where do you want to eat? Area, price range, food type?\n"],
         
                "Area":
                    ["What part of town do you have in mind?\n"],
                "Price":
                    ["What’s your budget? Cheap, moderate, or expensive?\n"],
                "Food":
                    ["What sort of food would you like? \n"],
                
                "AffirmPreferences":
                    ["So, you want a restaurant in the {0} part of town, with {1} price range, serving {2} food, am I right?\n"], 
                
                "Answer":
                    ["Okay, I came up with a recommendation: '{}'. Sounds good? \n",
                    "I have found a cool place matching your preferences: ‘{}’. You like it? \n"],
                
                "NoOptions":
                    ["Sorry, there’s nothing matching your needs. Wanna try something else? \n"],
                    
                "Goodbye":
                    ["Thanks, hope it was useful. See you! \n"]
                    }
        self.responses=self.responses_informal
            
        self.implication_rules={ "cheap,good food":["busy"],
            "spanish":["long time"], 
            'busy':['long time','not romantic'], 
            'long time':['not children', 'romantic'],  
            'not hygiene,open kitchen':['not romantic'],
            'not good food,not hygiene':['not busy'],
            'open kitchen':['children'],
            'long time,not open kitchen':['boring'],
            'boring,expensive':['not busy'],
            'boring':['not romantic']
            }
            
        
        
        
    #%%
    def start_dialogue(self):
        self.dialogue("", "init", [0,0,0])
    #%%
        '''
    def configure_formality(self, formality):
        #formality needs to be a bool. True=formal, False=informal. Standard False.
        if formality==True:
            self.responses=self.responses_formal
    def configure_delay(self, time_delay):
        #time_delay in seconds.
        self.delay=time_delay
            
    def configure(self,user_input):
        """
        dialog agent configuration options

        Parameters
        ----------
        user_input : str
            user utterance.

        Returns
        -------
        None.

        """
        if user_input=="configure formal":
            self.configure_formality(True)
        elif user_input=="configure informal":
            self.configure_formality(False)
        elif user_input=="configure delay":
            self.configure_delay(0.5)
        elif user_input=="configure no delay":
            self.configure_delay(0)
            '''
    #%%
    
    def dialogue(self, user_input, state, user_preferences):
        """
        recursive state transition function.

        Parameters
        ----------
        user_input : str
            DESCRIPTION.
        state : str
            State of the system.
        user_preferences : list
            list of user preferences (area,price,foodtype).

        Returns
        -------
        None.

        """
        
        if user_input in ["configure formal", "configure delay", "configure informal", "configure no delay"]:
            self.configure(user_input)
            user_input=""
            self.dialogue(user_input,state,user_preferences)
        
        time.sleep(self.delay)
        self.statelog.append([user_input,state]) #tuple of user utterance and its associated state. We use this to keep track of state jumps.
    
        if state == "exit":
            print("Dialog Agent: "+random.choice(self.responses.get("Goodbye")))
            return
        
        if state in ("init"):
            user_preferences = [0,0,0]
            user_input = input("Dialog Agent: "+random.choice(self.responses.get("Welcome"))+"User: ")
            state = self.classification(user_input)
            self.dialogue(user_input, state, user_preferences)
            return
            
        if state in ("inform", "reqalts", 'hello'):
            extracted_preferences = self.preference_extractor(user_input)
            for i,d in enumerate(user_preferences):
                if d == 0:
                    user_preferences[i] = extracted_preferences[i]
            
            state="fill_blanks" #if more slots to be filled
            self.suggestions=self.lookup(user_preferences)

            if (len(self.suggestions)==0) or (len(self.suggestions)==1):
                
                state="answer" #if there is none or 1 restaurant to suggest
            self.dialogue(user_input, state, user_preferences)
            return 
        
        
        if state == "fill_blanks": #ask user for area/foodtype/pricerange
            grounding=self.grounding(user_preferences)
            if user_preferences[0] == 0:
                user_input = input("Dialog Agent: "+grounding+random.choice(self.responses.get("Area"))+"User: ")
                
                state = self.classification(user_input)
                if "area" not in user_input:
                    user_input+=" area"
                if "dont care" in user_input:
                    user_input='any area'
            elif user_preferences[1] == 0:
                user_input = input("Dialog Agent: "+grounding+random.choice(self.responses.get("Price"))+"User: ")
                
                state = self.classification(user_input)
                if "price" not in user_input:
                    user_input+=" price"
                if "dont care" in user_input:
                    user_input='any price'
            elif user_preferences[2] == 0:
                user_input = input("Dialog Agent: "+grounding+random.choice(self.responses.get("Food"))+"User: ")
                
                state = self.classification(user_input)
                if "food" not in user_input:
                    user_input+=" food"
                if "dont care" in user_input:
                    user_input='any food'
            else:
                state='ask_extra_preferences'
            self.dialogue(user_input, state, user_preferences)
            return
        
        
        if state== 'ask_extra_preferences':
            state=self.ask_extra_preferences(user_preferences)
            self.dialogue(user_input, state, user_preferences)
            return
        
        
        if state=="confirmpreferences":
            user_input = input("Dialog Agent: "+random.choice(self.responses.get("AffirmPreferences")).format(user_preferences[0],user_preferences[1],user_preferences[2])+"User: ")
            accept = self.agree(user_input)
            if accept is True:
                self.suggestions = self.lookup(user_preferences)
                state = "answer"
            elif accept is False:
                state = "inform"
                user_input = ""
                user_preferences = [0,0,0]
            elif accept=="reqalts":
                user_preferences=[0,0,0]
            else:    
                state = "accept"
            self.dialogue(user_input, state, user_preferences)
            return
        
        
        if state == "answer": 
            if self.suggestions:  #found at least 1 restaurant
                user_input=input("Dialog Agent: "+self.suggest_restaurant()+"User: ")
                state = self.classification(user_input)
                if state in ["ack", "affirm"]:
                    state = "goodbye"
                elif state in ["reqalts", "reqmore", "deny", "negate"]:
                    state = "answer"
            else: #no restaurants found. Search for alternatives
                alternatives=self.get_alternative_restaurants(self.alternative_preferences(user_preferences))#offer alternatives
                if len(alternatives)==1: #found 1 alternative
                    print("Dialog Agent: "+random.choice(self.responses.get("NoOptions"))+"Let me look for an alternative for you...\n")
                    self.suggestions=alternatives
                    self.recommendation=self.suggestions[0]
                    user_input=input("Dialog Agent: "+self.suggest_restaurant()+"User: ")
                    if self.agree(user_input):
                        self.get_restaurant_contacts(self.recommendation)
                        state="goodbye"
                elif alternatives: #found multiple alternatives
                    print("Dialog Agent: "+random.choice(self.responses.get("NoOptions"))+"Here is a list of alternatives:")
                    for a in alternatives:
                        print("Dialog Agent: "+self.get_restaurant_info(a))
                    user_input = input("Dialog Agent: "+'Would you like to choose one (1) or change your preferences(2)?\n'+"User: ")
                    if user_input=="1":
                        user_input=input("Dialog Agent: "+"Which one would you like to choose?\n"+"User: ")
                        for alternative in alternatives:
                            if dt(user_input.lower(), alternative.lower())<3:# take into account misspellings
                                self.recommendation=alternative
                                state="thankyou"
                    elif user_input=="2":
                        user_preferences=[0,0,0]
                        state='inform'
                    elif user_input=="exit":
                        state='exit'
                    else:
                        print("Dialog Agent: "+"Please choose one of the two options")
                else:#didnt find any alternative
                    print("Dialog Agent: "+random.choice(self.responses.get("NoOptions")))
                    user_preferences=[0,0,0]
                    state='inform'
                    user_input=""
            self.dialogue(user_input, state, user_preferences)
            return
            
        
        if state in ["reqalts","thankyou", "goodbye", "reset"]:
            
            user_input=input("Dialog Agent: "+self.get_restaurant_contacts(self.recommendation)+". Would you like to finish here?\n"+"User: ")

            if (self.classification(user_input) in ("ack","affirm")):
                state="exit"
            else:
                state="init"
            self.dialogue(user_input, state, user_preferences)
            return
        
        
        if state == "repeat":
            try:
                user_input = self.statelog[len(self.statelog) - 3][0]
                state = self.statelog[len(self.statelog) - 3][1]
            except IndexError:
                print("Dialog Agent: "+"Nowhere to go back, starting again\n")
                state = "init"
            self.dialogue(user_input, state, user_preferences)
            return
        
    
        else:
            print("Dialog Agent: "+"I could not understand that, could you phrase it differently?")#statelog[len(statelog) + 1][0]
            state = self.statelog[len(self.statelog) - 2][1]
            self.dialogue(user_input, state, user_preferences)
            return
            
        
    # %%
    def agree(self,user_input):
        """
        check whether user agrees or denies
    
        Parameters
        ----------
        user_input : str
            DESCRIPTION.
    
        Returns
        -------
        bool
            true for agree, false for deny.
    
        """
        response = self.classification(user_input)
        if response in ["ack", "affirm"]:
            return True
        elif response in ["deny", "negate"]:
            return False
        else:
            return response    
    #%%

    def classification(self,phrase):
        y_pred=self.clf_agent.predict(phrase)
        return y_pred
    
    #%%
    def stopwords_removal(self,s):
        tk=w_t(s)
        s=[i for i in tk if not i in (s_w.words('english'))]
        s=" ".join(s)
        s = s.lower()
        s=s.split(" ")
        return s
    
    #%%
    def preference_extractor(self,user_input):
        """
        Parameters
        ----------
        user_input : str
            user utterance.

        Returns
        -------
        user_preferences : list
            list of user preferences extracted from utterance.
            p[0] for the area, p[1] for the price range and p[2] for the food type

        """
        
        user_preferences=[0 for i in range(3)] #the preferences are stored in a list of three elements, p[0] for the area, p[1] for the price range and p[2] for the food type
        s=self.stopwords_removal(user_input)
    
        user_preferences=self.no_preference(user_input, user_preferences) #check if user indicated no preference
        
        #keyword matching for the area
        for i in s:
            for j in self.area:
                if i == j:
                    user_preferences[0] = j
        if(('north' and 'american' in s) and (s.count('north'))>1):
            user_preferences[0]=0
        #keyword matching for the price range
        for i in s:
            for j in self.price_range:
                if i == j:
                    user_preferences[1] = j
                    
        #keyword matching for the food type
        for i in s:
            for j in self.food_types:
                if i == j:
                    user_preferences[2] = j
                elif ('asian' and 'oriental' in s):
                    user_preferences[2]='asian oriental'
                elif ('modern' and 'european' in s):
                    user_preferences[2]='modern european'
                elif ('north' and 'american' in s):
                    user_preferences[2]='north american'
                    
        #In case the area has been mispelt
        if (user_preferences[0] == 0):
            d = {}
            l=[]
            z=['south', 'centre', 'west', 'east', 'north']
            for i in s:
                for j in z:
                    if (dt(i, j)<=2) and i!='want' and i!='eat':   
                        d[j] = dt(i, j)
            for i in d.values():
                l.append(i)
            if len(l)>0:
                k = min(l)
                key_list = list(d.keys())
                val_list = list(d.values())
                if k<=2:
                    user_preferences[0] = key_list[val_list.index(k)]
    
        #In case the price range has been mispelt
        if (user_preferences[1] == 0):
            d = {}
            l=[]
            d = {}
            l=[]
            for i in s:
                for j in list(set(self.price_range)):
                    if (dt(i, j)<=3):   
                        d[j] = dt(i, j)
            for i in d.values():
                l.append(i)
            if len(l)>0:
                k = min(l)
                key_list = list(d.keys())
                val_list = list(d.values())
                if k<=2:
                    user_preferences[1] = key_list[val_list.index(k)]
            
        #In case the  food type has been mispelt                
        #thresolds for Levenshtein distances might need to be better tuned for each preference
        if (user_preferences[2] == 0):
            d = {}
            l=[]
            for i in s:
                for j in list(set(self.food_types)):
                    if (dt(i, j)<=2):   
                        d[j] = dt(i, j)
                    elif (dt('asian',i)<=2 or dt('oriental',i)<=2 in s):
                        d['asian oriental']=min([dt('asian',i),dt('oriental',i)])
                    elif (dt('modern',i)<=2 or dt('european',i)<=2 in s):
                        d['modern european']=min([dt('modern',i),dt('european',i)])
                    elif (dt('north',i)<=2 or dt('american',i)<=2 in s):
                        if('north' and 'american' in s):
                            d['north american']=min([dt('north',i),dt('american',i)])
            for i in d.values():
                l.append(i)
            if len(l)>0:
                k = min(l)
                key_list = list(d.keys())
                val_list = list(d.values())
                if k<=3:
                    user_preferences[2] = key_list[val_list.index(k)]    
        return user_preferences
    #%%
        
    def no_preference(self,user_input, user_preferences):
        """
        check if user indicated no preference by using keyword matching and fill the slot.
    
        Parameters
        ----------
        user_input : str
            input from the user
        user_preferences : list
            list of preferences.
    
        Returns
        -------
        user_preferences : list
            return the user_preferences with keyword any if user indicated no preference.
    
        """
        if "world food" in user_input.lower():
            user_preferences[2]='any'
        
        
        keywords=re.findall( "any\s(\w+)", user_input.lower())
        if ("area" in keywords):
            user_preferences[0]='any'
        if ("price" in keywords):
            user_preferences[1]='any'
        if ("food" in keywords):
            user_preferences[2]='any'
        return user_preferences
 
 
    #%%
    def alternative_preferences(self,user_preferences):
        """
        State: alternative restaurants in state transition diagram. 
        Used if no restaurants can be found in either the "inform" or "fill blank slots" state. 
        Use alternative preferences based on set membership to find additional restaurants.

        Parameters
        ----------
        user_preferences : list
            list of the user preferences.

        Returns
        -------
        area_alternatives : list
            list of alternative area preferences.
        price_alternatives : list
            list of alternative price preferences.
        food_alternatives : list
            list of alternative food type preferences.

        """
        #The input for this function is list S of composed of 3 strings equivalent to user preferences
        #S[0], S[1] and S[2] respectively store area, price range and food type
        a_1=list({'centre', 'north', 'west'})
        a_2=list({'centre', 'north', 'east'})
        a_3=list({'centre', 'south', 'west'})
        a_4=list({'centre', 'south', 'east'})
        areas=[a_1,a_2,a_3,a_4]
        
        
        
        #Price range sub-sets
        
        p_1=list({'cheap', 'moderate'})
        p_2=list({'moderate', 'expensive'})
        prices=[p_1,p_2]
        
        
        #Food type set & sub-sets
        
        
        f_1=list({'thai', 'chinese', 'korean', 'vietnamese','asian oriental'})
        f_2=list({'mediterranean', 'spanish', 'portuguese', 'italian', 'romanian', 'tuscan', 'catalan'})
        f_3=list({'french', 'european', 'bistro', 'swiss', 'gastropub', 'traditional'})
        f_4=list({'north american', 'steakhouse', 'british'})
        f_5=list({'lebanese', 'turkish', 'persian'})
        f_6=list({'international', 'modern european', 'fusion'})
        food_types=[f_1,f_2,f_3,f_4,f_5,f_6]
        #Retrieving the criterias
        s_1=user_preferences[0]
        s_2=user_preferences[1]
        s_3=user_preferences[2]
        
        #Retrieving affiliated subset of area s_1
        k=[]
        for i in range(len(areas)):
            for j in range(len(areas[i])):
                if s_1 in areas[i]:
                  k.extend(areas[i])
        area_alternatives=list(set(k))
        if area_alternatives:
            del area_alternatives[area_alternatives.index(s_1)]  
            
    
        #price
        l=[]
        for i in range(len(prices)):
            for j in range(len(prices[i])):
                if s_2 in prices[i]:
                    l.extend(prices[i])
        price_alternatives=list(set(l))  #remove pairs
        if price_alternatives:
            del price_alternatives[price_alternatives.index(s_2)]
        #food
        food_alternatives=[]
        for i in range(len(food_types)):
            for j in range(len(food_types[i])):
                if s_3 in food_types[i]:
                    food_alternatives=food_types[i] #no possible intersections within these sets
        if food_alternatives:
            del food_alternatives[food_alternatives.index(s_3)]        
        return area_alternatives,price_alternatives,food_alternatives 
    
    #%%
    def get_user_extra_preferences(self,requirement_options,user_input):
        """
        get extra preferences from the user input using keyword matching

        Parameters
        ----------
        requirement_options : list
            list of strings corresponding to the extra options.
        user_input : str
            

        Returns
        -------
        user_requirements : list
            list of extra options extracted using key words.

        """
        user_requirements=[]
        if "closed kitchen" in user_input or 'a la carte' in user_input: 
            user_requirements.append("not open kitchen")
        if "bad food" in user_input: 
            user_requirements.append("not good food")
        for requirement in requirement_options:
            if requirement in user_input:
                
                if "no "+requirement in user_input or "not "+requirement in user_input or 'bad '+requirement in user_input:
                    user_requirements.append("not "+requirement)
                else:
                    user_requirements.append(requirement)
        return user_requirements
    #%%
    def ask_extra_preferences(self,user_preferences):
        """
        state: "ask for extra preferences" in State Transition Diagram.
        Ask the user for additional preferences using keyword matching, suggest restaurant and give reasoning steps.

        Parameters
        ----------
        user_preferences : list
            list of standard preferences used to get restaurants.

        Returns
        -------
        state : str
            transition to thankyou state if the user agrees to a restaurant.

        """
        state='confirmpreferences'
        user_input = input("Dialog Agent: Any other requirements? You can choose from:\nGood food, open kitchen, good hygiene, children friendly, romantic or busy\nUser: ")
        requirement_options=['good food','open kitchen','hygiene', 'children', 'romantic','busy','not boring' ]
        
        user_requirements=self.get_user_extra_preferences(requirement_options,user_input)#extra prefs from user

        self.suggestions = self.lookup(user_preferences)

        for restaurant in self.suggestions:
            
            i=self.restaurant_names.index(restaurant)
            #save restaurant info as knowledge base of restaurant
            restaurant_KB={self.price_range[i], self.area[i], self.food_types[i],self.good_food[i], self.open_kitchen[i],self.hygiene[i]}
            
            #apply implication rules to gain knowledge about restaurant
            applied_rules,KB=self.make_inferences(restaurant_KB)
            

            #check whether to suggest or not and return the rule associated with the decision
            suggest_or_not, (a,c)=self.check_viability(applied_rules, user_requirements)
            if (suggest_or_not):
                self.present_steps(applied_rules)
                user_input=input("{}, this restaurant is recommended because of {}->{}. Would you like to choose this restaurant?\n".format(restaurant.capitalize(),a,c))
                answer=self.classification(user_input)
                if answer in ["affirm", "ack"]:
                    self.recommendation=restaurant
                    state="thankyou"
                    return state
            elif not suggest_or_not:
                if a:
                    self.present_steps(applied_rules)
                    print("{}, this restaurant is not recommended because of {} -> {}".format(restaurant.capitalize(),a,c))
                
           
            else:
                print("No rules applied for {}...".format(restaurant))
        return state
    
    
     

    #%%
    def get_alternative_restaurants(self,alternative_preferences):
        """
        get alternative restaurants based on the alternative preferences extracted using membership

        Parameters
        ----------
        alternative_preferences : list of lists
            

        Returns
        -------
        all_alternative_restaurants : list
            names of found alternatives.

        """
        import itertools
        all_alternative_pref=[]
        all_alternative_restaurants=[]
        for r in itertools.product(alternative_preferences[0], alternative_preferences[1],alternative_preferences[2]): 
            all_alternative_pref.append([r[0], r[1],r[2]])
        for a in all_alternative_pref:
            all_alternative_restaurants.append(self.lookup(a))
        all_alternative_restaurants = [item for sublist in all_alternative_restaurants for item in sublist]

        return all_alternative_restaurants
        
    
        
    

    #%%
    def make_inferences(self,KB):
        """
        Add knowledge to knowledge base KB by making use of implication rules.

        Parameters
        ----------
        KB : set
            convert to list first.

        Returns
        -------
        KB
            as a set, to eliminate duplicates.

        """
        applied_rules={}
        KB=list(KB)
        for knowledge in KB:
            applied_rules[knowledge]=[knowledge]
            for antedecent,consequent in self.implication_rules.items(): #split in antedecent and consequent
                if type(knowledge)==str:
                    if knowledge == antedecent: #if knowledge is the antedecent of the rule
                        for v in consequent:
                            applied_rules[antedecent]=consequent
                            KB.append(v)
                    
                    
                    elif knowledge in antedecent:
                        atoms=antedecent.split(",")
    
                        if (set(atoms) & set(KB) == set(atoms)):
                            applied_rules[antedecent]=consequent
                            KB.extend(consequent)
        return applied_rules,set(KB)  
    #%%
    def check_viability(self, applied_rules,user_requirements):
        """
        return first rule associated with decision to suggest or not suggest a restaurant

        Parameters
        ----------
        applied_rules : dict
            dictionary of rules that were applied.
        user_requirements : list
            list of user requirements.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        for req in user_requirements:
            label1=True
            if "not" in req:
                label1=False
            for a,c in applied_rules.items():
                if not (c!=c):
                    for item in c:
                        
                        label2=True
                        if "not" in str(item):
                            label2=False
                        if req in str(item) and (label1==label2):
                            return True, (a,c)
                        elif req in str(item) and (label1!=label2):
                            return False, (a,c)
        return False, ("","")
                    
                                    
    #%%
    def present_steps(self,applied_rules):
        i=0
        for a,c in applied_rules.items():
            
            print("step {} : {}->{}".format(i, a,c))
            i+=1          
    

        
          
    


    # %%
    def lookup(self,user_preferences):
        """
        Look for restaurants in database using user preferences
    
        Parameters
        ----------
        user_preferences : list
            list of preferences.
    
        Returns
        -------
        res : list
            list of restaurants.
    
        """
        res = list()
    
        fit_area = set()
        fit_price = set()
        fit_food = set()
        
        if user_preferences[0] == "any" or user_preferences[0] == 0:
            fit_area = set(range(len(self.area)))
        else:
            for i,a in enumerate(self.area):
                if a == user_preferences[0]:
                    fit_area.add(i)
        if user_preferences[1] == "any" or user_preferences[1] == 0:
            fit_price = set(range(len(self.price_range)))
        else:
            for j,p in enumerate(self.price_range):
                if p == user_preferences[1]:
                    fit_price.add(j)
        if user_preferences[2] == "any" or user_preferences[2] == 0:
            fit_food = set(range(len(self.food_types)))
        else:
            for k,f in enumerate(self.food_types):
                if f == user_preferences[2]:
                    fit_food.add(k)
        option_numbers = fit_area.intersection(fit_price, fit_food)
        if option_numbers:
            for i in option_numbers:
                res.append(self.restaurant_names[i])
                
        return res
    #%%
    def grounding(self, user_preferences):
        """
        generate sentence for grounding with the user.

        Parameters
        ----------
        user_preferences : list
            list of user preferences.

        Returns
        -------
        answer template with slots filled by user preferences
        """
        #the preferences are stored in a list of three elements, p[0] for the area, p[1] for the price range and p[2] for the food type
        answer_template= "So you would like me to find a restaurant "
        p=user_preferences
        if p[0]:
            answer_template+="in the {} of town ".format(p[0])
        if p[1]:
            answer_template+="priced {}ly ".format(p[1])
        if p[2]:
            answer_template+="serving {} cuisine ".format(p[2])
        return answer_template.rstrip()+". "
        
    #%%
    def get_restaurant_info(self, restaurant_name):
        #return the restaurant information given a restaurant_name
        index=self.restaurant_names.index(restaurant_name)
        
        return "Restaurant '{}' serving {} food in {} part of town for {} price".format(restaurant_name.capitalize(),self.food_types[index], self.area[index], self.price_range[index])
    #%%
    def get_restaurant_contacts(self,recommendation):
        #return the restaurant contact information
        i=self.restaurant_names.index(recommendation)#get index of recommendation
        phone=self.phone[i]
        address=self.address[i]
        return "Alright, here are the contacts:'{}', {}, {}".format(recommendation.capitalize(),phone,address)
    #%%
    def suggest_restaurant(self):
        answer=""
        if len(self.suggestions)==1:
            answer="I could only find one option for you: {}. Is this fine?\n"
        else:
            answer=random.choice(self.responses.get("Answer"))
        self.recommendation=random.choice(self.suggestions)
        self.suggestions.remove(self.recommendation)
        return answer.format(self.recommendation.capitalize())
            