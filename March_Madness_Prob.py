# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 20:45:06 2020

March Madness Bracket Prob

@author: kaica
"""

Sample_Space = 2**63

#Team format
#Team = [seed,win]
import os
import math
#os.chdir("/Users/kaica/Documents")
import csv
adjust_prob= True #Keep on
num_of_sims =1000 #How Many Simulations do you run, 1000 = quickest,
#                                          10 thousand = Most Normalized,closer max



#First, Second, Third Writing
ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n/10%10!=1)*(n%10<4)*n%10::4])

#Below are all of the Brackets
import Brackets_to_Import
from Brackets_to_Import import Person_Brackets_list

test_list = Person_Brackets_list()
import numpy as np
import pandas as pd
def Game_Simulator(team1, team2,round_num):
    """try:

    except TypeError:
        team1[3] = 0
        team2[3] = 0
        return 
    """
    team1_seed = team1[1]
    team2_seed = team2[1]  
    game_prob = seed_prob[team1_seed][team2_seed]
    game_prob_2 = seed_prob[team2_seed][team1_seed]
    Team2_Won= 0
    Team1_Won= 0
    Team1_Lost= 0
    Team2_Lost= 0
    In_there = False
    for rower in Teams_array:
        

        if team1[0] == rower[1]:
            In_there = True
            added_prob = rower[7+round_num]
            #print(rower)
            if adjust_prob == True:
                if float(added_prob) == 0:
                    game_prob = 0
                    Team1_Lost = True
                    
                elif float(added_prob) == 1:
                    game_prob = 1
                    Team1_Won = True
                else:
                    #print(added_prob)
                     #Toggle if you don't just want the baseline
                        
                        game_prob = added_prob
                        #game_prob = float(game_prob + float(added_prob))/2 #Normalize for your seed, 
                        for divided_prob in range(round_num):
                            game_prob = game_prob/rower[7+round_num-(divided_prob+1)] #Divides the current rounds probability by the previous round
            else:
                game_prob = game_prob
    if In_there == False:
        game_prob = 0 
    In_there = False
    for rower in Teams_array:
        if team2[0] == rower[1]:
            In_there = True
            added_prob = rower[7+round_num]
            #print(added_prob)
            #print(added_prob)
            if adjust_prob == True:
                if float(added_prob) == 0:
                    game_prob_2 = 0
                    Team2_Lost = True
                    
                elif float(added_prob) == 1:
                    game_prob_2 = 1
                    Team2_Won = True
                else:
                    #print(added_prob)
                     #Toggle if you don't just want the baseline
                        game_prob_2 = float(game_prob + float(added_prob))/2
            else:
                game_prob_2 = game_prob_2

    if In_there == False:
        game_prob2 = 0 
    randum = np.random.rand()
    """
    if round_num == 5:
        print(game_prob)
        print("Random: " + str(randum))
    """
    #Fix below, this means bad matchups happening, i.e. impossible
    
    if Team2_Lost == True and Team1_Lost == True:
        """
        team1[3] = 0
        team2[3] = 0
        return team1, team2
        """
        print("False Matchup")
        game_prob = seed_prob[team1_seed][team2_seed]
        game_prob_2 = seed_prob[team2_seed][team1_seed]
    
    """Final Probability of beating your opponent Calculation"""
    #Toggling the log function - off makes it more balanced, many teams have high chance to win
        #Log weighs the seeds far more heavily, 1 seed way more likely than 16 seed so that makes sense
            #Add 10 to the log to equalize this somewhat
    
    #Benefit of No Log- Initial Distributions very equal, which is accurate
    
        
    Adj_gameprob = (game_prob/(game_prob_2+game_prob))#**math.log(team1_seed)
    
    Adj_gameprob2 = (game_prob_2/(game_prob_2+game_prob))#**math.log(team2_seed)
    
    Full_gameprob = Adj_gameprob+Adj_gameprob2
    Normalized_Game_prob = Adj_gameprob/Full_gameprob
    #print(Full_gameprob)
    if Team1_Won == True and Team2_Lost == True:
        Normalized_Game_prob = 1
        #print(team1[0] + "won")
    if Team2_Won == True and Team1_Lost == True:
        Normalized_Game_prob = 0
        #print(team1[0] + "lost")

    if randum>=(Normalized_Game_prob):
        team1[3] = 0
        winner = team2
    else:
        team2[3] = 0
        winner = team1
    winner_list.append([winner[0],round_num])
    len(winner_list)
    #print(team1,team2)
    return team1, team2#, winner
    
import tqdm

from random import random
        
def Path_finder(Sixteen_team_array): #Bracket order
    for team in Sixteen_team_array:
        if (team[1] == 1) or (team[1] == 16):
            team.append([1,1,1,1])
        elif (team[1] == 2) or (team[1] == 15):
            team.append([8,4,2,1])
        elif (team[1] == 3) or (team[1] ==  14):
            team.append([6,3,2,1])            
        elif (team[1] == 4) or (team[1] ==  13):
            team.append([4,2,1,1])   
        elif (team[1] == 5) or (team[1] ==  12):
            team.append([3,2,1,1])      
        elif (team[1] == 6) or (team[1] ==  11):
            team.append([5,3,2,1])      
        elif (team[1] == 7) or (team[1] ==  10):
            team.append([7,4,2,1])   
        elif (team[1] == 8) or (team[1] ==  9):
            team.append([2,1,1,1])
        else:
            continue
    return Sixteen_team_array
            #Seed_probs = np.array(spamreader)

def Unused(Used_team_array):
    for teamster in Used_team_array:
        teamster[4]= 'unused'
    return Used_team_array

def Individual_Round(Sixteen_team_array,round_num):
        for team in Sixteen_team_array:
            
            for team2 in Sixteen_team_array:
                
                if (team2[5][round_num:] == team[5][round_num:]) and (team2[4] != 'used') and (team[4] != 'used') and (team != team2) and (team[3] != 0)and (team2[3] != 0):
                    teamer_1 = team
                    teamer_2 = team2
                    team[4] ='used'
                    team2[4] ='used'
                    pteam1, pteam2 =Game_Simulator(teamer_1, teamer_2, round_num)
                    break
        Sixteen_team_array= Unused(Sixteen_team_array)
        return Sixteen_team_array
    
def Round_of_16(Sixteen_team_array):
       
    for Round in range(4):
       Sixteen_team_array = Individual_Round(Sixteen_team_array,Round)
    for teamer in Sixteen_team_array:
        if teamer[3] == 1:
            #print("The Winner in the " + str(teamer[2]) + " is: " + str(teamer[0]))
            return teamer
        
        
def Final_Four(): 
    South_team= Round_of_16(South_16)
    North_team=Round_of_16(North_16)
    East_team= Round_of_16(East_16)
    West_team =Round_of_16(West_16)

    if np.random.rand() > .5:
        South_Outcome, North_Outcome = Game_Simulator(South_team,North_team,4)
        
        East_Outcome, West_Outcome = Game_Simulator(East_team,West_team,4)
    else:
        North_Outcome, South_Outcome  = Game_Simulator(North_team,South_team,4)
        
        West_Outcome,East_Outcome, = Game_Simulator(West_team,East_team,4)
    
    if South_Outcome[3] == 1:
        Finalist_1 = South_Outcome
    else:
        Finalist_1 = North_Outcome
        
    if East_Outcome[3] == 1:
        Finalist_2 = East_Outcome
    else:
        Finalist_2 = West_Outcome
        
    if np.random.rand() > .5: 
        
        Finalist_1,Finalist_2= Game_Simulator(Finalist_1,Finalist_2,5)
    else:
        Finalist_2,Finalist_1= Game_Simulator(Finalist_2,Finalist_1,5)
        
    if Finalist_1[3] == 1:
        #print(Finalist_1)
        return Finalist_1[0]
    else:
        #print(Finalist_2)
        return Finalist_2[0]
def Val_list_Summer(Brackets,winner_list,All_Val_Lists): #Goes through and adds dist for each person

    Person_num =0
    for Single_Brack in Brackets:
        Person_num +=1
        if ("Person " + str(Person_num)) not in All_Val_Lists.keys():
            All_Val_Lists["Person " + str(Person_num)] = []
        
        sumvals= 0
        for val in Single_Brack:
            if val in winner_list:
    
                sumvals += (2**val[1])*10  #Adjusts scoring per round
        #print(sumvals)
        All_Val_Lists["Person " + str(Person_num)].append(sumvals)
    return All_Val_Lists
def takeSecond(elem):
    return elem[1]

def Relative_Place(All_Val_Lists): #Calculate total percentage of finishs in confusion matrix
    Null_conf_matrix = np.zeros(shape= (len(All_Val_Lists),len(All_Val_Lists)))
    places=[]
    
    #List of Places per person
    Final_Mean_Places= np.zeros(shape= (len(All_Val_Lists)))
    Places_Count = []
    for liser in range(len(All_Val_Lists)):
        Places_Count.append([])
    for Unique_sim in range(len(All_Val_Lists["Person 1"])):
        Total_sim_sum_vals = []
        Person_names = []
        for Unique_Person in All_Val_Lists:
            Total_sim_sum_vals.append([Unique_Person,All_Val_Lists[Unique_Person][Unique_sim]])
            Person_names.append(Unique_Person)

        Total_sim_sum_vals.sort(key=takeSecond, reverse = True)
        #print(Total_sim_sum_vals)
        for val in range(len(Total_sim_sum_vals)):
            PErson_place = int(Total_sim_sum_vals[val][0].split(" ")[1])
            Null_conf_matrix[PErson_place-1][val] +=1
            Places_Count[PErson_place-1].append(val+1)
    for Unique_Place in range(len(All_Val_Lists)):
        places.append(str(Unique_Place+1) + " place")    
    df_cm = pd.DataFrame(Null_conf_matrix/(num_of_sims), index = Person_names, columns = places)
    plt.figure(figsize = (10,7))
    seaborn.heatmap(df_cm, cmap = plt.cm.Blues,vmin=0, vmax=1,annot=True)

    plt.title("Relative Finishes of Everyone")
    plt.show()
    
    #Print Projected Most Likely Place
    Person_Mean_Placement = []
    for row in range(len(Places_Count)):
        Person_Mean_Placement.append([Person_names[row],sum(Places_Count[row])/len(Places_Count[row])])
    Person_Mean_Placement.sort(key=takeSecond, reverse = False)
    #Final Placement Print Statement
    print("Mean Placement Projections: ")
    n=0
    for personage in Person_Mean_Placement:
        n+=1
        print(personage[0] + " finishes " + ordinal(n) + " with a mean placement of " + str(personage[1]))
    
    #Box and whisker plot of placements
    plt.figure(figsize = (10,7))
    All_Place_Lists = {}
    Person_numer=0
    for Single_Place in Places_Count:
        Person_numer +=1
        if ("Person " + str(Person_numer)) not in All_Place_Lists.keys():
            All_Place_Lists["Person " + str(Person_numer)] = []
        
        
            All_Place_Lists["Person " + str(Person_numer)] = Single_Place
    df_places = pd.DataFrame(All_Place_Lists)  

    seaborn.boxplot( data = df_places );
    plt.title("Boxplots of Place Distribution")
    
    plt.show() 
    
    bin2s = np.arange(0,len(All_Place_Lists)+.1,1)
    plt.figure(figsize = (10,7))
    for PersPlace in All_Place_Lists:
        seaborn.distplot(All_Place_Lists[PersPlace], label =PersPlace,bins=bin2s)
    plt.title("Density Plot of Place Distribution")
    plt.legend(loc='upper right')
    plt.show()

    
seed_prob=pd.read_csv('Seed_prob.csv', sep=',',header=None)
Teams_array=pd.read_csv('Teams_and_Seed.csv', sep=',',header=None)
Teams_array_date=pd.read_csv('Teams_and_Seed_adj_date.csv', sep=',')
Teams_array_date_mens = Teams_array_date[Teams_array_date['gender'] == 'mens']
Teams_array_date_mens = Teams_array_date_mens[Teams_array_date_mens['rd1_win'] != 0]



Unique_dates = (Teams_array_date_mens['forecast_date']).unique() #Lists unique dates
Unique_dates =np.flip(Unique_dates) #Flips to correct order
import datetime
First_Date= datetime.datetime.strptime(Unique_dates[0],'%m/%d/%Y').date()
for Spec_Date in Unique_dates:
    Final_points_projs = {}
    Val_list=[]
    Val_list2=[]
        
    Sim_list=[]
    Teams_array_date_mens_date = Teams_array_date_mens[Teams_array_date_mens['forecast_date'] == Spec_Date]
    Teams_array_test = np.array(Teams_array_date_mens_date['gender'],
                                Teams_array_date_mens_date['forecast_date'])
    
    Teams_array_test = Teams_array_date_mens_date.filter(['team_id','team_name','team_seed','team_region','playin_flag','team_alive',
     'rd1_win',	'rd2_win',	'rd3_win',	'rd4_win',	'rd5_win'	,'rd6_win'	,'rd7_win'
                                                     ], axis=1)

    
    Teams_array= np.array(Teams_array_test)
    Teams_array[:,6].shape
    from tqdm import tqdm

    for all_sims in tqdm(range(num_of_sims)):
        winner_list = []
        winner_list2 = []
    
        
        #Teams_array_date =Teams_array_date.filter('gender' = 'mens')
    
        
        
        
        South_16 = []
        North_16 = []
        East_16 = []
        West_16 = []
        
        for row in Teams_array:
            if row[3] == 'South':
                South_16.append([row[1],int(row[2][:2]),row[3],1,"Unused"])
            if row[3] == 'Midwest':
                North_16.append([row[1],int(row[2][:2]),row[3],1,"Unused"])
            if row[3] == 'East':
                East_16.append([row[1],int(row[2][:2]),row[3],1,"Unused"])
            if row[3] == 'West':
                West_16.append([row[1],int(row[2][:2]),row[3],1,"Unused"])
        
        
                    
        South_16=Path_finder(South_16)           
        North_16=Path_finder(North_16)
        East_16=Path_finder(East_16)
        West_16=Path_finder(West_16)
        

    
        
        
        Sim_list.append(Final_Four())
        Final_points_projs = Val_list_Summer(test_list, winner_list,Final_points_projs)

    import pandas
    from collections import Counter
    a = Sim_list
    letter_counts = Counter(a)
    df = pandas.DataFrame.from_dict(letter_counts, orient='index')
    import matplotlib.pyplot as plt

    plt.figure(figsize = (10,7))

    df.plot(kind='bar')
    
    
    
    import seaborn
    import pandas as pd

    plt.show()
    
    "Final True Histogram of Possible Values"
    import statistics 
    from matplotlib import pyplot

    #plt.hist(Val_list)
    bins = np.arange(0,2000,10)
    plt.figure(figsize = (10,7))

    for Individual_hist in Final_points_projs:
        pyplot.hist(Final_points_projs[Individual_hist],bins=bins, ec ="k", label=Individual_hist)
        Mode_num = (max(set(Final_points_projs[Individual_hist]), key=Final_points_projs[Individual_hist].count))
        Mean_num = sum(Final_points_projs[Individual_hist])/len(Final_points_projs[Individual_hist])
        print(Individual_hist + " Stats:Mode: " + str(Mode_num)+ " Mean: "+ str(Mean_num) +" Min: " +str(min(Final_points_projs[Individual_hist]))+" Max: " +str(max(Final_points_projs[Individual_hist])))

    pyplot.legend(loc='upper right')
    #plt.xticks(np.arange(0, 2000,400))
    #plt.yticks(np.arange(0, 1000,200)) #Set this constant for gif making, Plot height
    plt.title("Density of Scores of the Given March Madness Bracket at " + str(Spec_Date))

    plt.show()
    
    plt.figure(figsize = (10,7))

    for Individual_hist in Final_points_projs:
        seaborn.distplot(Final_points_projs[Individual_hist],bins=bins, hist = False, label=Individual_hist)
    plt.legend(loc='upper right')

    plt.title("Density of Scores of the Given March Madness Bracket at " + str(Spec_Date))
    plt.figure(figsize = (10,7))
    plt.show()
        
    plt.figure(figsize = (10,7))
    import seaborn as sns
    import matplotlib.pyplot as plt
    df_projs = pd.DataFrame(Final_points_projs)

    seaborn.boxplot( data = df_projs );
    plt.title("Boxplots of Points Distribution")

    plt.show()   
    Relative_Place(Final_points_projs)  
    winner_list.sort(key=takeSecond, reverse = False)

"""
Things to add:
    1) Upset counter/ average upsets in first place finishes
"""
