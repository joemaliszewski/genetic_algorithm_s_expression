#!/usr/bin/env python
# coding: utf-8

#INSTRUCTIONS TO RUN CODE WITH DATA SET BEERSALES#

# Note: unfortunately I could not get docker working with my file. It was producing strange values #although I did my best to get it working. Furthermore, some tests were using 5 seconds which was not long enough to get. further more the function lambda already has functionalty in python for lambas functions.lambda is therefore denoted as l

#apologies for this. I have the following instructions which allow the program to be run from bash that work for me with beer dataset time series data. 

# example commands for testing:

# Question 1: Evaluation of expression:
# "python jtm812.py  -question 1  -expr "(div (exp (ifleq -0.155378496782 1.03118717712 0.206816441971 1.03118717712)) (log 1.09965249175))" -n 10 -x "-0.695851789742 -0.189900702591 -1.31924661442 1.45786585151 -2.31217027969 -0.549620673918 0.250946596997 1.36131035577 -0.96456960303 0.200333135047"

# Question 2: Evaluation of the fitness function:
# python jtm812.py  -question 2  -expr "(exp 1.92108563144)" -n 2 -m 1000 -data ‘beer_sales’

# Question 3: Does the GP produce better solutions? (approx 10 min)
# python jtm812.py -question 3 -n 2 -m 478 -data 'beer_sales' -l 200 -time_budget 600



# # Lab 2 MSc: Time Series Prediction

import random
import re
import numpy as np
import math
from statistics import mean 
import matplotlib.pyplot as plt
import copy
import signal
from contextlib import contextmanager
import argparse
import time


#The code for the parser was taken from : http://rosettacode.org/wiki/S-Expressions
dbg = False
 
term_regex = r'''(?mx)
    \s*(?:
        (?P<brackl>\()|
        (?P<brackr>\))|
        (?P<num>\-?\d+\.\d+|\-?\d+)|
        (?P<sq>"[^"]*")|
        (?P<s>[^(^)\s]+)
       )'''

def parse_sexp(sexp):
    stack = []
    out = []
    if dbg: print("%-6s %-14s %-44s %-s" % tuple("term value out stack".split()))
    for termtypes in re.finditer(term_regex, sexp):
        term, value = [(t,v) for t,v in termtypes.groupdict().items() if v][0]
        if dbg: print("%-7s %-14s %-44r %-r" % (term, value, out, stack))
        if   term == 'brackl':
            stack.append(out)
            out = []
        elif term == 'brackr':
            assert stack, "Trouble with nesting of brackets"
            tmpout, out = out, stack.pop(-1)
            out.append(tmpout)
        elif term == 'num':
            v = float(value)
            if v.is_integer(): v = int(v)
            out.append(v)
        elif term == 'sq':
            out.append(value[1:-1])
        elif term == 's':
            out.append(value)
        else:
            raise NotImplementedError("Error: %r" % (term, value))
    assert not stack, "Trouble with nesting of brackets"
    return out[0]

def print_sexp(exp):
    out = ''
    if type(exp) == type([]):
        out += '(' + ' '.join(print_sexp(x) for x in exp) + ')'
    elif type(exp) == type('') and re.search(r'[\s()]', exp):
        out += '"%s"' % repr(exp)[1:-1].replace('"', '\"')
    else:
        out += '%s' % exp
    return out

      

def evaluate_QU1(sexpr, data):
    #agregated result
    agg_result = 0.0
    if sexpr == 'x':
        sexpr = data
        agg_result = sexpr 

    #checks if leaf or list. If nested list, will traverse inside recursively
    if type(sexpr) == list:
        
        all_nest_args = [evaluate_QU1(nest_arg, data) for nest_arg in sexpr[1:]]
        
        if sexpr[0] == 'add':
                agg_result = 0.0
                for nest_arg in all_nest_args:
                    agg_result = agg_result + nest_arg
                             
        elif sexpr[0] == 'sub':
            agg_result = all_nest_args[0] - all_nest_args[1]
                
        elif sexpr[0] == 'div': 
            if(all_nest_args[0]==0 or all_nest_args[1]==0):
                agg_result = 0
            else: 
                agg_result = (all_nest_args[0]/all_nest_args[1])
                
        elif sexpr[0] == 'mul': 
                agg_result = 1
                for nest_arg in all_nest_args:
                    agg_result = agg_result* nest_arg
                             
        elif sexpr[0] == 'log':
            if len(all_nest_args) == 1:
                if (all_nest_args[0] <= 0):
                    agg_result = 0

                else:
                    try:
                
                        agg_result = math.log(all_nest_args[0], 2) 
                        
                    except:
                        agg_result = 0
                    
            elif len(all_nest_args) == 2:
                if (all_nest_args[0]<=0 or all_nest_args[1]<=0):
                    agg_result = 0
                else:
                    try:
                        agg_result = math.log(all_nest_args[0], all_nest_args[1])
                    except:
                        agg_result = 0
                     
        elif sexpr[0] == 'sqrt':
            if all_nest_args[0] < 0:
                agg_result = 0
            else:
                agg_result = math.sqrt(all_nest_args[0])
                 
        elif sexpr[0] == 'exp':
            try:
                agg_result = math.exp(all_nest_args[0])
            except:
                
                agg_result = 0
            
        elif sexpr[0] == 'max':
            agg_result = max(all_nest_args[0], all_nest_args[1])
               
        elif sexpr[0] == 'ifleq':
            if (all_nest_args[0] < all_nest_args[1]):
                agg_result = all_nest_args[2]
                
            elif (all_nest_args[0] > all_nest_args[1]):
                agg_result = all_nest_args[3]
        
        elif sexpr[0] == 'data':
            agg_result = data[all_nest_args[0]]
                   
        elif sexpr[0] == 'diff':
            agg_result = abs(all_nest_args[0] - all_nest_args[1])
             
        elif sexpr[0] == 'avg':
            try:
                if all_nest_args[1] == all_nest_args[0]:
                    agg_result = 0
                else:
                    agg_result = mean(all_nest_args)
            except:
                    agg_result = 0
                   
        elif sexpr[0] == 'pow':
            try:
                agg_result = math.pow(all_nest_args[0], all_nest_args[1])
            except:
                agg_result = 0
              
        #extentions
        elif sexpr[0] == 'sin':
            try:
                agg_result = math.sin(all_nest_args[0])
            except:
                agg_result = 0
          
    elif sexpr == 'e':
            res = math.e     
    elif type(sexpr) is str:
        if sexpr == 'pi':
            res = math.pi
        
    elif type(sexpr) in [int, float]:
        agg_result = sexpr
        
                    
    return agg_result



def cal_indiv_fitness(Actual_data, Y_pred):

    num_x_values = len(Actual_data)
    Y = Actual_data
    
    invalid = False
    cumulative_loss = 0

    for i in range(len(Y)):
        try:
            loss = ((abs(Y[i] - Y_pred[i])**2))
            
        except:
            loss = 999999999
            invalid = True
            
            if invalid == True:
                MSE_loss = "INVALID"
                return MSE_loss
            
        cumulative_loss = cumulative_loss + loss
    
    MSE_loss = cumulative_loss//len(Actual_data)      
       
    return MSE_loss


def evaluate(sexpr, data):
    
    #agregated result
    agg_result = 0.0
    
    ###ADDED HIGH ORDER POLYNOMIAL TO BETTER RESULTS##
    #add in high polynomial probailty in addition to x,(as low probabilty of long polynomials naturally)
    if sexpr == 'x':
        sexpr = data
        agg_result = sexpr 

    #checks if leaf or list. If nested list, will traverse inside recursively
    if type(sexpr) == list:
        
        
        all_nest_args = [evaluate(nest_arg, data) for nest_arg in sexpr[1:]]
        
        
        if sexpr[0] == 'add':
                agg_result = 0.0
                for nest_arg in all_nest_args:
                    agg_result = agg_result + nest_arg
                             
        elif sexpr[0] == 'sub':
            agg_result = all_nest_args[0] - all_nest_args[1]
                
        elif sexpr[0] == 'div': 
            if(all_nest_args[0]==0 or all_nest_args[1]==0):
                agg_result = 0
            else: 
                agg_result = (all_nest_args[0]/all_nest_args[1])
                
        elif sexpr[0] == 'mul': 
                agg_result = 1
                for nest_arg in all_nest_args:
                    agg_result = agg_result* nest_arg
                             
        elif sexpr[0] == 'log':
            if len(all_nest_args) == 1:
                if (all_nest_args[0] <= 0):
                    agg_result = 0

                else:
                    try:
                        agg_result = math.log(all_nest_args[0])   
                    except:
                        agg_result = 0
                    
            elif len(all_nest_args) == 2:
                if (all_nest_args[0]<=0 or all_nest_args[1]<=0):
                    agg_result = 0
                else:
                    try:
                        agg_result = math.log(all_nest_args[0], all_nest_args[1])
                    except:
                        agg_result = 0
                     
        elif sexpr[0] == 'sqrt':
            if all_nest_args[0] < 0:
                agg_result = 0
            else:
                agg_result = math.sqrt(all_nest_args[0])
                 
        elif sexpr[0] == 'exp':
            try:
                agg_result = math.exp(all_nest_args[0])
            except:
                
                agg_result = 0
            
        elif sexpr[0] == 'max':
            agg_result = max(all_nest_args[0], all_nest_args[1])
               
        elif sexpr[0] == 'ifleq':
            if (all_nest_args[0] < all_nest_args[1]):
                agg_result = all_nest_args[2]
                
            elif (all_nest_args[0] > all_nest_args[1]):
                agg_result = all_nest_args[3]
        
        elif sexpr[0] == 'data':
            agg_result = data[all_nest_args[0]]
                   
        elif sexpr[0] == 'diff':
            agg_result = abs(all_nest_args[0] - all_nest_args[1])
             
        elif sexpr[0] == 'avg':
            try:
                if all_nest_args[1] == all_nest_args[0]:
                    agg_result = 0
                else:
                    agg_result = mean(all_nest_args)
            except:
                    agg_result = 0
            
                  
        elif sexpr[0] == 'pow':
            try:
                agg_result = math.pow(all_nest_args[0], all_nest_args[1])
            except:
                agg_result = 0
              
        #extentions
        elif sexpr[0] == 'sin':
            try:
                agg_result = math.sin(all_nest_args[0])
            except:
                agg_result = 0
        
          
    elif sexpr == 'e':
            res = math.e     
    elif type(sexpr) is str:
        if sexpr == 'pi':
            res = math.pi
        
    elif type(sexpr) in [int, float]:
        agg_result = sexpr
        
                    
    return agg_result

def generate_individual(nesting_threshold, const_threshold):
    
    # -----------------------------------------------------
    # nesting_threshold : 
    # Increasing the threshold value will reduce nesting likelyhood and 
    # therefore increase chance of a shorter equation. Decreasing the value 
    # will increase chance of a longer equation.
    
    # const_threshold :
    # Increasing the value will increase the likelihood a nest will contain a special constant,
    # decreasing the value will increase the likehood of a numeric constant between 1-100
    # -----------------------------------------------------
    
    operators = ['sin','add', 'div', 'mul', 'sub','div', 'log','exp','diff','sqrt','avg','pow','max']             
                                         
    nest_layer = []
    
    # choose nest operator
    nest_layer.append(random.choice(operators))

    
    if (nesting_threshold < random.uniform(0,1)):
        
        # deeper nesting occurs
        nest_layer.append(generate_individual(nesting_threshold, const_threshold))
        
        # 3rd value chosen is always a constant 
        nest_layer.append(get_rand_constant(const_threshold)) 
    else: 
        
        # No deep nesting, both values are a constants
        for i in range(2):
            nest_layer.append(get_rand_constant(const_threshold))
                 
    return nest_layer
    
    
def get_rand_constant(const_threshold):
    special_constants = ['e', 'pi', 'x', 'x', 'x'] 
    
    if const_threshold < random.uniform(0,1):
        return random.random()*100
    else:
        return random.choice(special_constants)

    return nest_layer

def generate_initial_population(population_limit,
                                Y_actual,
                                nesting_threshold,
                                const_threshold, 
                                birth_fitness_limit):
        # -----------------------------------------------------
    # nesting_threshold : 
    # Increasing the threshold value will reduce nesting likelyhood and 
    # therefore increase chance of a shorter equation. Decreasing the value 
    # will increase chance of a longer equation.
    
    # const_threshold :
    # Increasing the value will increase the likelihood a nest will contain a special constant,
    # decreasing the value will increase the likehood of a numeric constant between 1-100
    
    # birth_fitness_limit :
    # prevents survival of extremely out of range loss values that can slow down program
    # and increases fitness of initial population.
    
    # population limit = size of initial population
    
    # Y_actual is the data we are trying to match to
    # -----------------------------------------------------

        population = []
        population_count = 0
        fitness_scores = []
        while population_count < population_limit:
        
            individual = generate_individual(nesting_threshold, 
                                              const_threshold)
            
            # generate predicted data from equation (individual)
            Y_pred = []    
            num_x_values = len(Y_actual)
            for x_value in range(num_x_values):
                y_pred = evaluate(individual, x_value)
                Y_pred.append(y_pred)
              
            individual_fitness = cal_indiv_fitness(Y_actual, Y_pred)
            
            #prevents exploding decimals
            if individual_fitness is not "INVALID":
                individual_fitness = float(individual_fitness)
                round(individual_fitness, 1)
                
            #prevents repeating or equal individuals 
            #kills at birth if too unfit 
            #prevents invalid individuals from being added
            if individual_fitness != "INVALID" and individual_fitness < birth_fitness_limit and (individual_fitness not in fitness_scores):
                fitness_scores.append(individual_fitness)
                
                population.append((population_count, individual_fitness, individual, Y_pred))
                population_count += 1
                
            
        return  population
    
def visualize(top,population,Y_actual, X):
    
    for i in range(top):
        
        print(population[i][2])
        Y_pred = population[i][3]
            
        plt.figure(figsize=(20,10))
        plt.plot(X, Y_pred)
        plt.plot(X, Y_actual)
        plt.xlabel('months')  
        plt.ylabel('sales')
        plt.title('Beer sales monthly') 
        plt.show()
        
def select_individual(contestants):
    
    #select the k best members to participate in tournement
    
    individual_1_indx = random.randint(0, len(contestants)-1)
    individual_2_indx = random.randint(0, len(contestants)-1)
    
    individual_1_fitness = contestants[individual_1_indx]
    individual_2_fitness = contestants[individual_2_indx]
    
    if (individual_1_fitness > individual_2_fitness):
        survivor = individual_1_fitness
    else:
         survivor = individual_2_fitness
    
    return survivor
    
def select_individuals(num_elites, num_contestants, population):
    
    indiv_ID_inx= 0
    fitness_score_inx=1
    sexpr_inx =2
    predicted_results_inx = 3
   
    survivors = []
    contestants = []
    
    for i in range(num_elites):
        survivors.append(population[i])
        
    for i in range(num_contestants):
        contestants.append(population[i])
     
    for i in range(len(population)-num_elites):
        survivors.append(select_individual(contestants))
        
    return survivors



def mutate_population(num_elites,
                      selected_individuals,
                      intra_mutation_rate,
                      inter_mutation_rate,
                      const_threshold, 
                      branch_threshold,
                      mutant_fitness_limit,
                      mutation_rep_lim, 
                      Y_actual
                     ):
    
    indiv_ID_inx= 0
    fitness_score_inx=1
    sexpr_inx =2
    predicted_results_inx = 3
    mutants = []
    
    # elites go through 
    for elite in range(num_elites):
        mutants.append(selected_individuals[elite])
    
    population_count = 0
    fitness_scores = []
    for indiv in range(len(selected_individuals)-num_elites):
        
        
        random_muation_select = random.uniform(0,1)
        if random_muation_select < intra_mutation_rate:
            invalid = True 
            repeat_mutation_counter = 0
            
            while invalid == True:

                mutant = mutate_individual(selected_individuals[indiv][2], inter_mutation_rate, const_threshold, branch_threshold)
                Y_pred = []    
                num_x_values = len(Y_actual)
                for x_value in range(num_x_values):
                    y_pred = evaluate(mutant, x_value)
                    Y_pred.append(y_pred)

                mutant_fitness = cal_indiv_fitness(Y_actual, Y_pred)
                
                if mutant_fitness in fitness_scores:
                        repeat_mutation_counter = repeat_mutation_counter +1

                #prevents exploding numbers
                if mutant_fitness is not "INVALID" and (mutant_fitness < mutant_fitness_limit):
                    
                    if (mutant_fitness not in fitness_scores or (repeat_mutation_counter > mutation_rep_lim)):
                        mutant_fitness = float(mutant_fitness)
                        round(mutant_fitness, 1)
                        invalid = False
                        mutants.append((population_count, mutant_fitness,mutant, Y_pred))
                        population_count = population_count+1
                        fitness_scores.append(mutant_fitness)
                            
    mutants.sort(key = lambda mutants: mutants[1])                 
    return mutants 



def mutate_individual(individual, inter_mutation_rate, const_threshold, branch_threshold):
    
    operators = ['sin','add', 'div', 'mul', 'sub','div', 'log','exp','diff','sqrt','avg','pow','max'] 
    
    # add polynomial element as too rare in natural evolution 
    special_constants = ['e', 'pi','x', 'x','x','x',]
    
    if type(individual) == list:
        if inter_mutation_rate < random.uniform(0,1):
            rand_arg = random.randint(0, 2)
            if rand_arg == 0:
                individual[rand_arg] = random.choice(operators)

            elif (rand_arg == 1 or rand_arg == 2):
                #BRANCH MUTATION
                rand_select = random.uniform(0,1)
                if(branch_threshold < rand_select):
                    #const to branch mutation
                    if (type(individual[rand_arg]) is not list):
                        individual[rand_arg] = generate_individual(nesting_threshold = 0.3, 
                                                      const_threshold = 0.3)
                    #branch to const mutation
                    elif (type(individual[rand_arg]) == list):
                        individual[rand_arg] = random.random()*100
                    
                #SINGLE MUTATION
                else:
                    if (type(individual[rand_arg]) is list):
                        individual[rand_arg] = mutate_individual(individual[rand_arg], inter_mutation_rate, const_threshold, branch_threshold)

                    elif (type(individual[rand_arg]) is not list):
                        individual[rand_arg]= random.choice(special_constants)
                                 
    return individual 


def parent_crossover(parent_1,parent_2):
    
    #traverse the tree to get map of both parents
    arg_tree_parent_1 = []
    arg_tree_parent_1 = traverse_parent(parent_1, arg_tree_parent_1)
    arg_tree_parent_2 = []
    arg_tree_parent_2 = traverse_parent(parent_2, arg_tree_parent_2)
    
    #take random arg from parent 1 replace with random arg in parent 2.
    rand_branch_parent_1 = random.choice(arg_tree_parent_1)
    rand_branch_parent_2 = random.choice(arg_tree_parent_2)
    child = branch_donation(rand_branch_parent_1,rand_branch_parent_2, parent_2)
    
    return child

    
def traverse_parent(parent, tree):
    
    tree.append(parent)
    if type(parent) == list:
        all_nest_args = [traverse_parent(nest_arg, tree) for nest_arg in parent[1:]]    
    return(tree)

    
def branch_donation(donated_branch,recipient_branch, recipient):
    if type(recipient) == list:
        if recipient == recipient_branch:
            recipient = donated_branch
        
        else:
            recipient_args =[branch_donation(donated_branch, recipient_branch, nest_arg) for nest_arg in recipient[:]]
            return recipient_args
            
    return recipient
    
    
    
def crossover_selected(selected_individuals, num_elites, population, Y_actual, population_limit, birth_fitness_limit):
    
    children = []
    population_count = 0
    for elite in range(num_elites):
            children.append(selected_individuals[elite])
                    
    population_count = population_count + num_elites
    fitness_scores = []
    
    while population_count < population_limit:

        parent_1 = random.choice(selected_individuals)
        parent_2 = random.choice(selected_individuals)

        child = parent_crossover(parent_1[2],parent_2[2])

        Y_pred = []    
        num_x_values = len(Y_actual)
        for x_value in range(num_x_values):
            y_pred = evaluate(child, x_value)
            Y_pred.append(y_pred)

        child_fitness = cal_indiv_fitness(Y_actual, Y_pred)

        if child_fitness is not "INVALID":
            child_fitness = float(child_fitness)
            round(child_fitness, 1)


        #prevents repeating or equal individuals 
        #kills at birth if too unfit 
        #prevents invalid individuals from being added

        if child_fitness != "INVALID" and child_fitness < birth_fitness_limit and (child_fitness not in fitness_scores):
            fitness_scores.append(child_fitness)
            
            children.append((population_count,child_fitness,child,Y_pred))
            population_count = population_count + 1
                
    children.sort(key = lambda children: children[1])
    return children      



def GA(num_elites,
       num_contestants,
       population,
       Y_actual,
       population_limit,
       birth_fitness_limit,
       intra_mutation_rate,
       inter_mutation_rate,
       const_threshold, 
       branch_threshold,
       mutant_fitness_limit,
       mutation_rep_lim,
      num_generations, 
      minutes):

    generation_count = 0
    top_gen = []
    t_end = time.time() +(60 * minutes)
    #print("time.time() ", time.time())
    
    for i in range(num_generations):
        print("generation_count", generation_count)

        selected_individuals = []


        selected_individuals = select_individuals(num_elites, num_contestants, population)



        children = crossover_selected(selected_individuals, num_elites, population, Y_actual, population_limit, birth_fitness_limit)



        new_generation = mutate_population(num_elites,
                  children,
                      intra_mutation_rate,
                      inter_mutation_rate,
                      const_threshold, 
                      branch_threshold,
                    mutant_fitness_limit,
                                  mutation_rep_lim)




        population = new_generation
        generation_count = generation_count+1

        top_gen.append(new_generation[0][1])

        if(time.time() > t_end):
            print("Overshoot ")
            break
        
    return top_gen, population



def GA(num_elites,
       num_contestants,
       population,
       Y_actual,
       population_limit,
       birth_fitness_limit,
       intra_mutation_rate,
       inter_mutation_rate,
       const_threshold, 
       branch_threshold,
       mutant_fitness_limit,
       mutation_rep_lim,
      num_generations, 
      minutes):

    generation_count = 0
    top_gen = []
    t_end = time.time() +(60 * minutes)
    #print("time.time() ", time.time())
    
    for i in range(num_generations):
       # print("generation_count", generation_count)

        selected_individuals = []


        selected_individuals = select_individuals(num_elites, num_contestants, population)



        children = crossover_selected(selected_individuals, num_elites, population, Y_actual, population_limit, birth_fitness_limit)



        new_generation = mutate_population(num_elites,
                  children,
                      intra_mutation_rate,
                      inter_mutation_rate,
                      const_threshold, 
                      branch_threshold,
                    mutant_fitness_limit,
                                  mutation_rep_lim, Y_actual)



        print("generation ",generation_count, "   current best : ", new_generation[0][1] )
        population = new_generation
        generation_count = generation_count+1

        top_gen.append(new_generation[0][1])
        if(time.time() > t_end):
            print("Overshoot ")
            break
        
    return top_gen, population







        
parser = argparse.ArgumentParser(description = ("Add arguments"))
parser.add_argument('-question',type=int,help="Question Number"  )
parser.add_argument('-expr',type=str,help="expression"  )
parser.add_argument('-n',type=int,help="size of vector"  )
parser.add_argument('-x',type=str,help="vector"  )
parser.add_argument('-m',type=str,help="size of training data (X,Y)"  )
parser.add_argument('-data',type=str,help="training data file"  )
parser.add_argument('-l',type=int,help="population size"  )
parser.add_argument('-time_budget',type=int,help="number of seconds"  )
args = parser.parse_args()


def main(question, expr, n, x, data, m, l, time_budget):
   
    
    
    
    if question == 1:
        
        parsed_vector_x = list(map(float, x.split()))
        
        parsed_sexp = parse_sexp(expr)
        return evaluate_QU1(parsed_sexp,parsed_vector_x)

    elif question == 2:
        
        print("expr  ", expr)
        print("data", data)
        expr = parse_sexp(expr)
        with open(data) as f:
            w, h = [float(x) for x in next(f).split()] # read first line
            array = []
            for line in f: # read rest of lines
                array.append([float(x) for x in line.split()])

        Y_actual = []
        X = []
        for i in range(len(array)):
            Y_actual.append(array[i][1])
            X.append(i)

        Y_pred = []    
        num_x_values = len(Y_actual)
        for x_value in range(num_x_values):
            y_pred = evaluate(expr, x_value)
            Y_pred.append(y_pred)
        fitness_value = cal_indiv_fitness(Y_actual, Y_pred)
        print("FITNESS : ",fitness_value)
        
    elif question == 3:
        
        print("START")
        with open(data) as f:
            w, h = [float(x) for x in next(f).split()] # read first line
            array = []
            for line in f: # read rest of lines
                array.append([float(x) for x in line.split()])
        Y_actual = []
        X = []
        
        for i in range(len(array)):
            Y_actual.append(array[i][1])
            X.append(i)
        
        
        print('question',question)
        print('expr', expr)
        print('n', n)
        print('x', x)
        print('data', data)
        print('m', m)
        print('l', l)
        print('time_budget',time_budget)
        
        
        minutes = time_budget  
        num_elites = 5
        num_contestants = 60
        population_limit = l
        nesting_threshold = 0.3
        const_threshold = 0.5
        birth_fitness_limit = 60000
        num_generations =  40
        intra_mutation_rate = 1
        inter_mutation_rate = 0.2
        const_threshold = 0.5
        branch_threshold =0.5
        mutant_fitness_limit = 70000
        mutation_rep_lim = 5

        population = generate_initial_population(population_limit,
                                        Y_actual,
                                        nesting_threshold,
                                        const_threshold, 
                                        birth_fitness_limit)

        population.sort(key = lambda population: population[1])
        print("Intial POP FINISHED -- generations can take a while, please wait...")



        top_gens, final_pop = GA(num_elites,
        num_contestants,
        population,
        Y_actual,
        population_limit,
        birth_fitness_limit,
        intra_mutation_rate,
        inter_mutation_rate,
        const_threshold, 
        branch_threshold,
        mutant_fitness_limit,
        mutation_rep_lim,
        num_generations, 
        minutes)
        
        print("Fittest Individual score: ", final_pop[0][1])
        print( "Individual  :  ", final_pop[0][2])
        print("")
       
        
        
        
    
    
if __name__ == '__main__':
    print(main(args.question,
               args.expr, 
               args.n,
               args.x,
               args.data,
               args.m,
               args.l,
               args.time_budget))



  
