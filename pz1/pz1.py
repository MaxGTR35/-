from random import choice, shuffle, randint
from time import time

def generate_simple_rules(code_max, n_max, n_generate, log_oper_choice=["and","or","not"]):
	rules = []
	for j in range(0, n_generate):

	    log_oper = choice(log_oper_choice)  #not means and-not (neither)
	    if n_max < 2:
		    n_max = 2
	    n_items = randint(2,n_max)
	    items = []
	    for i in range(0,n_items):
		    items.append( randint(1,code_max) )
	    rule = {
	          'if':{
	              log_oper:	 items
	           },
	           'then':code_max+j
	        }
	    rules.append(rule)
	shuffle(rules)
	return(rules)

def generate_stairway_rules(code_max, n_max, n_generate, log_oper_choice=["and","or","not"]):
	rules = []
	for j in range(0, n_generate):

	    log_oper = choice(log_oper_choice)  #not means and-not (neither)
	    if n_max < 2:
		    n_max = 2
	    n_items = randint(2,n_max)
	    items = []
	    for i in range(0,n_items):
		    items.append( i+j )
	    rule = {
	          'if':{
	              log_oper:	 items
	           },
	           'then':i+j+1
	        }
	    rules.append(rule)
	shuffle(rules)
	return(rules)

def generate_ring_rules(code_max, n_max, n_generate, log_oper_choice=["and","or","not"]):
	rules = generate_stairway_rules(code_max, n_max, n_generate -1, log_oper_choice)
	log_oper = choice(log_oper_choice)  #not means and-not (neither)
	if n_max < 2:
	    n_max = 2
	n_items = randint(2,n_max)
	items = []
	for i in range(0,n_items):
	    items.append( code_max-i )
	rule = {
	       'if':{
	          log_oper:	 items
	       },
	       'then':0
	       }
	rules.append(rule)
	shuffle(rules)
	return(rules)

def generate_random_rules(code_max, n_max, n_generate, log_oper_choice=["and","or","not"]):
	rules = []
	for j in range(0, n_generate):

	    log_oper = choice(log_oper_choice)  #not means and-not (neither)
	    if n_max < 2:
		    n_max = 2
	    n_items = randint(2,n_max)
	    items = []
	    for i in range(0,n_items):
		    items.append( randint(1,code_max) )
	    rule = {
	          'if':{
	              log_oper:	 items
	           },
	           'then':randint(1,code_max)
	        }
	    rules.append(rule)
	shuffle(rules)
	return(rules)

def generate_seq_facts(M):
	facts = list(range(0,M))
	shuffle(facts)
	return facts

def generate_rand_facts(code_max, M):
	facts = []
	for i in range(0,M):
		facts.append( randint(0, code_max) )
	return facts

def validate_rules(facts, rules):
	for j in rules:
		mas = []
		keys = []
		for value in j.values():
			mas.append(value)
		for key in mas[0].keys():
			keys.append(key)

		if keys[0] == 'or':
			for i in facts:
				if i in mas[0]['or']:
					facts.append(mas[1])

		elif keys[0] == 'and':
			if (set(facts) & set(mas[0]['and'])) == set(facts):
				facts.append(mas[1])

		elif keys[0] == 'not':
			if len(set(facts) & set(mas[0]['not'])) == 0:
				facts.append(mas[1])
	return set(facts)

a = generate_rand_facts(1000, 1000)
b = generate_simple_rules(1000, 4, 10000)

# a = [1,2,3]
#
# b = [
# 	{'if': {'and': [1, 2, 100]}, 'then': 1000},
#     {'if': {'not': [1001]}, 'then': 1002},
# 	{'if': {'or': [1, 2, 100]}, 'then': 1001},
# ]
time_start = time()
print(validate_rules(a,b))

time_start = time()
print("%d facts validated vs %d rules in %f seconds" % (100,1000,time()-time_start))