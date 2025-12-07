import networkx as nx
import pickle
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import random
import community.community_louvain as louvain
from itertools import product

import agent


class citizen: 
    
    def __init__(self):
        self.community = None
        
        self.status = 'susceptible'
        self.baseInfectProb = .4
        self.symptomatic = np.random.random()
        self.superspreader = .3 + .7 * np.random.random() ** 2 
        
        self.communityRestrictions = .9
        self.influencedMitigation = False
        self.riskTolerance = .7 + .3 * np.random.random()    
        self.campaignInfluence = .25 + .75 * np.random.random()
        
        self.symptom_days = 0
        
        self.openToVacc = np.random.random() < .9
        self.openToVaccByReferral = np.random.random() < .6
        self.costToVaccinate = 100 + 100 * np.random.random() 
        
        

    def infect(self):
        self.status = 'infected'
        self.baseInfectProb = 0
        
    
    def vaccinate(self, referral=False):
        acceptVaccination = (
            self.openToVaccByReferral 
            if referral else 
            self.openToVacc
            )
        
        if acceptVaccination: 
            self.status = 'immune'
            self.baseInfectProb = 0
            return True
        else:
            return False

    
    def overnight(self):
        if self.status == 'infected':
            self.symptom_days += 1
        
        if self.symptom_days >= 5:
            self.status = 'removed'
    
    
    def resetInfluence(self): 
        self.influencedMitigation = False
        self.communityRestrictions = .9
        
 
    def contact(self, infected):
        
        if self.status != 'susceptible':
            return
        
        selfRisk = self.riskTolerance
        if self.influencedMitigation: 
            selfRisk *= self.campaignInfluence
        
        variables = [
            self.baseInfectProb,
            selfRisk,
            self.communityRestrictions,
            infected.communityRestrictions,
            infected.riskTolerance,
            infected.superspreader,
            infected.symptomatic,
            ]
        
        # transmission probability
        tp = 1
        for v in variables: 
            tp *= v
        
        if tp > random.random(): 
            self.infect()
            


class world:   
    
    graphFP = "Community.gml"
    
    citizenColors = {
        'susceptible' : '#0650c7',
        'infected' : '#f02400',
        'immune' : '#808080',
        'removed' : '#000000'
        }
        
    def constructGraph(self, N=10000):
        self.G = nx.barabasi_albert_graph(n=N, m=3)
        self.defineCommunities()
        
        for node in self.G.nodes:
            c = citizen()
            c.neighbors = list(self.G.neighbors(node))
            c.community = self.citizenCommunity[node]
            self.G.nodes[node]['citizen'] = c
        
        with open(self.graphFP, 'wb') as f:
            pickle.dump(self.G, f)
    
    
    def loadGraph(self):
        with open(self.graphFP, 'rb') as f:
            self.G = pickle.load(f)
        return self.G
    
    
    def defineCommunities(self):
        partition = louvain.best_partition(self.G, random_state=42)
        partitionSizes = Counter(partition.values())
        sorted_ = sorted(
            partitionSizes.items(), 
            key = lambda x: x[1],
            reverse=True
            )
        
        namedCommunities = [
            ('highlands', 3),
            ('vistas', 7),
            ('midlands', 11),
            ('parkville', 14),
            ('lowlands', 18)
            ]
        
        cid_to_name = {}   
        for name, rank in namedCommunities:
            if rank - 1 < len(sorted_): 
                cid = sorted_[rank - 1][0]
                cid_to_name[cid] = name
        
        self.citizenCommunity = {
            node: cid_to_name.get(cid, None)
            for node, cid in partition.items()
            }
        
        
                
        


class simulator:
    
    def __init__(self):    
        w = world()
        self.G = w.loadGraph()
        
        self.day = 0

        self.infectionLog = set()
        self.log = pd.DataFrame(columns=world.citizenColors.keys())
        self.prevCumulative = 0
        self.newInfections = 0
        
        self.communitySize = len(self.G)
        self.initializeInfection()
        
        self.citizens = [
            self.G.nodes[n]['citizen']
            for n in self.G.nodes
            ]

    
    def get_statuses(self):    
        self.communityStatus = defaultdict(list)
        
        for n in self.G.nodes:
            status = self.G.nodes[n]['citizen'].status
            self.communityStatus[status].append(n)
        
        self.infectionLog.update(self.communityStatus['infected']) 
        self.cumulativeInfections = len(self.infectionLog)
        
        self.newInfections = self.cumulativeInfections - self.prevCumulative
        self.prevCumulative = self.cumulativeInfections
        
        self.sir = {k:len(v) for k, v in self.communityStatus.items()}
        self.log.loc[len(self.log)] = self.sir
  
    
    def initializeInfection(self, infections=5):
        patientZero = np.random.choice(
            self.communitySize, 
            size=infections, 
            replace=False
            )
        
        for node in patientZero: 
            self.G.nodes[node]['citizen'].infect()
    
        self.get_statuses()
    
    
    def overnight(self):        
        for n in self.G.nodes:
            citizen = self.G.nodes[n]['citizen']
            
            if citizen.status != 'infected':
                continue
            
            for j in citizen.neighbors:            
                neighbor = self.G.nodes[j]['citizen']        
                neighbor.contact(citizen)

        for c in self.citizens:
            c.overnight()

        self.get_statuses()        
        self.day += 1
    
    
    def morningReset(self):
        for citizen in self.citizens: 
            citizen.resetInfluence()


        

class treasury: 
    
    def __init__(self, sim):
        self.sim = sim
        
        self.liquid = 3000
        self.passiveIncome = 300
        self.spentToday = 0
        self.spentTotal = 0
        
        self.roiSelect = {
            'community' : [.8, 1.0, 1.8],
            'grant' : [1.0, 4, 4, 4],
            'treasury' : [2]
            }
        
        self.maturationSelect = {
            'community' : 3,
            'grant' : 7,
            'treasury' : 1
            }
        
        self.accounts = []
        
        
    def overnight(self):
        # Driving the account balances
        
        for acct in self.accounts: 
            acct['days_remaining'] -= 1
            
            if acct['days_remaining'] <= 0:  
                self.liquid += acct['payout']
        
        self.accounts = [
            acct for acct in self.accounts
            if acct['days_remaining'] > 0
            ]
        
        if self.sim.day == 10:
            self.passiveIncome = 500
        if self.sim.day == 30: 
            self.passiveIncome = 800
        
        self.liquid += self.passiveIncome
        
    
    def morningReset(self):
        self.spentToday = 0


    def _commitFunds(self, amount, type_):
        if amount > self.liquid: 
            raise ValueError(
                'Insufficient Liquid Funds '
                f'Need: {amount} '
                f'Balance: {self.liquid}'
                )
   
        roi = np.random.choice(self.roiSelect[type_])
        maturation = self.maturationSelect[type_]
        payout = amount * roi
        
        account = {
            'amount' : amount,
            'days_remaining' : maturation,
            'roi' : roi,
            'payout' : payout,
            'type' : type_
            }
        self.accounts.append(account)
        return {'success' : True, 'payout' : payout}
        

    def communityFunding(self, cost=500):
        return self._commitFunds(cost, type_='community')
        
    def researchGrant(self, cost=1000):
        return self._commitFunds(cost, type_='grant')

    def treasuryInvestment(self, value):
        costDict = {'low' : 200, 'med' : 400, 'high' : 600}
        cost = costDict[value]
        
        return self._commitFunds(cost, type_='treasury')
    
    def income(self, cost=0):
        self.liquid += 300
        

        

class vaccine: 
    
    def __init__(self, sim):
        self.sim = sim

        self.susceptibleCitizens = self.sim.citizens
        self.surveyNeighbors = []

        self.vaccineApproval = np.random.choice([18,19,20])
        self.vaccineApproved = False
        self.approvalDate = None
        self.manufactureDay = 0
        self.manufactureMidpoint = 7
        
        self.availableDoses = 0
        self.maxAllocateGeneral = 0
        self.maxAllocateReferral = 0
        
        self.administered = 0
        
        
    
    def overnight(self):         
        approved = self.vaccineApproved
        releasable = self.sim.day >= self.vaccineApproval
 
        if (not approved) and releasable:
            self.vaccineApproved = True
            self.approvalDate = self.sim.day
            
        if self.vaccineApproved:
            self.manufacture()
            
        self.susceptibleCitizens = [
            citizen for citizen in self.sim.citizens
            if citizen.status == 'susceptible'
            ]
            
        self.surveyNeighbors = [
            citizen for citizen in self.surveyNeighbors
            if citizen.status == 'susceptible'
            ]


    def manufacture(
            self,
            max_rate = 200,
            steepness=0.4): 
        
        doses = (
            max_rate / 
            (1 + np.exp(
                -(self.manufactureDay - 
                  self.manufactureMidpoint) * 
                steepness)
                )
            )
        self.availableDoses += int(doses)
        self.manufactureDay += 1
    
    
    def distributionNetworks(self):
        # speeds up the production rate of the vaccine
        self.manufactureMidpoint = np.random.choice([3,4,5])
    
    
    def research(self, speed):
        # speeds up vaccine approval date

        availabilityDates = {
            'low' :  [16,17,18],
            'med' :  [14,15,16],
            'high' : [12,13,14]}
        
        projectedAvailability = availabilityDates[speed]
        self.vaccineApproval = np.random.choice(projectedAvailability)
        return {
            'action' : 'accelerate_research',
            'success' : True,
            'info' : f'Vaccine available on day {self.vaccineApproval}'
            }
    
    
    def getNeighborSocialRanks(self, citizen):
        neighbors = citizen.neighbors
        degrees = [self.sim.G.degree[n] for n in neighbors]
        degree_map = dict(zip(neighbors, degrees))
        ranked = sorted(neighbors, key=lambda n: degree_map[n], reverse=True)
        N = len(ranked)
        nRanks = np.arange(N)
        weights = np.exp(-2 * (nRanks / (N - 1)))
        weights /= weights.sum()
        return ranked, weights
    
    
    def getNeighbors(
            self, 
            nCitizens=1,
            nNeighbors=1, 
            socialAwareness=True
            ):
        
        for citizen in np.random.choice(
                self.sim.citizens, 
                size=nCitizens, 
                replace=False): 
        
            if socialAwareness:
                pool, probs = self.getNeighborSocialRanks(citizen)
            else: 
                pool = citizen.neighbors
                probs = None
    
            size = min(len(pool), nNeighbors)
            neighborIDs = np.random.choice(
                pool, 
                size = size, 
                replace = False,
                p = probs
                )
            
            neighbors = [
                self.sim.G.nodes[node_id]['citizen']
                for node_id in neighborIDs
                ]
            
            self.surveyNeighbors += neighbors
    
    
    def attemptVaccination(self, nCitizens, referral=False):
        if referral: 
            pool = self.surveyNeighbors
            poolName = 'survey candidates'
        else: 
            pool = self.susceptibleCitizens
            poolName = 'susceptible citizens'

        # --- weighted sampling ---
        weightMap = Counter(pool)
        unique_citizens = list(weightMap.keys())
        
        if len(pool) < nCitizens:
            return {
                'action' : 'attempt_vaccination',
                'success' : False,
                'info' : (
                    f'Not enough {poolName}.' 
                    f'Max {len(pool)}'
                    )
                }
        
        weights = np.array(list(weightMap.values()), dtype=float)
        prob = weights / weights.sum()
        chosen = np.random.choice(
            unique_citizens, 
            size=nCitizens, 
            replace=False, 
            p=prob
            )
        
        # --- dose check ---
        if self.availableDoses < nCitizens: 
            return {
                'action' : 'attempt_vaccination',
                'success' : False,
                'info' : (
                    'Not enough available doses.'
                    f'Max: {self.availableDoses}'
                    )
                }

        # --- vaccination attempt ---
        administeredToday = 0
        for citizen in chosen: 
            
            doseAdministered = citizen.vaccinate(
                referral=referral
                )
            
            if doseAdministered: 
                self.availableDoses -= 1
                self.administered += 1
                administeredToday += 1
        
        return {
            'success': True,
            'administered': administeredToday,
            'cost': None,    
            'info': f"Administered {administeredToday} doses"
            }
     

class engagement: 
    
    def __init__(self, sim):
        self.sim = sim
        self.campaignDaysRemaining = 7
    
    def globalAwareness(self, investment):
        marketingMultiplier = {
            'low' : .3,
            'med' : .5, 
            'high' : .7}
        
        probabilityOfInfuence = marketingMultiplier[investment]
        
        x = 0
        for citizen in self.sim.citizens:
            if probabilityOfInfuence > np.random.random():
                x += 1   
                citizen.influencedMitigation = True
        
        
        if False:
            print(
                sum(
                    c.influencedMitigation 
                    for c in self.sim.citizens
                    ) 
                / 
                len(self.sim.citizens)
                )
        

    def communityRestriction(self, community):
        for citizen in self.sim.citizens: 
            if citizen.community == community:
                citizen.communityRestrictions = .4
    
    
    def individualPayments(self, n): 
        if n > len(self.sim.citizens): 
            return {
                'action' : 'individual_payments',
                'success' : False,
                'info' : 'Not enough citizens to influence'
                }
        
        for citizen in np.random.choice(
                self.sim.citizens, 
                size=n, 
                replace=False):
            citizen.riskTolerance = .3 + .3 * np.random.random()
            

class GameState:
    def __init__(self, sim, vaccine, treasury):
        c_stat = sim.communityStatus
        self.day = sim.day
        
        # --- epidemiology ---
        self.susceptible = len(c_stat.get('susceptible', []))
        self.infected = len(c_stat.get('infected', []))
        self.immune = len(c_stat.get('immune', []))
        self.removed = len(c_stat.get('removed', []))
        self.cumulative_infections = sim.cumulativeInfections
        self.new_infections = sim.newInfections
        
        # --- vaccine info ---
        self.vaccine_approved = vaccine.vaccineApproved
        self.available_doses = vaccine.availableDoses
        self.administered_doses = vaccine.administered
        self.days_until_approval = (
            max(0, vaccine.vaccineApproval - sim.day)
            if not vaccine.vaccineApproved else 0
        )
        self.survey_pool_size = len(vaccine.surveyNeighbors)
        
        # --- treasury info ---
        self.liquid = treasury.liquid
        self.active_accounts = len(treasury.accounts)
        self.total_pending_payout = sum(a['payout'] for a in treasury.accounts)
        self.spentToday = treasury.spentToday
        self.spentTotal = treasury.spentTotal
        
        
        # --- community infections ---
        citizens = sim.citizens

        named_comms = ["highlands", "vistas", "midlands", "parkville", "lowlands"]
        infections_by_community = {
            comm: sum(
                1 for c in citizens 
                if c.community == comm and c.status == 'infected'
            )
            for comm in named_comms
            }
        for comm, count in infections_by_community.items():
            setattr(self, f"infections_{comm}", count)
        
        # --- citizen behavioral metrics ---
        self.mean_community_restrictions = float(
            np.mean([c.communityRestrictions for c in citizens])
            )
        
        self.mean_risk_tolerance = float(
            np.mean([c.riskTolerance for c in citizens])
            )
        
        self.mean_campaign_influence = float(
            np.mean([c.campaignInfluence for c in citizens])
            )
        
        self.proportion_influenced = (
            sum(c.influencedMitigation for c in citizens) / len(citizens)
            )


    def as_dict(self):
        return self.__dict__
    

class gameEnvironment: 
    
    def __init__(self, rl_agent=None, verbose=False):
        self.sim = simulator()
        self.vaccine = vaccine(self.sim)
        self.engagement = engagement(self.sim)
        self.treasury = treasury(self.sim)
                
        self._build_actions()
        self.build_discrete_action_list()
        
        self.isHuman = self.verbose = rl_agent is None
        
        self.agent = (
            human(self) if self.isHuman else
            rl_agent        
            )
        
        self.cutoff = 100
        self.end = False
    
        self.state = pd.DataFrame()
        self.get_state()
        
        self.turn_actions = 0
        self.turns_per_day = 3
        self.turn_log = []
    
        self.accelerate_research_used = False
        self.accelerate_distribution_used = False
        self.verbose= verbose


    def dictionaryLookup(self):    
        if self.verbose:
            for action in self.actions: 
                description = self.actions[action].get('def')
                print(f'------ {action} ------')  
                print(description)
                print('')
        
    
    def _build_actions(self):
        self.actions = {
            '*' : {
                'def' : 'Help menu',
                'func' : self.dictionaryLookup,
                'params' : [],
                'cost' : 0
                },
            'income' : {
                'def' : 'Get $300.',
                'func' : self.treasury.income,
                'params' : [],
                'cost' : 0,
                },
            'community_funding' : {
                'def' : (
                    'Invest $500 for 3 days \n' 
                    'Returns one of: $400, $500, or $900.'
                    ),
                'out' : 'Invested $500.\nWill yield ${payout} in 3 days.',
                'func' : self.treasury.communityFunding,
                'params' : [],
                'cost' : 500,
                },
            'research_grant' : {
                'def' : (
                        'Invest $1000 for 7 days. \n'
                        '75% chance of returning: $4000, \n'
                        '25% chance of returning: $1000.'
                        ),
                'out' : 'Invested $1000. Will yield ${payout} in 7 days.',
                'func' : self.treasury.researchGrant,
                'params' : [],
                'cost' : 1000,
                },
            'treasury_investment' : {
                'def' : (
                    'Next day double cash. \n' 
                    'low:  $200 -> $400, \n'
                    'med:  $400 -> $800, \n'
                    'high: $600 -> $1200'
                    ),
                'out' : lambda result, value: (
                    f'Invested ${result["cost"]}, '
                    f'yielding ${round(result.get("payout", "???"))} tomorrow.'
                    ),
                'func' : self.treasury.treasuryInvestment,
                'params' : ['value'],
                'cost' : lambda value: (
                    {'low' : 200, 'med' : 400, 'high' : 600}[value]
                    ),
                'validators' : {
                    'value' : ['low', 'med', 'high']}
                },
            'global_awareness' : {
                'def' : (
                    'Launches an awareness campaign \n'
                    'that has a % chance of convincing \n'
                    'each citizen to avoid risky behavior \n'
                    'for a single day. \n' 
                    'low:  $500  -> 30% \n'
                    'med:  $1000 -> 50% \n'
                    'high: $2000 -> 70%.'
                    ),
                'out' : 'Global awareness campaign launched',
                'func' : self.engagement.globalAwareness,
                'params' : ['investment'],
                'validators' : {
                    'investment' : ['low', 'med', 'high'],
                    },
                'cost' : lambda investment : ( 
                    {'low' : 500, 'med' : 1000, 'high' : 2000}[investment]
                    )
                },
            'community_restriction' : {
                'def' : (
                    'Restricts mobility among citizens \n' 
                    'in a given community, significantly \n'
                    'reducing their risk of contact. \n'
                    'highlands: $800 \n'
                    'vistas:    $600 \n'
                    'midlands:  $400 \n'
                    'parkville: $300 \n' 
                    'lowlands:  $200'
                    ),
                'out' : 'Community restricted for 1 day',
                'func' : self.engagement.communityRestriction,
                'params' : ['community'],
                'validators' : {
                    'community' : [
                        'highlands', 
                        'vistas', 
                        'midlands',
                        'parkville',
                        'lowlands'
                        ]                    
                    },
                'cost' : lambda community: (
                    {'highlands' : 800, 
                     'vistas' : 600,
                     'midlands' : 400,
                     'parkville' : 300, 
                     'lowlands' : 200
                     }[community]
                    ),
                },
            'individual_payments' : {
                'def' : (
                    'Encourage (though not guarantee) \n'
                    'a random citizen to stay inside \n'
                    'for the rest of the game. \n'
                    '$3 per citizen'
                    ),
                'out' : lambda result, n: f'{n} citizens encouraged to stay inside', 
                'func' : self.engagement.individualPayments,
                'params' : ['n'],
                'cost' : lambda n: 3 * n
                },
            'accelerate_research' : {
                'def' : (
                    'Speeds up the initial availability of the vaccine. \n'
                    'no acceleration: available between days 18-20 \n'
                    'low: $6000 -> available between days 16-18 \n'
                    'med: $8000 -> available between days 14-16 \n'
                    'high: $10000 -> available between days 12-14'
                    ), 
                'out' : lambda result, speed: result["info"],
                'func' : self.vaccine.research,
                'params' : ['speed'],
                'validators' : {
                    'speed' : ['low', 'med', 'high']
                    },
                'cost' : lambda speed: (
                    {'low' : 6000, 
                     'med' : 8000, 
                     'high' : 10000
                     }[speed]
                    )
                },
            'accelerate_distribution' : {
                'def' : (
                    'Significantly accelerates the rate \n'
                    'of vaccine dose availability. \n'
                    'Cost: $12000'
                    ),
                'out' : 'Distribution accelearted',
                'func' : self.vaccine.distributionNetworks,
                'params' : [],
                'cost' : 12000
                },
            'survey_neighbors' : {
                'def' : (
                    'Survey a given number of citizens to \n'
                    'refer a random neighbor for vaccination. \n'
                    'If socialAwareness = True, the citizen \n'
                    'is more likely to refer a well-connected \n'
                    'neighbor. \n'
                    '$2 per citizen per random referral, \n'
                    '$4 per citizen per socially-aware referral.'
                    ),
                'out': lambda result, nCitizens, nNeighbors, socialAwareness: (
                        f"{nCitizens * nNeighbors} referred"
                        ),
                'func' : self.vaccine.getNeighbors,
                'params' : [
                    'nCitizens', 
                    'nNeighbors', 
                    'socialAwareness'
                    ],
                'cost' : (
                    lambda **k: (
                        2 * k['nCitizens'] * k['nNeighbors']
                        * (2 if k['socialAwareness'] else 1)
                        )
                    )
                },
            'vaccinate_person' : {
                'def' : (
                    'Attempts to vaccinate a random citizen \n'
                    'in the network. The selected citizen has \n'
                    'a percent chance of refusing the vaccine. \n'
                    'The treasury is still charged, even if the \n'
                    'citizen refuses the vaccine. \n'
                    '$20 per citizen if referred via the survey, \n'
                    '$10 per citizen if random.'
                    ),
                'out' : lambda result, nCitizens, referral: result['info'],
                'func' : self.vaccine.attemptVaccination,
                'params' : ['nCitizens', 'referral'],
                'cost': lambda nCitizens, referral, administered=None: ( 
                    (20 if referral else 10) * nCitizens
                    )
                }                
            }
    
    def build_discrete_action_list(self):
        flat_actions = []
    
        for action_name, action in self.actions.items():
            params = action.get("params", [])
    
            if not params:
                flat_actions.append((action_name, {}))
                continue
    
            validators = action.get('validators', {})
    
            if validators:    
                keys = list(validators.keys())
                values = [validators[k] for k in keys]
    
                for combo in product(*values):
                    flat_actions.append((
                        action_name,
                        dict(zip(keys, combo))
                    ))
                continue
    
            numeric_ranges = {
                "nCitizens": [ 20, 50, 100, 200, 500],
                "nNeighbors": [  10, 50],
                "n": [10, 50, 100, 300, 1000],
                "socialAwareness" : [True, False],
                "referral" : [True, False]
                }
    
            try:
                candidate_lists = [numeric_ranges[p] for p in params]
            except KeyError:
                raise ValueError(f"Missing numeric range for param {params}")
    
            for combo in product(*candidate_lists):
                flat_actions.append((
                    action_name,
                    dict(zip(params, combo))
                ))
    
        self.flat_actions = flat_actions

    
    
    def do(self, action_name, **kwargs):
        
        # --- action validation ---
        if action_name not in self.actions: 
            return {
                'action' : action_name,
                'success' : False,
                'info' : 'Invalid Action'
                }
        
        action = self.actions[action_name]
        
        # --- parameter validation ---            
        missingParams = [
            p for p in action['params'] 
            if p not in kwargs
            ]
        
        if missingParams: 
            return {
                'action': action_name,
                'success': False,
                'info': f"Missing parameters: {missingParams}"
                }
        
        validators = action.get('validators', {})
        for param, allowedValues in validators.items():
            if kwargs[param] not in allowedValues:
                return {
                    "action": action_name,
                    "success": False,
                    "info": (
                        f"Invalid value for '{param}': "
                        f"{kwargs[param]}. " 
                        f"Accepts {allowedValues}."
                        )
                }
        
        # --- compute cost --  
        cost_fn = action['cost']
        
        try: 
            if callable(cost_fn):
                cost = cost_fn(**kwargs)
            else:
                cost = cost_fn
        except Exception as e: 
            return {
                'action' : action_name,
                'success' : False,
                'info' : f'Cost function failed: {e}'
                }
        
            
        # --- check funds --- 
        if cost > self.treasury.liquid: 
            return {
                'action' : action_name,
                'success' : False,
                'cost' : cost,
                'info' : f'Insufficient funds. Need {cost}'
                }
        
        # --- execute --- 
        try: 
            result = action['func'](**kwargs)
        except Exception as e: 
            return {
                'action' : action_name,
                'success' : False, 
                'info' : str(e)
                }
        
        # --- validate result ---
        if result is None: 
            result = {'success' : True}
        elif isinstance(result, dict) and result.get('success') is False:
            return result
        else:
            result.setdefault('success', True)
        
        # --- do cost math ---
        self.treasury.liquid -= cost
        self.treasury.spentToday += cost
        self.treasury.spentTotal += cost
        
        result['cost'] = cost
        result['action'] = action_name
        
        out_def = action.get('out')
        if out_def:
            if callable(out_def):
                try:
                    # Pass both params and result fields
                    result['out'] = out_def(result, **kwargs)
                except Exception as e:
                    result['out'] = f"[Output failed: {e}]"
            else:
                # treat as format string
                try:
                    result['out'] = out_def.format(**result)
                except Exception as e:
                    result['out'] = f"[Output format error: {e}]"
        
        return result


    def take_action(self, action_name, params):
        result = self.do(action_name, **params)

        if result["success"] and action_name != '*':
            self.turn_actions += 1

            #------ tracking for valid actions------
            if action_name == 'accelerate_research':
                self.accelerate_research_used = True
            elif action_name == 'accelerate_distribution':
                self.accelerate_distribution_used = True

        if not result['success']: 
            if self.verbose: 
                print(result['info'])
        
        if 'out' in result: 
            statement = result['out']
            if statement.isdigit(): 
                if self.verbose:
                    print(int(statement))
            else:
                if self.verbose:
                    print(statement)
    
    
        return result
    
    
        
    def _is_terminal(self):
        gs = self.state.iloc[-1]
    
        if gs['infected'] == 0:
            return True
        
        if gs['susceptible'] == 0:
            return True
    
        if gs['day'] >= self.cutoff:
            return True
        
        return False
    
    
    def get_state(self):
        currentState = GameState(
            sim=self.sim,
            vaccine=self.vaccine,
            treasury=self.treasury
            ).as_dict()
        
        if len(self.state) == 0: 
            self.state = pd.DataFrame(
                currentState, 
                index=[0]
                )
        else:
            self.state.loc[len(self.state)] = currentState

        
    
    def printDailyInfo(self):
        print('')
        print('------------------')
        print(f'Day: {self.sim.day}')
        print('Type * for help')
        print('')
        print(self.state.iloc[-1])
        

    def overnight(self):        
        self.sim.overnight()
        self.vaccine.overnight()
        self.treasury.overnight()
        
        self.get_state()
        
        if self.verbose:
            self.printDailyInfo()
        
        self.turn_actions = 0
        self.sim.morningReset()
        self.treasury.morningReset()
        
    
    def parseChoice(self, choice): 
        if choice.get('action'):
            return choice['action'], choice['params']
        else: 
            idx = choice.get('index')
            return self.flat_actions[idx]
        
    
    def play(self, filtered_state=True):
        while not self.end: 
            while self.turn_actions < self.turns_per_day:
                if self.verbose:
                    print('\nCash: ', round(self.treasury.liquid))
                    print(f'{3-self.turn_actions}/3 actions remaining')
                
                state = self.state
                if filtered_state:
                    state = self.get_observation()
                choice = self.agent.orchestrator(state)
                action, params = self.parseChoice(choice)
                self.take_action(action, params)
            
            self.overnight()
            self.end = self._is_terminal()
        print(
            f'Simulation complete after {self.sim.day} days \n'
            f'Cumulative infections: {self.sim.cumulativeInfections}'
            )
    
  
    def reset(self):
 
        w = world()

        try:
            self.G = w.loadGraph()
        except FileNotFoundError:
            w.constructGraph(N=10000)
            self.G = w.loadGraph()
    
        self.sim = simulator()
        self.sim.G = self.G   
 
        for node in self.sim.G.nodes:
            c = citizen()
            c.neighbors = list(self.sim.G.neighbors(node))
            c.community = self.sim.G.nodes[node]['citizen'].community
            self.sim.G.nodes[node]['citizen'] = c
        
        # Update sim's citizen list
        self.sim.citizens = [
            self.sim.G.nodes[n]['citizen']
            for n in self.sim.G.nodes
        ]
 
        self.sim.initializeInfection()
 
        self.vaccine = vaccine(self.sim)
        self.engagement = engagement(self.sim)
        self.treasury = treasury(self.sim)

        self._build_actions()  
        self.build_discrete_action_list() 
        self.end = False
        self.turn_actions = 0
        self.turn_log = []
        self.state = pd.DataFrame()
        self.accelerate_research_used = False
        self.accelerate_distribution_used = False
        
        self.get_state()
        
        return self.get_observation()

    def get_observation(self):

        if len(self.state) == 0:
            self.get_state()
        
        current_state = self.state.iloc[-1]

        features = [
            'day',
            'susceptible',
            'infected',
            'immune',
            'removed',
            'cumulative_infections',
            'new_infections',
            'vaccine_approved',
            'available_doses',
            'administered_doses',
            'days_until_approval',
            'survey_pool_size',
            'liquid',
            'active_accounts',
            'total_pending_payout',
            'spentToday',
            'spentTotal',
            'infections_highlands',
            'infections_vistas',
            'infections_midlands',
            'infections_parkville',
            'infections_lowlands',
            'mean_community_restrictions',
            'mean_risk_tolerance',
            'mean_campaign_influence',
            'proportion_influenced'
        ]

        obs = []
        for feature in features:
            value = current_state[feature]

            if isinstance(value, bool):
                value = int(value)
            obs.append(float(value))
        
        return np.array(obs, dtype=np.float32)


    def step(self, action_idx):

        action_name, params = self.flat_actions[action_idx]
        if action_name == 'vaccinate_person':
            print(f"DEBUG step(): About to vaccinate, doses available: {self.vaccine.availableDoses}")
    
        result = self.take_action(action_name, params)
        if result is None or result['success'] == False:
            # If it is with agent I think raising an error is appropiate
            raise RuntimeError('Unsuccesful Action: ', result['info'])
        
        if self.turn_actions >= self.turns_per_day:
            self.overnight()

        next_obs = self.get_observation()

        done = self._is_terminal()
        reward = self.calculate_reward( )

        info = {
            'action_name': action_name,
            'action_params': params,
            'action_result': result,
            'day': self.sim.day,
            'turn': self.turn_actions,
            'cumulative_infections': self.sim.cumulativeInfections,
            'new_infections': self.sim.newInfections
        }
        
        return next_obs, reward, done, info


    def calculate_reward(self):

        state = self.state.iloc[-1]
    
        reward = 0.0
    
        #reward += state['day'] * 10
        reward -= state['new_infections']  / 10000
    
        if state['administered_doses'] > 0 :
            reward += state['administered_doses'] / 10000

        reward += state['immune']/10000
        #if state['infected'] >0:
           #reward -= (state['infected'] / 20)
        
        #reward -= state['removed']  

        #if state['new_infections'] == 0:
           # reward += 8

        #reward -= next_obs.get('cost', 0) / 200
        
        return reward
    

    def get_valid_actions(self):
        valid_indices = []
        
        accelerate_research_used = hasattr(self, 'accelerate_research_used') and self.accelerate_research_used
        accelerate_distribution_used = hasattr(self, 'accelerate_distribution_used') and self.accelerate_distribution_used

        for idx, (action_name, params) in enumerate(self.flat_actions):
            if action_name == '*':
                continue
            if action_name == 'income':
                valid_indices.append(idx)
                continue
            
            action = self.actions[action_name]
            
            if action_name == 'accelerate_research':
                if self.vaccine.vaccineApproved or accelerate_research_used:
                    continue

            if action_name == 'accelerate_distribution':
                if not self.vaccine.vaccineApproved or accelerate_distribution_used:
                    continue

            if action_name == 'survey_neighbors':
                nCitizens = params.get('nCitizens', 0)
                if nCitizens > len(self.sim.citizens) or nCitizens == 0:
                    continue

            if action_name == 'vaccinate_person':
                nCitizens = params.get('nCitizens', 0)
                referral = params.get('referral', False)

                if nCitizens == 0:
                    continue

                if not self.vaccine.vaccineApproved:
                    continue

                if self.vaccine.availableDoses < nCitizens:
                    continue

                if referral:
                    unique_survey_citizens = len(set(self.vaccine.surveyNeighbors))
                    if unique_survey_citizens < nCitizens:
                        continue
                else:
                    unique_susceptible = len(set(self.vaccine.susceptibleCitizens))
                    if unique_susceptible < nCitizens:
                        continue
        
            
            if action_name == 'individual_payments':
                n = params.get('n', 0)
                if n > len(self.sim.citizens) or n == 0:
                    continue

            cost_fn = action['cost']

            try:
                if callable(cost_fn):
                    cost = cost_fn(**params)
                else:
                    cost = cost_fn

                if cost <= self.treasury.liquid:
                    valid_indices.append(idx)

            except Exception as e:
                raise RuntimeError("Cost calculation failed: ", e)

        return valid_indices
    

class human: 
    
    def __init__(self, game):
        self.game = game
        self._type = 'human'


    def validateAction(self, action):
        if action not in self.game.actions: 
            raise ValueError(f'Invalid action: "{action}"')
        

    def normalizeParams(self, val):
        val_lower = val.strip().lower()
        if val_lower in ('true', 't', 'yes', '1'):
            value = True
        elif val_lower in ('false', 'f', 'no', '0'):
            value = False
        else:
            try:
                value = int(float(val))
            except:
                value = val
        return value
    
    
    def validateParamValue(self, action, param, value):
        validators = self.game.actions[action].get("validators", {})
        if param in validators:
            allowed = validators[param]
            if value not in allowed:
                raise ValueError(
                    f"Invalid value '{value}' for {param}. "
                    f"Allowed: {allowed}"
                    )
                
                
    def orchestrator(self, gameState):
        while True: 
        
            # --- get, validate actions ---
            action = input('Action: ').strip()
            
            try: 
                self.validateAction(action)
            except ValueError as e:
                print(e, " | Type '*' for help.")
                continue
            
            # --- get, validate params ---
            params = {}
            paramList = self.game.actions[action]['params']
    
            paramsValid = True
            for p in paramList:
                val = input(f'{p}: ')
                norm = self.normalizeParams(val)
                
                try: 
                    self.validateParamValue(action, p, norm)
                except ValueError as e: 
                    print(e)
                    paramsValid = False
                    break
            
                params[p] = norm
                
            if not paramsValid:
                continue
            
            return {
                'action' : action, 
                'params' : params,
                'index' : None
                }
            

if __name__ == '__main__':    
    g = gameEnvironment(rl_agent=None)
    g.play()
