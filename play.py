import numpy as np
import gym
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import pandas as pd
import re
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvolutionalNeuralNetworkModel(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()

        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 26),
            #nn.LogSoftmax(1)
            
        )
    # Predictor
    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = torch.flatten(x,1) #SJEKK OM BATCHENE GÅR OPP MED TOTAL INPUTEN
        #print(x.shape)
        x = self.classifier(x)
        return (x)

    # Cross Entropy loss
    def loss(self, x, y):
         return nn.functional.cross_entropy(self(x), y.argmax(1))

    #def loss(self, x, y):
    #    totalloss = 0
    #    for xi in range(len(self(x))): 
    #        print(self(x)[xi])
    #        print(y[xi])
    #        totalloss += (self(x)[xi]-y[xi])**2
    #        print("xi: " +  str(self(x)))
    #        print("yi: " + str(y[xi]))
    #    return torch.tensor(totalloss)

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()
model = torch.load("X:\data3de\ML\prosjekt\model")
model.eval()

#state, reward, done, info
class StepData:
  def __init__(self, state, reward, done, info):
    self.state = state
    self.reward = reward
    self.done = done
    self.info = info
#Nodes of tree
class Node:
  def __init__(self, gameInfo, visits, totalScore , parent_node, action):
    self.gameInfo = gameInfo
    self.visits = visits
    self.totalScore = totalScore
    self.next_node = []
    self.parent_node = parent_node
    self.action = action
    
  def set_next_node(self, next_node):
    self.next_node.append(next_node)
    
  def get_next_node(self):
    return self.next_node
    
  def set_parent_node(self, parent_node):
    self.parent_node = parent_node
    
  def get_parent_node(self):
    return self.parent_node

  def set_totalScore(self, totalScore):
     self.totalScore = totalScore

  def get_totalScore(self):
    return self.totalScore

  def set_visits(self, visits):
     self.visits = visits

  def get_visits(self):
    return self.visits
  
  def get_gameInfo(self):
    return self.gameInfo

  def set_action(self,action):
    self.action = action
  
  def get_action(self):
    return self.action

class Tree:
    def __init__(self, startNode):
        self.startNode = startNode

    def findBestAction(self, s1, roundNr):## return index of the best move in the discrete action space
        if s1.get_next_node() == []:
          print("This node has no leafNodes")
          return -1
        totalScoreToBeat = float("-inf")
        ##Gets nnullType error
        for i in s1.get_next_node():
          #print("Action: " + str(i.get_action()) + " Value: " + str(i.get_totalScore()/i.get_visits()) + " Visits: " + str(i.get_visits()))
          if(i.get_totalScore()/i.get_visits()>= totalScoreToBeat):
            totalScoreToBeat  = i.get_totalScore()/i.get_visits()
            bestNode = i
        return bestNode.get_action()

    def findWorstAction(self, s1, roundNr):## return index of the best move in the discrete action space
        if s1.get_next_node() == []:
          print("This node has no leafNodes")
          return -1
        totalScoreToBeat = float("inf")
        ##Gets nnullType error
        for i in s1.get_next_node():
          #print("Action: " + str(i.get_action()) + " Value: " + str(i.get_totalScore()/i.get_visits()) + " Visits: " + str(i.get_visits()))
          if(i.get_totalScore()/i.get_visits()< totalScoreToBeat):
            totalScoreToBeat  = i.get_totalScore()/i.get_visits()
            bestNode = i
        return bestNode.get_action()
    
    def get_start_node(self):
      return self.startNode

    def findNodeBasedOnActions(self, actions):
      itNode = self.get_start_node()
      for a in actions:
        for next in itNode.get_next_node():
          if(next.get_action() == a):
            itNode = next
            break
      return itNode


    ##remember to backpropagate the value we get here. UPDATE THE TREE
    def rollout(self, s1, gameEnv): ## Implement neural network
        simOfGame = copy.deepcopy(gameEnv)
        iterationNode = s1
        gamelength = 0
        while True:
            if iterationNode.get_gameInfo().done:
              return iterationNode.get_gameInfo().reward
            elif gamelength > 100:
              return 0
            else:
              ##Feed gamestate into CNN and use for rollout
              #Make gamestate into X
              x = nodeToXInput(iterationNode, simOfGame.turn())
              y = torch.softmax(model(x.unsqueeze(0)),1)
              y = y.cpu()
              y = y.detach().numpy()
              action = -1
              while action == -1:
                if(simOfGame.valid_moves()[np.argmax(y[0])] == 1):
                  action = np.argmax(y)
                y[0][np.argmax(y[0])] = 0
              state, reward, done, info = simOfGame.step(action)
              # Take Y and find Value, if not playable, take next value
              iterationNode = Node(StepData(state, reward, done, info),0,0,None, action)
              gamelength += 1

    def backProp(self, s1, value, stopNode):
        while True:
            s1.set_totalScore(s1.get_totalScore() + value)
            s1.set_visits(s1.get_visits() + 1)
            if s1 == stopNode:
              break
            s1 = s1.get_parent_node()
            
    def expandingTree(self, s1, iteratorNr, gameEnv, blacksTurn):
        #getting leafNode:
        startOfExplorationNode = s1
        cloneOfGameEnv = copy.deepcopy(gameEnv)
         
        while s1.get_next_node() != []:
            if(blacksTurn):
              maxUcb1 = ucb1Black(s1.get_next_node()[0],iteratorNr)
            else:
              maxUcb1 = ucb1White(s1.get_next_node()[0],iteratorNr)
            newS1 = s1.get_next_node()[0]
            for n in s1.get_next_node():
              if blacksTurn:
                newUcb1 = ucb1Black(n,iteratorNr)
              else:
                newUcb1 = ucb1White(n,iteratorNr)
              if newUcb1 >= maxUcb1:
                maxUcb1 = newUcb1
                newS1 = n
            s1 = newS1
            cloneOfGameEnv.step(newS1.get_action())

        #rollingout if the leafnode has not been visited
        #Might be useless if i run each leafnoede at inits
        if s1.get_visits() == 0:
            self.backProp(s1, self.rollout(s1, cloneOfGameEnv),startOfExplorationNode)
            return

        # if not terminal state, add new nodes to s1, and roolout for the fist action
        if(cloneOfGameEnv.game_ended()):
          return

        for i in range(0, sizeOfGame*sizeOfGame+1): #all moves 
          simOfGame = copy.deepcopy(cloneOfGameEnv)
          
          if(simOfGame.valid_moves()[i]==1 ):
            state, reward, done, info = simOfGame.step(i)
            newChild = Node(StepData(state, reward, done, info),0,0,s1,i)
            s1.set_next_node(newChild)
            self.backProp(newChild, self.rollout(newChild,cloneOfGameEnv),startOfExplorationNode)
        
        
        return  

        
#Calc used for tree tranversion
def ucb1Black(node,iterationNr):
    if node.get_visits() == 0 or iterationNr == 0:
        return float('inf')
    return node.get_totalScore()/node.get_visits() + explorationCoefficent *(np.sqrt(np.log(iterationNr)/node.get_visits()))

def ucb1White(node,iterationNr):
    if node.get_visits() == 0 or iterationNr == 0:
        return float('inf')
    return -node.get_totalScore()/node.get_visits() + explorationCoefficent *(np.sqrt(np.log(iterationNr)/node.get_visits()))



#Make this feed into CNN model and get output?
def nodeToXInput(node, blacksTurn):
  board = node.get_gameInfo().state
  if(blacksTurn):
    X = board[0]
    Y = board[1]
  else:
    X = board[1]
    Y = board[0]
  encodedBoard = [[ float(X[i][j] + (Y[i][j] * -1))  for j in range(len(X[0]))] for i in range(len(X))]

  encodedBoard = [encodedBoard]
  encodedBoard = torch.tensor(encodedBoard).to(device= device, dtype=torch.float)
  return encodedBoard

#Coefficents and data
games = []
wins = []
sizeOfGame = 5
#30 runs works for 5x5 with little wait
#100 runs 5x5 is a bit of a wait in the first iterations 
explorationRuns=30


explorationCoefficent = 2
printStatment= explorationRuns/5


##Run Game
for game in range(0,30):
  go_env = gym.make('gym_go:go-v0', size=sizeOfGame, komi=0.5, reward_method='real')
  startNode = Node(StepData(go_env.state(),0,0,0),0,0,None, None)
  gameTree = Tree(startNode)
  startNode = gameTree.get_start_node()
  


#TreeExploration
  actions = []

  currentNode = gameTree.get_start_node()
  done = False
  round = 0
  while not done:
      t = time.process_time()
      for _ in range(0,explorationRuns):
        if(_%printStatment == 0):
          print("Tree Exploration Iteration: " + str(_))
          
        gameTree.expandingTree(currentNode,_,go_env, True)
      print("Time Used simulations: " + str(time.process_time() - t))
      #print("PlayingGame")
      blackAction = gameTree.findBestAction(currentNode ,round)
      actions.append(blackAction)
      print("Blacks best Action is: ", end= "" )
      print(blackAction)
      state, reward, done, info = go_env.step(blackAction)
      go_env.render('terminal')
      currentNode = gameTree.findNodeBasedOnActions(actions)
      
      if(go_env.game_ended()):
          break

      #for _ in range(0,explorationRuns):
      #  if(_%printStatment == 0):
      #    print("Tree Exploration Iteration: " + str(_))
          
      #  gameTree.expandingTree(currentNode,_,go_env, False)
      #whiteAction =int(input("What should white play?: "))
      #print("White round values:")
      #whiteAction = gameTree.findWorstAction(currentNode, round)
      whiteAction = go_env.uniform_random_action()
      print("White run values")
      actions.append(whiteAction)
      
      print("White played: ", end= "" )
      print(whiteAction)
      state, reward, done, info = go_env.step(whiteAction)
      go_env.render('terminal')
      
      if(go_env.game_ended()):
          break

      currentNode = gameTree.findNodeBasedOnActions(actions)
      print("STATE")
      
      print("actions So Far:")
      print(actions)
      
      round +=1
        
  print("RESULT OF MATCH-------------------------------------")
  print(game)
  print(go_env.winning())
  print(go_env.reward())
  print("----------------------------------------------------")
  wins.append(go_env.winning())
  games.append(game+1)

plt.plot(games,wins)
plt.show()
