import os
import sys
import optparse
import subprocess
import random
import numpy as np
import keras
import tensorflow
import datetime
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model
from tensorflow.keras import optimizers

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
PORT = 8874
import traci


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.1  # exploration rate
        self.learning_rate = 0.0002
        self.memory = deque(maxlen=200)
        self.model = self._build_model()
        self.action_size = 2

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(4,1))
        x1 = Dense(64, activation='relu')(input_1)
        x1 = Dense(64, activation='relu')(x1)
        x1 = Dense(64, activation='relu')(x1)
        #x1 = Flatten()(x1)


        input_2 = Input(shape=(4,1))
        x2 = Dense(64, activation='relu')(input_2)
        x2 = Dense(64, activation='relu')(x2)
        x2 = Dense(64, activation='relu')(x2)
        #x2 = Flatten()(x2)

        #input_3 = Input(shape=(1,1))
        #x3 = Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(7, activation='softmax')(x)

        model = Model(inputs=[input_1, input_2], outputs=[x])
        model.compile(optimizer=tensorflow.keras.optimizers.RMSprop(lr=self.learning_rate), loss='mse')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print("act values",act_values)
        return (np.argmax(act_values[0]))  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0] = target
            history=self.model.fit(state, target_f, epochs=1, verbose=0)
            #print(history.history.keys())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class SumoIntersection:

    def chooseMode(self, junctions):
        # This method uses fuzzy logic to determine the mode of operation(Fair, Priority, Emergency)
        mode = ["fair", "priority", "emergency"]
        currentDT = datetime.datetime.now()
        time = currentDT.hour
        wgMatrix = self.mapVehicleToWg(weight_matrix)
        averageLoad = self.calculateReward(wgMatrix)[0]/4
        averageQueueLength = traci.edge.getLastStepHaltingNumber('R8') + traci.edge.getLastStepHaltingNumber(
                        'R39') + traci.edge.getLastStepHaltingNumber('R37') + traci.edge.getLastStepHaltingNumber('R33')
        averageQueueLength = averageQueueLength/4

        # Define vehicleId's Of Emergency Vehicles
        emergencyVIds = []
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('R8')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('R39')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('R37')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('R33')

        # if emergency vehicle is present in any of the road just return emergency mode
        for v in vehicles_road1:
            if len(emergencyVIds) != 0 and  v in emergencyVIds:
                return mode[2]
        for v in vehicles_road2:
            if len(emergencyVIds) != 0 and v in emergencyVIds:
                return mode[2]
        for v in vehicles_road3:
            if len(emergencyVIds) != 0 and v in emergencyVIds:
                return mode[2]
        for v in vehicles_road4:
            if len(emergencyVIds) != 0 and v in emergencyVIds:
                return mode[2]


        fair = 0
        priority = 0
        # Rules for fuzzification (More rules can be added later)
        if (time > 6 and time < 10) or (time > 15 and time < 18):
            fair = fair + 1
        else:
            priority = priority + 1

        if averageQueueLength < 10 :
            fair = fair + 1
        else:
            priority = priority + 1
            fair = fair + 1

        if averageLoad < 10 :
            fair = fair + 1
        else:
            priority = priority + 1
            fair = fair + 1

        fair = fair/3
        priority = priority/3

        #Defuzzification
        if fair > priority and fair <= 0.5:
            return mode[0]

        if priority > fair and priority <= 0.5:
            return mode[1]

        if random.uniform(0,1) < 0.5:
            return mode[0]
        else:
            return mode[1]



    def mapVehicleToWg(self, wgMatrix):
        # Define weights of each vehicle before the simuation
        return wgMatrix

    def findDuration(self, turn, wgMatrix):
        return 5


    def calculateReward(self, wgMatrix):
        reward_wt = 0
        reward_ql=0


        laness=['R8_0','R39_0','R37_1','R33_0']
        #vehicles_road1 = traci.edge.getLastStepVehicleIDs('R43')
        #vehicles_road2 = traci.edge.getLastStepVehicleIDs('R49')
        #vehicles_road3 = traci.edge.getLastStepVehicleIDs('R46')
        #vehicles_road4 = traci.edge.getLastStepVehicleIDs('R48')

        for i in range(len(laness)):
            if (len(weight_matrix) != 0 and v in wgMatrix):
                reward_wt += wgMatrix[v]*traci.lane.getWaitingTime(laness[i])
        for i in range(len(laness)):
            if (len(weight_matrix) != 0 and v in wgMatrix):
                reward_ql += wgMatrix[v]*traci.lane.getLastStepHaltingNumber(laness[i])
        final_reward=reward_wt+ reward_ql
        return [final_reward]



       
    def getState(self):
        #positionMatrix = []
        #velocityMatrix = []
        ql=[]
        wt=[]
        phase=[]

        #cellLength = 7
        #offset = 11
        #speedLimit = 14

        junctionPosition = traci.junction.getPosition('N8')[0]
        vehicles_road1 = traci.edge.getLastStepVehicleIDs('R8')
        vehicles_road2 = traci.edge.getLastStepVehicleIDs('R39')
        vehicles_road3 = traci.edge.getLastStepVehicleIDs('R37')
        vehicles_road4 = traci.edge.getLastStepVehicleIDs('R33')

        for i in range(4):
            ql.append(0)
            wt.append(0)

        laness=['R8_0','R39_0','R37_1','R33_0']
        for i in range(len(laness)):
            ql[i]=traci.lane.getWaitingTime(laness[i])
            wt[i]=traci.lane.getLastStepHaltingNumber(laness[i])


        phase=traci.trafficlight.getPhaseDuration("N8")

        ql1=np.reshape(ql,(1,4,1))


        wt1=np.reshape(wt,(1,4,1))
        phase1=np.reshape(phase,(1,1,1))


        return [ql1, wt1]


if __name__ == '__main__':
    sumoInt = SumoIntersection()
    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run


    #sumoInt.generate_routefile()

    # Main logic
    # parameters
    episodes = 25
    batch_size = 32

    #tg = 10
    #ty = 6
    agent = DQNAgent()
    try:
        agent.load('Models/model_junction_1.h5')
    except:
        print('No models found')

    for e in range(episodes):
        print('EPISODE-->' + str(e))
        # DNN Agent
        # Initialize DNN with random weights
        # Initialize target network with same weights as DNN Network
        log = open('log.txt', 'a')
        step = 0
        waiting_time = 0
        reward1 = 0
        reward2 = 0
        reward = 0
        stepz = 0
        action = 0
        #ph=30
        ch=[10,20,30,10,20,30]
        PORT = PORT + 1
        sumoBinary = checkBinary('sumo-gui')
        sumoProcess = subprocess.Popen([sumoBinary, "-c", "map/gwalior.sumocfg", "--tripinfo-output",
                                    "tripinfo.xml","--quit-on-end","false","--start","true", "--remote-port",str(PORT)], stdout=sys.stdout, stderr=sys.stderr)
        traci.init(PORT)
        junctions = traci.trafficlight.getIDList()
        traci.trafficlight.setPhase("N8", 0)
        traci.trafficlight.setPhaseDuration("N8", 200)
        cpd1=traci.trafficlight.getPhaseDuration("N8")
        print("cycle phase duration initial", cpd1)
        cpd=cpd1
        turn = 0
        # contains vehicle to its wg matrix
        weight_matrix = []
        while traci.simulation.getMinExpectedNumber() > 0 and stepz < 10000:
            traci.simulationStep()
            state = sumoInt.getState()
            action = agent.act(state)
            mode = sumoInt.chooseMode(junctions)
            # map vehicle to wg based on mode of operation
            if mode == "priority" or mode == "emergency":
                sumoInt.mapVehicleToWg(weight_matrix)
            # state change
            #print("weight_matrix",weight_matrix)
            if(action==0 or action == 1 or action ==2):
                turn = (turn + 1) % 4
                reward1 = sumoInt.calculateReward(weight_matrix)[0]
                for i in range(5):
                    stepz += 1
                    traci.trafficlight.setPhase("N8", turn)
                    # print(sumoInt.findDuration(turn, weight_matrix))
                    #traci.trafficlight.setPhaseDuration("N7", 20)                    
                traci.simulationStep()
                reward2 = sumoInt.calculateReward(weight_matrix)[0]
                reward = reward1 + reward2
                cpd=cpd1+(ch[action]*cpd1)/100
                if (cpd>=20 and cpd<=80):
                    traci.trafficlight.setPhaseDuration("N8", cpd)
                else:
                    traci.trafficlight.setPhaseDuration("N8", 80)
                print("cycle phase duration final and action+", cpd,action)

	        # no state change
            if(action==3 or action == 4 or action ==5):
                turn = (turn + 1) % 4
                reward1 = sumoInt.calculateReward(weight_matrix)[0]
                for i in range(5):
                    stepz += 1
                    traci.trafficlight.setPhase("N8", turn)
                    # print(sumoInt.findDuration(turn, weight_matrix))
                    #traci.trafficlight.setPhaseDuration("N7", 20)                    
                traci.simulationStep()
                reward2 = sumoInt.calculateReward(weight_matrix)[0]
                reward = reward1 + reward2
                cpd=cpd1-(ch[action]*cpd1)/100
                if (cpd>=20 and cpd<=80):
                    traci.trafficlight.setPhaseDuration("N8", cpd1)
                else:
                    traci.trafficlight.setPhaseDuration("N8", 20)
                print("cycle phase duration final and action, action-", cpd,action)


            
            new_state = sumoInt.getState()
            print("JJJJJJJJJJJ",state, action, reward, new_state)
            agent.remember(state, action, reward, new_state, False)
            # Randomly Draw 32 samples and train the neural network by RMS Prop algorithm
            if(len(agent.memory) > batch_size):
                agent.replay(batch_size)
            # print('EPISODE-->' + str(stepz))

        # print('AWT-->' + str(stepz))
        # print('AQLENGTH-->' + str(stepz))
        mem = agent.memory[-1]
        del agent.memory[-1]
        agent.memory.append((mem[0], mem[1], reward, mem[3], True))
        # log.write('episode - ' + str(e) + ', total waiting time - ' +
        #           str(waiting_time) + ', static waiting time - 338798 \n')
        # log.close()
        print('episode - ' + str(e) + ' total waiting time - ' + str(waiting_time))
        traci.close(wait=False)
    agent.save('model_new'+ str(e) + '.h5')
    print("end")
sys.stdout.flush()
