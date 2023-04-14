import numpy as np
import random
from time import sleep
color_map = {
    0:'\x1b[1;37;47m   \x1b[0m',
    1:'\x1b[0;30;40m   \x1b[0m',
    -2:'\x1b[1;35;47m @ \x1b[0m',
    -1:'\x1b[1;33;47m * \x1b[0m',
    5:'\x1b[1;32;42m ! \x1b[0m',
    10:'\x1b[1;31;41m ! \x1b[0m',
}

# weight formula
euclidean = lambda a,b: np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
gaussian_kernel = lambda x, y, tau=0.1: np.exp(-(euclidean(x,y) / (2 * (tau**2)))) 

#               l     f      r       b
directions = [(0,-1),(-1,0),(0,1),(1,0)] # front
            
nxt_direction = lambda to: (to+4)%4

def tuple_sum(*args):
    a,b = 0,0
    for x,y in args:
        a += x
        b +=y
    return a,b

class ActionSpace:
    def __init__(self,n):
        self.n = n 
        self.asp = list(range(n))
    def sample(self):
        return random.choice(self.asp)

class Environment:
    def __init__(self,x=10,y=10,max_moves=10000,window=10):
        self.X = x if 10<=x<=50 else 10
        self.Y = y if 10<=y<=50 else 10
        self.CELLS = x * y
        self.avg = (x+y)//2 # avg distance
        self.index = lambda x: (x//self.X,x%self.Y)
        self.action_space = ActionSpace(4)
        self.max_moves = max_moves
        self.state = np.empty((self.X,self.Y),dtype=int)
        self.recording =False
        self.window = window if window <= self.avg else 10


    def reset(self,walls=.25,nds=False):
        # Env Init
        self.fruits = set()
        
        for i in range(self.X):
            for j in range(self.Y):
                self.state[i][j] = 0
        if walls>0.3: walls = 0.3
        if walls >0: self.__con_walls(int(self.CELLS*walls))

        # Player Init
        self.move_count = 0
        self.no_down_synd = nds
        self.direction = 0
        self.body = []
        self.score = 0
        x,y = self.give_free_cell(3,k=1000)
        self.body.append((x,y))
        self.size=1
        self.state[x][y] = -2

        return self.compute_state()

    def step(self,action):
        self.direction = nxt_direction(action)
        next_step = self.connected_index(*tuple_sum(self.body[0],directions[action]))
        reward,done = self.take_action(*next_step)
        state = self.compute_state()

        return state,reward,done         

    def take_action(self,i,j):
        self.move_count+=1
        reward = 0
        growth = False
        done = False
        
        if self.state[i][j] == 1 or self.state[i][j] == -1 or (self.move_count >= self.max_moves):
            reward = -10
            done = True
            
        else:
            if self.state[i][j]>1:
                self.state[i][j], power= 0,self.state[i][j]
                growth = (power == 10)
                reward += (10*power)
                self.fruits.remove((i,j))
                self.score += 1

        poison = 0
        for x,y in self.fruits: poison += self.state[x][y]
        reward -= poison/10
        
        
        if not done: self.__move_to(i,j,growth)
        if len(self.fruits)<self.avg: self.random_fruit_spawn(3)
        
        return reward, done

    def __move_to(self,i,j,grow):
        tail = None
        k = (i,j)
        if self.no_down_synd and grow and self.size<self.avg: 
            self.body.insert(0,k)
            self.size +=1
        else:
            tail = k
            for x in range(self.size):
                tail,self.body[x] = self.body[x],tail   
        
        neck = self.body[1] if self.size > 1 else None 
        
        self.state[i][j] = -2
        if tail: self.state[tail[0]][tail[1]] = 0
        if neck: self.state[neck[0]][neck[1]] = -1


    def __con_walls(self,walls):
          
        while walls>0:
            i,j = self.index(np.random.randint(self.CELLS))
            if self.state[i][j] == 0:
                self.state[i][j] = 1
                walls-=1
        
    def connected_index(self,i,j):
        return i%self.X,j%self.Y

    def __pos_dir_count(self,i,j):
        out = 0
        for x,y in directions:
            x,y = self.connected_index(i+x,j+y)
            if not self.state[x][y]!=0: out+=1
        return out 

    def give_free_cell(self,lim=3,k=5):
        
        cell = None
        while k>0:
            x,y = self.index(np.random.randint(self.CELLS))
            if self.state[x][y] == 0 and self.__pos_dir_count(x,y)>=lim: 
                cell = (x,y)
                k = 0
            else: k-=1

        return cell 

    def random_fruit_spawn(self,k):
        cell = self.give_free_cell(1,k)
        if not cell: return False 
        
        self.add_fruit(*cell, 5 if np.random.random() > 0.3 else 10)
        return True


    def add_fruit(self,i,j,power):
        if self.state[i][j]!=0: return False
        self.state[i][j] = power
        self.fruits.add((i,j))
        return True 


    def compute_state(self):
        l,f,r,b = directions
        head = self.body[0]
        # danger & food within the window
        wd_l,wf_l = self.__see_window_straight(head,l,(0,0),self.window)
        wd_f,wf_f = self.__see_window_straight(head,f,(0,0),self.window)
        wd_r ,wf_r = self.__see_window_straight(head,r,(0,0),self.window)
        wd_b ,wf_b = self.__see_window_straight(head,b,(0,0),self.window)
        
        # smell of food over the env
        s_fl,s_fr,s_bl,s_br = 0,0,0,0
        for fruit in self.fruits:
            s_fl += self.__smell_fruit(fruit,f,l)
            s_fr += self.__smell_fruit(fruit,f,r)
            s_bl += self.__smell_fruit(fruit,b,l)
            s_br += self.__smell_fruit(fruit,b,r)


        state = np.array([
            # directions
            0,0,0,0, 
            # danger
            wd_l,wd_f,wd_r, # front
            wd_b, # rear
            # food
            wf_l,wf_f,wf_r, # front
            wf_b, # rear
            # smell
            s_fl,s_fr,s_br,s_bl 
            ])
        state[self.direction] = 1
        return state

    def __smell_fruit(self,fruit,side1,side2):
        sx,sy = tuple_sum(side1,side2)
        i,j = self.body[0]
        x,y = fruit
        if sx > 0 and x<i: x = self.X+x
        if sx < 0 and x>i: x = (self.X-x) * -1
        if sy > 0 and y<j: y = self.Y+y
        if sy < 0 and y>j: y = (self.Y-y) * -1

        return self.state[fruit[0]][fruit[1]] * gaussian_kernel((i,j),(x,y)) * 10 # scaling factor 

    def __see_window_straight(self,curr,step,off,k):
        if k == 0: return (0,0)
        
        nxt_i,nxt_j = tuple_sum(curr,step)
        c_nxt_i,c_nxt_j = self.connected_index(nxt_i,nxt_j)
        off = (off[0] if nxt_i == c_nxt_i else curr[0]+1,off[1] if nxt_j == c_nxt_j else curr[1]+1)
        obst = self.state[c_nxt_i][c_nxt_j]
        curr = (c_nxt_i,c_nxt_j)
        
        if obst == 0: return self.__see_window_straight(curr,step,off,k-1)

        if obst > 1: return (0,euclidean(self.body[0],tuple_sum(curr,off))) # Food (0,+)

        return (euclidean(self.body[0],tuple_sum(curr,off)),0) # Danger (+,0)
        
        
    def __str__(self):
        out = ""
        for x in self.state:
            for i in x:
                out += color_map[i]
            out+="\n"
        out += f"Score: {self.score} Moves: {self.move_count} Direction:{self.direction}"
        return out
    
    def render(self,print_now=True):
        frame = str(self)
        if self.recording: self.frames.append(frame)
        if print_now: print(frame)
        
    def record(self,on=False):
        if on:
            self.recording = on
            self.frames = []
        else:
            self.recording = False
            return self.frames

        
        
def play_frames(frames,clear_function=lambda : print("\033[H\033[J"),slow=0.1):
    for frame in frames:
        clear_function()
        print(frame)
        sleep(slow)

        
if __name__ == '__main__':
    np.random.seed(69420)
    from nn import NeuralNetwork
    
   
    def play(net,env,record=False,print_now=True,slow=0.1):
        print("\033[H\033[J")
        nxt_state = env.reset(walls=0.2,nds=False)
        done = False
        if record: env.record(True)
        env.render(print_now)
        while not done:
            sleep(slow)
            state = nxt_state 
            action = np.argmax(net.predict_single(state))
            nxt_state,r,done = env.step(action)
            print("\033[H\033[J")
            env.render(print_now)
        if record:
            return env.record()
    
    agent = NeuralNetwork([16,4],["linear"])

    env = Environment(10,10,max_moves=50)
    frames = play(agent,env,True,False,0)
    
    play_frames(frames,slow=0.2)
