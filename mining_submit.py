# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 22:57:53 2021

@author: OptiplexUser
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:56:47 2021

@author: frederic

    
class problem with     

An open-pit mine is a grid represented with a 2D or 3D numpy array. 

The first coordinates are surface locations.

In the 2D case, the coordinates are (x,z).
In the 3D case, the coordinates are (x,y,z).
The last coordinate 'z' points down.

    
A state indicates for each surface location  how many cells 
have been dug in this pit column.

For a 3D mine, a surface location is represented with a tuple (x,y).

For a 2D mine, a surface location is represented with a tuple (x,).


Two surface cells are neighbours if they share a common border point.
That is, for a 3D mine, a surface cell has 8 surface neighbours.


An action is represented by the surface location where the dig takes place.


"""
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import itertools

import functools # @lru_cache(maxsize=32)

from numbers import Number

import search

def my_team():    

     '''    Return the list of the team members of this assignment submission 
     as a list    of triplet of the form (student_number, first_name, last_name)        ''' 

     return [ (10448888, 'Angus', 'Macdonald'), (10689753, 'Cassandra', 'Nolan'), (10112375, 'Riku', 'Oya') ]
    
def convert_to_tuple(a):
    '''
    Convert the parameter 'a' into a nested tuple of the same shape as 'a'.
    
    The parameter 'a' must be array-like. That is, its elements are indexed.

    Parameters
    ----------
    a : flat array or an array of arrays

    Returns
    -------
    the conversion of 'a' into a tuple or a tuple of tuples

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return ()
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return tuple(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return tuple(tuple(r) for r in a)
    
    
def convert_to_list(a):
    '''
    Convert the array-like parameter 'a' into a nested list of the same 
    shape as 'a'.

    Parameters
    ----------
    a : flat array or array of arrays

    Returns
    -------
    the conversion of 'a' into a list or a list of lists

    '''
    if isinstance(a, Number):
        return a
    if len(a)==0:
        return []
    # 'a' is non empty tuple
    if isinstance(a[0], Number):
        # 'a' is a flat list
        return list(a)
    else:
        # 'a' must be a nested list with 2 levels (a matrix)
        return [list(r) for r in a]    




class Mine(search.Problem):
    '''
    
    Mine represent an open mine problem defined by a grid of cells 
    of various values. The grid is called 'underground'. It can be
    a 2D or 3D array.
    
    The z direction is pointing down, the x and y directions are surface
    directions.
    
    An instance of a Mine is characterized by 
    - self.underground : the ndarray that contains the values of the grid cells
    - self.dig_tolerance : the maximum depth difference allowed between 
                           adjacent columns 
    
    Other attributes:
        self.len_x, self.len_y, self.len_z : int : underground.shape
        self.cumsum_mine : float array : cumulative sums of the columns of the 
                                         mine
    
    A state has the same dimension as the surface of the mine.
    If the mine is 2D, the state is 1D.
    If the mine is 3D, the state is 2D.
    
    state[loc] is zero if digging has not started at location loc.
    More generally, state[loc] is the z-index of the first cell that has
    not been dug in column loc. This number is also the number of cells that
    have been dugged in the column.
    
    States must be tuple-based.
    
    '''    
    
    def __init__(self, underground, dig_tolerance = 1):
        '''
        Constructor
        
        Initialize the attributes
        self.underground, self.dig_tolerance, self.len_x, self.len_y, self.len_z,
        self.cumsum_mine, and self.initial
        
        The state self.initial is a filled with zeros.

        Parameters
        ----------
        underground : np.array
            2D or 3D. Each element of the array contains 
            the profit value of the corresponding cell.
        dig_tolerance : int
             Mine attribute (see class header comment)
        Returns
        -------
        None.

        '''
        # super().__init__() # call to parent class constructor not needed
        
        self.underground = underground 
        # self.underground  should be considered as a 'read-only' variable!
        self.dig_tolerance = dig_tolerance
        assert underground.ndim in (2,3)
        
        if(underground.ndim == 2): #If the mine is 2D
            self.len_x = underground.shape[0] #The length of x is the first dimension of the mine
            self.len_z = underground.shape[1] #The length of z is the second dimension of the mine
            self.initial = np.zeros((self.len_x)) #The initial state is an array the length of x filled with 0's
            self.cumsum_mine = np.cumsum(underground, axis=1) #The cumilative sum along the Z axis
        
        elif(underground.ndim == 3): #If mine is 3D
            self.len_x = underground.shape[0] #The length of x is the first dimension of the mine
            self.len_y = underground.shape[1] #The length of y is the second dimension of the mine
            self.len_z = underground.shape[2] #The length of z is the third dimension of the mine
            self.initial = np.zeros((self.len_x ,self.len_y), dtype = int) #Initial state is a 2D state of x, y size
            self.cumsum_mine = np.cumsum(underground, axis = 2) #Cumilative sum along the Z axis
        
        
    def surface_neigbhours(self, loc):
        '''
        Return the list of neighbours of loc

        Parameters
        ----------
        loc : surface coordinates of a cell.
            a singleton (x,) in case of a 2D mine
            a pair (x,y) in case of a 3D mine
        Returns
        -------
        A list of tuples representing the surface coordinates of the
        neighbouring surface cells.

        '''
        L=[]
        assert len(loc) in (1,2)
        if len(loc)==1:
            if loc[0]-1>=0:
                L.append((loc[0]-1,))
            if loc[0]+1<self.len_x:
                L.append((loc[0]+1,))
        else:
            # len(loc) == 2
            for dx,dy in ((-1,-1),(-1,0),(-1,+1),
                          (0,-1),(0,+1),
                          (+1,-1),(+1,0),(+1,+1)):
                if  (0 <= loc[0]+dx < self.len_x) and (0 <= loc[1]+dy < self.len_y):
                    L.append((loc[0]+dx, loc[1]+dy))
        return L
     
    def actions(self, state):
        '''
        Return a generator of valid actions in the give state 'state'
        An action is represented as a location. An action is deemed valid if
        it doesn't  break the dig_tolerance constraint.

        Parameters
        ----------
        state : 
            represented with nested lists, tuples or a ndarray
            state of the partially dug mine

        Returns
        -------
        a generator of valid actions
        # '''
        
        #Convert the state into np array
        s = np.array(state) 
        
        #Make sure the state is 1D for 2D mine, or 2D state for a 3D mine as we have no Z values
        assert s.ndim in (1,2) 
        
        #Finds the total amount of elements within a state 
        total_e = np.size(state)
        
        #Creates a list of state copies in the amount of values within the state
        possible_states = list(state[:] for i in range(total_e))
        #Converts this list back into an array
        possible_states = np.array(possible_states)
        
        #If the state is 1D
        if(np.ndim(state) == 1): 
            #For each state within the list of state copies
            for i in range(len(possible_states)):
                #It adds one to the next index in the next state
                possible_states[i][i] += 1; 
                
            #Converts the possible actions back into a list
            possible_states = possible_states.tolist() 
            
            #For every state within possible states, creates a new list of indeces where they don't break 
            #the dig_tolerance OR dig deeper than the mine, the length of Z (via the is_dangerous function).
            safe_states = list((possible_states.index(j),) for j in possible_states if not self.is_dangerous(j))
            
            #Converts the safe states back to a tuple
            safe_states = convert_to_tuple(safe_states);
            
            #And returns
            return safe_states
        
        
        #If the state is 2 dimensional
        if(np.ndim(state) == 2):
            #For every element in state, increments the J value from 0 to x 
            #(In a 3,4 dimensional state mine it would increment 0 to 11).
            for j in (range(np.size(possible_states[0]))):
                #Adds 1 to the next element, in the next state copy using the floor and modulus of the increment
                #And our y-axis length
                possible_states[j][j//self.len_y][j% self.len_y] += 1
            
            #Converts this array of possible states into a list
            possible_states = possible_states.tolist()
            
            #Creates a list of safe_states indeces by checking the possible_states and removing where it is dangerous
            #Or exceeds the depth of the mine, by the is_dangerous function
            safe_states = list((possible_states.index(j) // self.len_y, possible_states.index(j) % self.len_y) for j in possible_states if not self.is_dangerous(j))
            
            #Returns the safe states
            return safe_states

        
        
  
    def result(self, state, action): ##NEEDS (x,) or (x,y)
        """Return the state that results from executing the given
        action in the given state. The action must a valid actions.
        That is, one of those generated by  self.actions(state)."""
        action = tuple(action) #Make sure processing as tuple if it were list
        new_state = np.array(state) # Make a copy
        new_state[action] += 1
        return convert_to_tuple(new_state)
                
    
    def console_display(self):
        '''
        Display the mine on the console

        Returns
        -------
        None.

        '''
        print('Mine of depth {}'.format(self.len_z))
        if self.underground.ndim == 2:
            # 2D mine
            print('Plane x,z view')
        else:
            # 3D mine
            print('Level by level x,y slices')
        #
        print(self.__str__())
        
    def __str__(self):
        if self.underground.ndim == 2:
            # 2D mine
            return str(self.underground.T)
        else:
            # 3D mine
            # level by level representation
            return '\n'.join('level {}\n'.format(z)
                   +str(self.underground[..., z]) for z in range(self.len_z))
                    
                        
                
            #return self.underground[loc[0], loc[1],:]
        
    
    @staticmethod   
    def plot_state(state):
        if state.ndim==1:
            fig, ax = plt.subplots()
            ax.bar(np.arange(state.shape[0]) ,
                    state
                    )
            ax.set_xlabel('x')
            ax.set_ylabel('z')
        else:
            assert state.ndim==2
            # bar3d(x, y, z, dx, dy, dz,
            # fake data
            _x = np.arange(state.shape[0])
            _y = np.arange(state.shape[1])
            _yy, _xx = np.meshgrid(_y, _x) # cols, rows
            x, y = _xx.ravel(), _yy.ravel()            
            top = state.ravel()
            bottom = np.zeros_like(top)
            width = depth = 1
            fig = plt.figure(figsize=(3,3))
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            ax1.set_title('State')
        #
        plt.show()

    def payoff(self, state):
        '''
        Compute and return the payoff for the given state.
        That is, the sum of the values of all the digged cells.
        
        No loops needed in the implementation!        
        '''
        #Converting state parameter into numpy
        npState = np.array(state)
        
        #Make sure our state dimensions are correct
        assert npState.ndim in (1,2)
        
        #If state has any value > 0 isMined is True
        isMined = np.any(npState) 
        
        #Initialising the payoff, starting at 0.
        payoff = 0.00; 
        
        #Initialising an array of values that have been extracted from mined state
        minedArray = [] 
        
        if not(isMined): #If the isMined value is false (State has no values besides 0)
            return 0; #Payoff is 0 (Nothing has happened)
        
        elif(isMined): #Else if the state does contain any values other than 0
            if(self.underground.ndim == 2): #If the mine is 2D
                for i in range(np.size(npState)): #For the each value in state
                    if npState[i] > 0: #If it is mined
                        minedArray.append(self.cumsum_mine[i][int(npState[i])-1])  #Append the value found in the mine at the depth of the state value
                payoff = sum(minedArray) #Summate all the values
                return payoff #And Return
                
            if(self.underground.ndim == 3): #If the mine is 3D
                for i in range(np.size(npState)): #In the range of all elements within the state
                    x = i // self.len_y; #Finds the x value by divding the i value by the length of y and flooring
                    y = i % self.len_y; #Finds the y value by mod the i value by the length of y
                    z = npState[x][y] #Finds the level at which the blocks have been mined by accessing the x,y of the state
                    if z > 0: #If the value is larger than 0 and is mined
                        minedArray.append(self.cumsum_mine[x][y][z - 1]) #Access the cumsum value of that block in x,y,z and append
                payoff = sum(minedArray); #Summate all these values
                return payoff; #Return
    
    def is_dangerous(self, state):
        '''
        Return True iff the given state breaches the dig_tolerance constraints.
        
        No loops needed in the implementation!
        '''
        #Converts the state into a numpy
        state = np.array(state)   
        
        #Dimensions checks
        assert state.ndim in (1,2)
        
        #If we have a 1D state
        if(np.ndim(state) == 1):
            #If any of the values in that state exceed the depth of the mine
            if np.any(state > self.len_z):
                #It returns true, this functionality is not a requirement for the function
                #but allowed us to remove recursive code to check in our actions function
                return True;
            
            #If the value in the state doesn't exceed the depth of the mine
            else:
                #Enumerates throughout the state to find the differences between neighbouring columns
                differences_between_c = np.array([abs(x - state[i - 1]) for i, x in enumerate(state) if i > 0])
                
                #Returns true if any of the values exceed the dig tolerance
                #This line of code was based on code found at 
                #https://stackoverflow.com/questions/20229822/check-if-all-values-in-list-are-greater-than-a-certain-number
                #Written by user 0.Rka on the 26th of November, 2013. 
                return len([*filter(lambda x: x > self.dig_tolerance, differences_between_c)]) > 0
        
        #If we have a 2D state
        if(np.ndim(state) == 2):
            
            #Flatten the state to a temporary variable
            temp_c = state.flatten()
            
            #If any of the values in the state exceed the length of Z, return as dangerous
            #Much like before, this functionality is not neccessary of the function
            #Follow reference in 1D state for the below line of code.
            if (len([*filter(lambda x: x > self.len_z, temp_c)]) > 0):
                return True
            
            #If the state doesn't exceed the length of Z
            else:
                #Increment along the X-axis
                for i in range(len(state)):
                    #Incremenet along the Y-axis
                    for j in range(len(state[0])):
                        #For every neighbour of the current (x,y) within out state 
                        for k in self.surface_neigbhours((i , j)):
                            #If the difference between them does not exceed the dig_tolerance, return true
                            if abs(state[k] - state[i,j]) > self.dig_tolerance:
                               return True;
                #Else return false, the state is safe
                return False;
            
    
    # ========================  Class Mine  ==================================
    
    

def search_dp_dig_plan(mine):
    '''
    Search using Dynamic Programming the most profitable sequence of 
    digging actions from the initial state of the mine.
    
    Return the sequence of actions, the final state and the payoff
    

    Parameters
    ----------
    mine : a Mine instance

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    
    '''
        This function applies the recursive function to create every possible state of a mine, with no duplicates,
        and if the payoff of a state is greater than the previous best_payoff, it updates that state as the new
        best_payoff and best_state
    '''
    
    #Console Printing for the markers to know our function is running
    #For a 2D mine, runs approx 2.5 times slower than sanity_tests, for 3D 6.5 times slower
    print("    ")
    print("    ")
    print("Currently, this function is operating up to roughly 6.5 times slower than the sanity_check tests...")
    print("For a (3, 4, 5) dimensional 3D Mine, it will execute in approx 40 seconds")
    print("Please wait ...")
    print("    ")
    print("    ")
    print("    ")
            
    #Initialising the starting state of the mine into a variable
    start_state = mine.initial
    
    #Creating a set to store the states that we visit
    explored = set()
    
    #Our recursive function 
    def recursive(state):
        #Takes the input state, and stores a hash copy in our explored
        hash_state = hash(convert_to_tuple(state))
        
        #Finds the payoff of the current processing state
        best_payoff = mine.payoff(state)
        
        #Storing the current state within our best_final variable
        best_final = state
        
        #If our hash state hasn't been processed yet
        if hash_state not in explored:
            
            #For all the possible actions
            for action in mine.actions(state):
                #The state after the action, the a child_state of our input state
                child_state = mine.result(state, action)
                
                #Call our recursive function again, inputting the new child_state for processing
                #Which wil return the best_payoff, best_state of the children of the input state
                child_payoff, child_final = recursive(child_state)
                
                #If the returned child_payoff is greater than the current best_payoff
                if child_payoff > best_payoff:
                    #The best_payoff is the child_payoff
                    best_payoff = child_payoff
                    #The best_final is our child_final
                    best_final = child_final
            #Adds our processed state into our explored set
            explored.add(hash_state)
        #Returns the best_payoff, best_final_state
        return best_payoff, best_final

    #Starts our recursive function with the returning value into the "final" variable
    final = recursive(start_state)
    #Stores the best_payoff from the final into a variable final_payoff
    final_payoff = final[0]
    
    #Stores the best_payoff
    final_state = final[1]
    
    #Finds the action sequence between the initial_state and the final_state
    seq = (find_action_sequence(start_state, final_state))
    
    #Returns the values
    return final_payoff, seq, final_state


    
def search_bb_dig_plan(mine):
    '''
    Compute, using Branch and Bound, the most profitable sequence of 
    digging actions from the initial state of the mine.
        

    Parameters
    ----------
    mine : Mine
        An instance of a Mine problem.

    Returns
    -------
    best_payoff, best_action_list, best_final_state

    '''
    #Console Printing for the markers to know our function is running
    #For a 2D mine, runs in similar time to sanity_tests, for 3D runs roughly 10 times slower
    print("    ")
    print("    ")
    print("Currently, this function is operating up to roughly 10 times slower than the sanity_check tests...")
    print("For a (3, 4, 5) dimensional 3D Mine, it will execute in approx 30 seconds")
    print("Please wait ...")
    print("    ")
    print("    ")
    print("    ")
    
    # initialise best state and payoff
    best_state = mine.initial
    best_payoff = mine.payoff(best_state)   # upper bound, b(s)
    # best_payoff = np.amax(np.cumsum(best_state)) # alternate b(s) with relaxed constraints
    
    frontier = []                       # initialise empty frontier
    frontier.append(best_state)         # add initial state to frontier
    explored = set()                    # init empty array for explored states
    while frontier:                     # while frontier isnt empty
        current_state = frontier.pop()  # remove and return last state added to frontier
        
        # add current state to explored states as hashed tuple
        hash_var = hash(convert_to_tuple(current_state))
        explored.add(hash_var)
        
        # for every not dangerous child state from current_state, 
        for state in mine.actions(current_state):
            this_state = mine.result(current_state, state)
            
            # bool checks for following if stmnt
            b0 = hash(convert_to_tuple(this_state)) not in explored
            b1 = this_state not in frontier
            # attempt at pruning branches but doesn't allow max payoff to be reached
            # b2 = mine.payoff(this_state) >= best_payoff   
            
            # if this child state isnt in explored set
            # and if it isnt already in the frontier,
            if b0 and b1: # and b2:
                # add this child state to the frontier 
                frontier.append(this_state) 
                
                # if the payoff of this child state is greater than the current best payoff,
                if mine.payoff(this_state) > best_payoff:
                    # make best payoff and best state now equal to this child state and this child payoff
                    best_payoff = mine.payoff(this_state)
                    best_state = this_state
                
    # use find_action_sequence to find the action sequence for the best final state
    action_seq = find_action_sequence(mine.initial, best_state)
                
   # return best_payoff, best_action_list, best_final_state
    return best_payoff, action_seq, best_state 
    
    
def find_action_sequence(s0, s1):
    '''
    Compute a sequence of actions to go from state s0 to state s1.
    There may be several possible sequences.
    
    Preconditions: 
        s0 and s1 are legal states, s0<=s1 and 
    
    Parameters
    ----------
    s0 : tuple based mine state
    s1 : tuple based mine state 

    Returns
    -------
    A sequence of actions to go from state s0 to state s1

    '''    
    
    '''
        This function is finding the largest difference between coresponding indexes between the 2 states,
        it is then updating a copied state of the start_state with the action and then comparing it to the goal state
        within the while function.
        
        The function checks if the action of the largest difference between 2 indexes was previously completed
        on the same index, and applies a small workaround to find a different new index to change to copied version.
        This is to help in issues with dig_tolerance.
        
        If the open_pit_mine problem had a cost for moving between columns, 
        for example cost = 1 to move between C1 and C2, this function would be highly invaluable due to the
        moving to a block at the opposite end with the dig_tolerance workaround.
        
        We are happy there are no movement costs :)
        
        
    '''
    
    #Assert that both parameters are of equal value
    assert len(s0) == len(s1)
    
    #Convert our parameters into numpy arrays
    s0 = np.array(s0)
    s1 = np.array(s1)
    
    #Assert we have 1D or 2D state
    assert s0.ndim and s1.ndim in (1,2)
    
    #Create a copy of our starting state
    s = s0[:]
    
    #Initialising an array to store out actions
    action_seq = []
    
    #If our state is 1D
    if np.ndim(s0) == 1:
        
        #If our copy starting state isn't the same as out goal state
        while convert_to_list(s1) != convert_to_list(s):
            
            #Find the difference between each element in both states
            diff = convert_to_list(s1 - s);
            
            #Find the index of the maximum value
            ind = diff.index(max(diff))
            
            #If we have run the function before
            if len(action_seq) != 0:
                
                #If the index isn't of the same value as previous index in action_seq
                if ind != action_seq[len(action_seq) - 1]:
                    #Add 1 to our copied state
                    s[ind] += 1;
                    #Append the index into the action_sequence
                    action_seq.append(ind)
                #If the index does match the last index of action_seq
                else:
                    #Reverse the array of differences
                    flip = diff[::-1]
                    #Find the max value index from the reversed array
                    ind = flip.index(max(flip))
                    #The previous index in comparison to our original index
                    second_Max = len(diff) - ind - 1
                    #Adds 1 to our copied initial state
                    s[second_Max] += 1;
                    #Adds the action to the sequences
                    action_seq.append(second_Max)
            #If it is our first time through the while loop
            else:
                #Uses the index to add 1 to our copied state
                s[ind] += 1;
                #Adds the action to the list
                action_seq.append(ind)
    
        #Returns the actions in a lsit
        action_seq = convert_to_tuple(action_seq)
        return action_seq
    
    #If our state is 2
    if np.ndim(s0) == 2:
        #Checks to see differences between the goal and copied state
        while convert_to_list(s1) != convert_to_list(s):
            #Initialise an array for the differences between the elements in both states
            diff = []
            #This nested loop iterates through both nested array states and find the differences between the lements
            for i in range(len(s)):
                for j in range(len(s[0])):
                    diff.append(s1[i][j] - s[i][j])
            #Finds the index of the max value of the differences in our differences
            ind = diff.index(max(diff))
            #Finds the x,y value like before, using floor and mod by the y length
            x = ind//(len(s[0]))
            y = ind % (len(s[0]))
            
            #If we have done 1 pass of the while loop
            if len(action_seq) != 0:
                #If the action hasn't been done before
                if (x,y) not in action_seq:
                    #Gets the x,y values again (Repeated for assurance)
                    x = ind//(len(s[0]))
                    y = ind % (len(s[0]))
                    #Adds 1 to the copied state of our x,y action 
                    s[x][y] += 1;
                    #Adds this action to the sequence
                    action_seq.append((x,y))
                #If the action IS in the previous actions
                else:
                    #Much like before, it flips the array and finds a different max value from the other side
                    flip = diff[::-1]
                    ind = flip.index(max(flip))
                    second_Max = len(diff) - ind - 1
                    x = second_Max // (len(s[0]))
                    y = second_Max % (len(s[0]))
                    #And adds 1 to that x,y element in comparison to our copied state
                    s[x][y] += 1;
                    #And appends to the list
                    action_seq.append((x,y))
            #If it's our first pass of the while loops
            else:
                #Like before, finds the values again for reassurance
                 x = ind//(len(s[0]))
                 y = ind % (len(s[0]))
                 #Adds one to the x,y
                 s[x][y] += 1;
                 #And adds the action to the sequence
                 action_seq.append((x,y))
        #Converting our sequence list to a tuple
        action_seq = convert_to_tuple(action_seq)
        #And returns
        return action_seq
        
        
        
    
        
        
        
        
        
    
    
    
