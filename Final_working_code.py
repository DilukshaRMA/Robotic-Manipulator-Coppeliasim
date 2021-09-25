import multiprocessing
import sim
import numpy as np
import time

flag_places = [[0, 0, 'Y'], [0, 1, 'Y'], [0, 2, 'Y'], [0, 3, 'Y'], [0, 4, 'Y'], [0, 5, 'Y'], [0, 6, 'Y'], [0, 7, 'Y'], [0, 8, 'Y'], 
[0, 9, 'Y'], [0, 10, 'Y'], [0, 11, 'Y'], [0, 12, 'Y'], [0, 13, 'Y'], [0, 14, 'Y'], [0, 15, 'Y'], [0, 16, 'Y'], [0, 17, 
'Y'], [0, 18, 'Y'], [0, 19, 'Y'], [0, 20, 'Y'], [0, 21, 'Y'], [0, 22, 'Y'], [0, 23, 'Y'], [1, 0, 'Y'], [1, 1, 'R'], [1, 2, 'R'], [1, 3, 'R'], [1, 4, 'R'], [1, 5, 'R'], [1, 6, 'R'], [1, 7, 'R'], [1, 8, 'R'], [1, 9, 'R'], [1, 10, 'R'], [1, 11, 'R'], [1, 12, 'R'], [1, 13, 'R'], [1, 14, 'R'], [1, 15, 'R'], [1, 16, 'Y'], [1, 17, 'O'], [1, 18, 'O'], [1, 19, 'O'], [1, 20, 'G'], [1, 21, 'G'], [1, 22, 'G'], [1, 23, 'Y'], [2, 0, 'Y'], [2, 1, 'R'], [2, 2, 'Y'], [2, 3, 'R'], [2, 
4, 'R'], [2, 5, 'R'], [2, 6, 'R'], [2, 7, 'R'], [2, 8, 'R'], [2, 9, 'R'], [2, 10, 'R'], [2, 11, 'R'], [2, 12, 'R'], [2, 13, 'R'], [2, 14, 'Y'], [2, 15, 'R'], [2, 16, 'Y'], [2, 17, 'O'], [2, 18, 'O'], [2, 19, 'O'], [2, 20, 'G'], [2, 21, 'G'], [2, 22, 'G'], [2, 23, 'Y'], [3, 0, 'Y'], [3, 1, 'R'], [3, 2, 'R'], [3, 3, 'R'], [3, 4, 'Y'], [3, 5, 'Y'], [3, 6, 
'Y'], [3, 7, 'R'], [3, 8, 'R'], [3, 9, 'R'], [3, 10, 'R'], [3, 11, 'R'], [3, 12, 'R'], [3, 13, 'R'], [3, 14, 'R'], [3, 
15, 'R'], [3, 16, 'Y'], [3, 17, 'O'], [3, 18, 'O'], [3, 19, 'O'], [3, 20, 'G'], [3, 21, 'G'], [3, 22, 'G'], [3, 23, 'Y'], [4, 0, 'Y'], [4, 1, 'R'], [4, 2, 'R'], [4, 3, 'R'], [4, 4,'R'], [4, 5,'R'], [4, 6, 'Y'], [4, 7,'R'], [4, 8, 'Y'], [4, 9, 'Y'], [4, 10, 'Y'], [4, 11, 'Y'], [4, 12, 'R'], [4, 13, 'Y'], [4, 14, 'R'], [4, 15, 'R'], [4, 16, 'Y'], [4, 17, 'O'], [4, 18, 'O'], [4, 19, 'O'], [4, 20, 'G'], [4, 21, 'G'], [4, 22, 'G'], [4, 23, 'Y'], [5, 0, 'Y'], [5, 1, 'R'], [5, 2, 'R'], 
[5, 3, 'R'], [5, 4, 'Y'], [5, 5, 'Y'], [5, 6, 'Y'], [5, 7,'R'], [5, 8,'Y'], [5, 9, 'Y'], [5, 10, 'Y'], [5, 11,'R'], [5, 12, 'R'], [5, 13, 'Y'], [5, 14, 'R'], [5, 15, 'R'], [5, 16, 'Y'], [5, 17, 'O'], [5, 18, 'O'], [5, 19, 'O'], [5, 20, 'G'], [5, 21, 
'G'], [5, 22, 'G'], [5, 23, 'Y'], [6, 0, 'Y'], [6, 1, 'R'], [6, 2, 'R'], [6, 3, 'Y'], [6, 4, 'R'], [6, 5, 'R'], [6, 6, 'R'], [6, 7, 'R'], [6, 8, 'Y'], [6, 9, 'Y'], [6, 10, 'Y'], [6, 11, 'Y'], [6, 12, 'R'], [6, 13, 'Y'], [6, 14, 'R'], [6, 15, 'R'], [6, 16, 'Y'], [6, 17, 'O'], [6, 18, 'O'], [6, 19, 'O'], [6, 20, 'G'], [6, 21, 'G'], [6, 22, 'G'], [6, 23, 'Y'], [7, 0, 'Y'], [7, 1, 'R'], [7, 2, 'R'], [7, 3, 'Y'], [7, 4, 'Y'], [7, 5, 'Y'], [7, 6, 'Y'], [7, 7, 'Y'], [7, 8, 'Y'], [7, 9, 'Y'], [7, 10, 'Y'], [7, 11, 'R'], [7, 12, 'R'], [7, 13, 'Y'], [7, 14, 'R'], [7, 15, 'R'], [7, 16, 'Y'], [7, 17, 'O'], [7, 18, 'O'], [7, 19, 'O'], [7, 20, 'G'], [7, 21, 'G'], [7, 22, 'G'], [7, 23, 'Y'], [8, 0, 'Y'], [8, 1, 'R'], [8, 2, 'R'], [8, 3, 'Y'], [8, 4, 'Y'], [8, 5, 'Y'], [8, 6, 'Y'], [8, 7, 'Y'], [8, 8, 'Y'], [8, 9, 'Y'], [8, 10, 'Y'], [8, 11, 'Y'], [8, 12,'R'], [8, 13, 'Y'], [8, 14, 'R'], [8, 15, 'R'], [8, 16, 'Y'], [8, 17, 'O'], [8, 18, 'O'], [8, 19, 'O'], [8, 20, 'G'], [8, 21, 'G'], [8, 22, 'G'], [8, 23, 'Y'], [9, 0, 'Y'], [9, 1, 'R'], [9, 2, 'R'], [9, 3, 'Y'], [9, 
4, 'Y'], [9, 5, 'Y'], [9, 6, 'Y'], [9, 7, 'Y'], [9, 8, 'Y'], [9, 9, 'Y'], [9, 10, 'Y'], [9, 11, 'Y'], [9, 12, 'Y'], [9, 13, 'Y'], [9, 14, 'R'], [9, 15, 'R'], [9, 16, 'Y'], [9, 17, 'O'], [9, 18, 'O'], [9, 19, 'O'], [9, 20, 'G'], [9, 21, 'G'], [9, 22, 'G'], [9, 23, 'Y'], [10, 0, 'Y'], [10, 1, 'R'], [10, 2, 'R'], [10, 3, 'R'], [10, 4, 'Y'], [10, 5, 'Y'], [10, 6, 'Y'], [10, 7, 'R'], [10, 8, 'R'], [10, 9, 'Y'], [10, 10, 'Y'], [10, 11, 'Y'], [10, 12,'R'], [10, 13, 'Y'], [10, 14, 
'R'], [10, 15, 'R'], [10, 16, 'Y'], [10, 17, 'O'], [10, 18, 'O'], [10, 19, 'O'], [10, 20, 'G'], [10, 21, 'G'], [10, 22, 'G'], [10, 23, 'Y'], [11, 0, 'Y'], [11, 1, 'R'], [11, 2, 'R'], [11, 3, 'Y'], [11, 4, 'Y'], [11, 5, 'Y'], [11, 6, 'R'], [11, 7, 'R'], [11, 8, 'R'], [11, 9, 'Y'], [11, 10, 'R'], [11, 11, 'R'], [11, 12, 'R'], [11, 13, 'Y'], [11, 14, 'R'], [11, 15, 'R'], [11, 16, 'Y'], [11, 17, 'O'], [11, 18, 'O'], [11, 19, 'O'], [11, 20, 'G'], [11, 21, 'G'], [11, 22, 'G'], [11, 23, 'Y'], [12, 0, 'Y'], [12, 1, 'R'], [12, 2, 'R'], [12, 3, 'R'], [12, 4, 'Y'], [12, 5, 'Y'], [12, 6, 'Y'], [12, 7, 'R'], [12, 8, 'R'], [12, 9, 'Y'], [12, 10, 'Y'], [12, 11, 'R'], [12, 12, 'R'], [12, 13, 'R'], [12, 14, 'R'], [12, 15, 'R'], [12, 16, 'Y'], [12, 17, 'O'], [12, 18, 'O'], [12, 19, 'O'], [12, 20, 'G'], [12, 21, 'G'], [12, 22, 'G'], [12, 23, 'Y'], [13, 0, 'Y'], [13, 1, 'R'], [13, 2, 'Y'], [13, 3, 'R'], [13, 4, 'R'], [13, 5, 'R'], [13, 6, 'R'], [13, 7, 'R'], [13, 8, 'R'], [13, 9, 'R'], [13, 10, 'R'], [13, 11, 'R'], [13, 12, 'R'], [13, 13, 'R'], [13, 14, 'Y'], [13, 15, 
'R'], [13, 16, 'Y'], [13, 17, 'O'], [13, 18, 'O'], [13, 19, 'O'], [13, 20, 'G'], [13, 21, 'G'], [13, 22, 'G'], [13, 23, 'Y'],[14, 0, 'Y'], [14, 1, 'R'], [14, 2, 'R'], [14, 3, 'R'], [14, 4, 'R'], [14, 5, 'R'], [14, 6, 'R'], [14, 7, 'R'], [14, 8, 'R'], [14, 9, 'R'], [14, 10, 'R'], [14, 11, 'R'], [14, 12, 'R'], [14, 13, 'R'], [14, 14, 'R'], [14, 15, 'R'], [14, 16, 'Y'], [14, 17, 'O'], [14, 18, 'O'], [14, 19, 'O'], [14, 20, 'G'], [14, 21, 'G'], [14, 22, 'G'], [14, 23, 'Y'], [15, 0, 'Y'], [15, 1, 'Y'], [15, 2, 'Y'], [15, 3, 'Y'], [15, 4, 'Y'], [15, 5, 'Y'], [15, 6, 'Y'], [15, 7, 'Y'], [15, 8, 'Y'], [15, 9, 'Y'], [15, 10, 'Y'], [15, 11, 'Y'], [15, 12, 'Y'], [15, 13, 'Y'], [15, 14, 'Y'], [15, 15, 'Y'], [15, 16, 'Y'], [15, 17, 'Y'], [15, 18, 'Y'], [15, 19, 'Y'], [15, 20, 'Y'], [15, 21, 'Y'], [15, 22, 'Y'], [15, 23, 'Y']] 

#made connectiom
sim.simxFinish(-1)
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)
if clientID!=-1:
    print("Connected to remote API server")
else:
    print("Connection Unsuccussfull")

#initial declarations
red_in_bas,green_in_bas,yellow_in_bas,orange_in_bas = [],[],[],[] # names of cubes currently in baskets
placed_flag=[]
r,g,y,o=0,0,0,0

#functions
def gripper(closing): #to open and close the gripper send 1 to close
    r,j1=sim.simxGetObjectHandle(clientID,'BaxterGripper_closeJoint',sim.simx_opmode_oneshot_wait)
    if(closing==1):
        sim.simxSetJointTargetVelocity(clientID,j1,0.5,sim.simx_opmode_blocking)
    else:
        sim.simxSetJointTargetVelocity(clientID,j1,-0.5,sim.simx_opmode_blocking)
def MoveLinear(handle,init_pos,end_pos,points): #move between two points in straightline
    x1,x2,y1,y2,z1,z2 = init_pos[0],end_pos[0],init_pos[1],end_pos[1],init_pos[2],end_pos[2]
    delta_x = (x2-x1)/points
    delta_y = (y2-y1)/points
    delta_z= (z2-z1)/points
    for steps in range(points):
        time.sleep(0.0001)
        sim.simxSetObjectPosition(clientID,handle,-1,[x1+steps*delta_x,y1+steps*delta_y,z1+steps*delta_z],sim.simx_opmode_oneshot)
    #sim.simxSetObjectPosition(clientID,handle,-1,end_pos,sim.simx_opmode_oneshot)
def getHandle(name): # get the handlle of object
    r1,handle= sim.simxGetObjectHandle(clientID,name,sim.simx_opmode_blocking)
    return handle
def getPosition(handle):# get pos as [x,y,z]
    sim.simxGetObjectPosition(clientID,handle,-1,sim.simx_opmode_streaming)
    sim.simxSynchronousTrigger(clientID)
    returnCode,pos=sim.simxGetObjectPosition(clientID,handle,-1,sim.simx_opmode_streaming)
    return pos
def getOrientation(handle): # get orientation of object as [alpa,geta, gamma]
    sim.simxGetObjectOrientation(clientID,handle,-1,sim.simx_opmode_streaming)
    sim.simxSynchronousTrigger(clientID)
    returnCode,orientation=sim.simxGetObjectOrientation(clientID,handle,-1,sim.simx_opmode_streaming)
    return orientation
def setPosition(handle,position): # set position to [x,y,z]
    sim.simxSetObjectPosition(clientID,handle,-1,position,sim.simx_opmode_oneshot)
def setOrientation(handle,orientation): #get orientation to [alpa,geta, gamma]
    sim.simxSetObjectOrientation(clientID,handle,-1,orientation,sim.simx_opmode_oneshot)
def MovePath(handle,init_pos,init_ori,end_pos,end_ori,cube_size,points):
    i=0
    for j in range(points):
        i=i+0.01
        time.sleep(0.0001)
        sim.simxSetObjectPosition(clientID,handle,-1,[init_pos[0],init_pos[1],init_pos[2]+i],sim.simx_opmode_oneshot)
    x1,x2,y1,y2 = init_pos[0],end_pos[0],init_pos[1],end_pos[1]
    m = (y2-y1)/(x2-x1)
    delta_x = (x2-x1)
    for steps in range(int(abs(delta_x)*100)):
        time.sleep(0.0001)
        sim.simxSetObjectPosition(clientID,handle,-1,[x1+steps*0.01,y1+m*(x1+steps*0.01),end_pos[2]+1],sim.simx_opmode_oneshot)
    sim.simxSetObjectPosition(clientID,handle,-1,[end_pos[0],end_pos[1],end_pos[2]+1],sim.simx_opmode_oneshot)
def getRGB(): # get color array from vision sensor
    sensorHandle = getHandle("Vision_sensor")
    retur,detectionState,auxPackets=sim.simxReadVisionSensor(clientID,sensorHandle,sim.simx_opmode_streaming)
    time.sleep(1) # essential dont remove
    retur,detectionState,auxPackets=sim.simxReadVisionSensor(clientID,sensorHandle,sim.simx_opmode_buffer)
    rgb = [auxPackets[0][1],auxPackets[0][2],auxPackets[0][3]]
    return rgb
def pick_and_place_in_basket(arr_of_box_names):
    target = getHandle("Target") # get handle of target
    attach = getHandle("suctionPadLink")
    red_basket = getHandle("BasketRed") # get handle of baskets 
    green_basket = getHandle("BasketGreen")
    orange_basket = getHandle("BasketOrange")
    yellow_basket = getHandle("BasketYellow")

    #get positions of static objects
    target_init = getPosition(target) #initial position of target at begining of simulation
    red_basket_pos = neibour_pos(red_basket,0.015)
    green_basket_pos = neibour_pos(green_basket,0.015)
    orange_basket_pos = neibour_pos(orange_basket,0.015)
    yellow_basket_pos = neibour_pos(yellow_basket,0.015)

    position_pool = [red_basket_pos,green_basket_pos,orange_basket_pos,yellow_basket_pos]

    arr_of_box_pos=[] #position of boxes 
    for i in arr_of_box_names:
        arr_of_box_pos.append(getPosition(getHandle(i)))
    
    box_count = len(arr_of_box_names) # number of boxes
    
    #color algorithm 
    # index[2]>0.9 & index[1] <0.2 --> Green
    # index[2]>0.15 & index[3] <0.15 --> Red
    # index[2]>0.6 & index[3] <0.2 --> Yellow
    # otherwise --> orange
    r,g,y,o=0,0,0,0
    Z_level_g,Z_level_r,Z_level_y,Z_level_o=0,0,0,0
    for turn in range(box_count):
        xyz  = arr_of_box_pos[turn]
        if turn==0: #first movemet is different
            MoveLinear(target,target_init,[xyz[0],xyz[1],xyz[2]+0.3],100)
            MoveLinear(target,[xyz[0],xyz[1],xyz[2]+0.29],xyz,30)
            box_handle = getHandle(arr_of_box_names[turn])
            sim.simxSetObjectParent(clientID,box_handle,attach,1,sim.simx_opmode_oneshot )
            rgb = getRGB()
        else:
            target_current = getPosition(target) #current pos of target
            MoveLinear(target,target_current,[xyz[0],xyz[1],xyz[2]+0.3],100)
            MoveLinear(target,[xyz[0],xyz[1],xyz[2]+0.29],xyz,30)
            box_handle = getHandle(arr_of_box_names[turn])
            sim.simxSetObjectParent(clientID,box_handle,attach,1,sim.simx_opmode_oneshot )
            rgb = getRGB()
        target_send_pos = []
        if(rgb[1]>0.9 and rgb[0]<0.2):#green
            g_pos = getPosition(green_basket)
            target_send_pos = [g_pos[0],g_pos[1],0.015]
            green_in_bas.append(arr_of_box_names[turn])
            # if g==24:
            #     g=0
            #     Z_level_g+=1
            #     green_basket_pos = neibour_pos(green_basket,0.03*Z_level_g+0.015)
            # else:
            #     g+=1
        elif(rgb[1]<0.15 and rgb[2]<0.15):#red
            red_in_bas.append(arr_of_box_names[turn])
            target_send_pos = red_basket_pos[r]
            if r==24:
                r=0
                Z_level_r+=1
                red_basket_pos = neibour_pos(red_basket,0.03*Z_level_r+0.015)
            else:
                r+=1
        elif(rgb[1]>0.6 and rgb[2]<0.2):#yellow
            yellow_in_bas.append(arr_of_box_names[turn])
            target_send_pos = yellow_basket_pos[y]
            if y==24:
                y=0
                Z_level_y+=1
                yellow_basket_pos = neibour_pos(yellow_basket,0.03*Z_level_y+0.015)
            else:
                y+=1
        else:#orange
            orange_in_bas.append(arr_of_box_names[turn])
            o_pos = getPosition(orange_basket)
            target_send_pos = [o_pos[0],o_pos[1],0.015]
            # if o==24:
            #     o=0
            #     Z_level_o+=1
            #     orange_basket_pos = neibour_pos(orange_basket,0.03*Z_level_o+0.015)
            # else:
            #     o+=1

        target_current = getPosition(target) #current pos of target
        MoveLinear(target,target_current,[target_current[0],target_current[1],target_current[2]+0.29],29)
        MoveLinear(target,[target_current[0],target_current[1],target_current[2]+0.29],[target_send_pos[0],target_send_pos[1],target_send_pos[2]+0.3],100)
        MoveLinear(target,[target_send_pos[0],target_send_pos[1],target_send_pos[2]+0.29],[target_send_pos[0],target_send_pos[1],target_send_pos[2]+0.05],24)
        sim.simxSetObjectParent(clientID,box_handle,-1,0,sim.simx_opmode_oneshot ) #remove parent 
        setOrientation(box_handle,[0,0,0]) #set the orientation in world frame
        setPosition(box_handle,target_send_pos) # get back to basket
        MoveLinear(target,[target_send_pos[0],target_send_pos[1],target_send_pos[2]+0.05],[target_send_pos[0],target_send_pos[1],target_send_pos[2]+0.29],24)
        if (turn+1)%10==0 and turn!=0: #for every 10 turn start robo 2
            #move robot 1 to aside
            target_current = getPosition(target)
            MoveLinear(target,target_current,[-1.525,-0.825,0.75],100)
            #call to robot 2 
            start_robot2(flag_places)
    target_current = getPosition(target)
    MoveLinear(target,target_current,[-2.0578,-0.79,0.595],150)

def neibour_pos(handle_of_basket,z_level): # generate 25 neibourhoods
    origin_pos = getPosition(handle_of_basket)
    positions_in_basket = []
    temp_pos = []
    for x in range(2,-3,-1):
        for y in range(2,-3,-1):
            temp_pos = [origin_pos[0]+0.1*x,origin_pos[1]+0.1*y,z_level]
            positions_in_basket.append(temp_pos)
    return positions_in_basket

def start_robot2(flag_pos):
    global r
    global g
    global y
    global o
    x0,y0 = 1.025,1.275 #set initial cube position
    r_index,g_index,y_index,o_index = 0,0,0,0

    ####### handles of objects
    target1 = getHandle("Target1")
    attach1 = getHandle("suctionPadLink#0")
    green_basket = getHandle("BasketGreen")
    red_basket = getHandle("BasketRed")
    yellow_basket = getHandle("BasketYellow")
    orange_basket = getHandle("BasketOrange")

    target_current = getPosition(target1) #initial position of target at begining of simulation
    red_basket_pos = neibour_pos(red_basket,0.015)
    green_basket_pos = neibour_pos(green_basket,0.015)
    orange_basket_pos = neibour_pos(orange_basket,0.015)
    yellow_basket_pos = neibour_pos(yellow_basket,0.015)

    
    for flag_pos_elements in flag_pos:
        color = flag_pos_elements[2]
        pos = [x0+flag_pos_elements[0]*0.031,y0+flag_pos_elements[1]*0.031]
        if flag_pos_elements in placed_flag:
            continue
        if color == "R" and len(red_in_bas)!=0:
            take_pos_red = red_basket_pos[r]
            if r==24:
                r=0
            else:
                r+=1
            box_handle = getHandle(red_in_bas[r_index])
            target_current = getPosition(target1)
            MoveLinear(target1,target_current,[take_pos_red[0],take_pos_red[1],take_pos_red[2]+0.3],100)
            MoveLinear(target1,[take_pos_red[0],take_pos_red[1],take_pos_red[2]+0.29],take_pos_red,29)
            sim.simxSetObjectParent(clientID,box_handle,attach1,1,sim.simx_opmode_oneshot )
            MoveLinear(target1,take_pos_red,[take_pos_red[0],take_pos_red[1],take_pos_red[2]+0.29],29)
            MoveLinear(target1,[take_pos_red[0],take_pos_red[1],take_pos_red[2]+0.3],[pos[0],pos[1],0.30],100)
            MoveLinear(target1,[pos[0],pos[1],0.29],[pos[0],pos[1],0.1],19)
            sim.simxSetObjectParent(clientID,box_handle,-1,0,sim.simx_opmode_oneshot ) #remove parent 
            setOrientation(box_handle,[0,0,0]) #set the orientation in world frame
            setPosition(box_handle,[pos[0],pos[1],0.015]) 
            MoveLinear(target1,[pos[0],pos[1],0.1],[pos[0],pos[1],0.29],19)
            red_in_bas.pop(r_index)
            placed_flag.append(flag_pos_elements)
            

        if color == "G" and len(green_in_bas)!=0:
            pos_g = getPosition(green_basket)
            take_pos_green = [pos_g[0],pos_g[1],0.015]
            #write to move take lift come placein flag
            box_handle = getHandle(green_in_bas[g_index])
            target_current = getPosition(target1)
            MoveLinear(target1,target_current,[take_pos_green[0],take_pos_green[1],take_pos_green[2]+0.3],100)
            MoveLinear(target1,[take_pos_green[0],take_pos_green[1],take_pos_green[2]+0.29],take_pos_green,29)
            sim.simxSetObjectParent(clientID,box_handle,attach1,1,sim.simx_opmode_oneshot )
            MoveLinear(target1,take_pos_green,[take_pos_green[0],take_pos_green[1],take_pos_green[2]+0.29],29)
            MoveLinear(target1,[take_pos_green[0],take_pos_green[1],take_pos_green[2]+0.3],[pos[0],pos[1],0.30],100)
            MoveLinear(target1,[pos[0],pos[1],0.29],[pos[0],pos[1],0.1],19)
            sim.simxSetObjectParent(clientID,box_handle,-1,0,sim.simx_opmode_oneshot ) #remove parent 
            setOrientation(box_handle,[0,0,0]) #set the orientation in world frame
            setPosition(box_handle,[pos[0],pos[1],0.015]) 
            MoveLinear(target1,[pos[0],pos[1],0.1],[pos[0],pos[1],0.29],19)
            green_in_bas.pop(g_index)
            placed_flag.append(flag_pos_elements)

        if color == "Y" and len(yellow_in_bas)!=0:
            take_pos_yellow = yellow_basket_pos[y]
            if y==24:
                y=0
            else:
                y+=1
            #write to move take lift come placein flag
            box_handle = getHandle(yellow_in_bas[y_index])
            target_current = getPosition(target1)
            MoveLinear(target1,target_current,[take_pos_yellow[0],take_pos_yellow[1],take_pos_yellow[2]+0.3],100)
            MoveLinear(target1,[take_pos_yellow[0],take_pos_yellow[1],take_pos_yellow[2]+0.29],take_pos_yellow,29)
            sim.simxSetObjectParent(clientID,box_handle,attach1,1,sim.simx_opmode_oneshot )
            MoveLinear(target1,take_pos_yellow,[take_pos_yellow[0],take_pos_yellow[1],take_pos_yellow[2]+0.29],29)
            MoveLinear(target1,[take_pos_yellow[0],take_pos_yellow[1],take_pos_yellow[2]+0.3],[pos[0],pos[1],0.30],100)
            MoveLinear(target1,[pos[0],pos[1],0.29],[pos[0],pos[1],0.1],19)
            sim.simxSetObjectParent(clientID,box_handle,-1,0,sim.simx_opmode_oneshot ) #remove parent 
            setOrientation(box_handle,[0,0,0]) #set the orientation in world frame
            setPosition(box_handle,[pos[0],pos[1],0.015]) 
            MoveLinear(target1,[pos[0],pos[1],0.1],[pos[0],pos[1],0.29],19)
            yellow_in_bas.pop(y_index)
            placed_flag.append(flag_pos_elements)

        if color == "O" and len(orange_in_bas)!=0:
            pos_o = getPosition(orange_basket)
            take_pos_orange = [pos_o[0],pos_o[1],0.015]
            #write to move take lift come placein flag
            box_handle = getHandle(orange_in_bas[o_index])
            target_current = getPosition(target1)
            MoveLinear(target1,target_current,[take_pos_orange[0],take_pos_orange[1],take_pos_orange[2]+0.3],100)
            MoveLinear(target1,[take_pos_orange[0],take_pos_orange[1],take_pos_orange[2]+0.29],take_pos_orange,29)
            sim.simxSetObjectParent(clientID,box_handle,attach1,1,sim.simx_opmode_oneshot )
            MoveLinear(target1,take_pos_orange,[take_pos_orange[0],take_pos_orange[1],take_pos_orange[2]+0.29],29)
            MoveLinear(target1,[take_pos_orange[0],take_pos_orange[1],take_pos_orange[2]+0.3],[pos[0],pos[1],0.30],100)
            MoveLinear(target1,[pos[0],pos[1],0.29],[pos[0],pos[1],0.1],19)
            sim.simxSetObjectParent(clientID,box_handle,-1,0,sim.simx_opmode_oneshot ) #remove parent 
            setOrientation(box_handle,[0,0,0]) #set the orientation in world frame
            setPosition(box_handle,[pos[0],pos[1],0.015]) 
            MoveLinear(target1,[pos[0],pos[1],0.1],[pos[0],pos[1],0.29],19)
            orange_in_bas.pop(o_index)
            placed_flag.append(flag_pos_elements)

        if len(red_in_bas)==0 and len(green_in_bas)==0 and len(yellow_in_bas)==0 and len(orange_in_bas)==0:
            break
        else:
            continue


## creating the list of names of box objects-----------------------
namelist = []
number_of_boxes = 384
for i in range(1,number_of_boxes+1):
    namelist.append("B"+str(i))

# Main----------------------------------------------
pick_and_place_in_basket(namelist)


# close connection------------------------------------
sim.simxGetPingTime(clientID)
sim.simxFinish(clientID)