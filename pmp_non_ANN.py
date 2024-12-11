import roboticstoolbox as rtb
import numpy as np


# Kompliance
KFORCE = 40
ITERATION=1000
RAMP_KONSTANT=0.005
t_dur=5
KOMP_JANG=0.0001
J2H=1;

inputL=6
outputL=3

# Define UR5 DH parameters for forward kinematics
#def dh_transform(a, alpha, d, theta):
#    """Calculate the transformation matrix using DH parameters."""
#    return np.array([
#        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
#        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
#        [0, np.sin(alpha), np.cos(alpha), d],
#        [0, 0, 0, 1]
#    ])

def ur5_jacobian(joint_angles_deg):

    # Define UR5 DH parameters in millimeters
    ur5= rtb.models.DH.UR5() #This is UR5 not UR5e

    joint_angles_rad = np.radians(joint_angles_deg)

    # Compute the Jacobian at the specified joint configuration
    jacobian = ur5.jacob0(joint_angles_rad)

    return jacobian
#put the Initial position of the arm
def InitializeJan():
    Jan = np.zeros(6)
    Jan[0]= 94.68
    Jan[1] = -86.32
    Jan[2]= -142.83
    Jan[3]= -133.48
    Jan[4]= -84.74
    Jan[5]= 0.77

    janini0 = 94.68
    janini1 = -86.32
    janini2 = -142.83
    janini3 = -133.48
    janini4 = -84.74
    janini5= 0.77

    x_iniIC= 91.2259
    y_iniIC= 315.2707
    z_iniIC= 118.6032
    x_ini = 91.2259
    y_ini = 315.2707
    z_ini = 118.6032

    return Jan, janini0, janini1, janini2, janini3, janini4, janini5, x_iniIC, y_iniIC, z_iniIC, x_ini, y_ini, z_ini

def forward_Kinematics(joint_angles_deg):
    """
    Compute the forward kinematics for the UR5 using the DH parameters.

    Parameters:
        joint_angles: A list or array of 6 joint angles (in radians).

    Returns:
        A numpy array [x, y, z] representing the position of the end-effector.
    """
    #joint_angles = np.radians(joint_angles_deg)
    # Define the DH parameters (a, alpha, d, theta)
    #dh_params = [
    #    (0, np.pi / 2, 89.46, joint_angles[0]),
    #    (-425, 0, 0, joint_angles[1]),
    #    (-392.2, 0, 0, joint_angles[2]),
    #    (0, np.pi / 2, 109.1, joint_angles[3]),
    #    (0, -np.pi / 2, 94.65, joint_angles[4]),
    #    (0, 0, 82.3, joint_angles[5])
    #]

    # Compute the transformation matrix for each joint
    #T = np.eye(4)  # Start with the identity matrix
    #for params in dh_params:
    #    T = np.dot(T, dh_transform(*params))

    # Extract the end-effector position from the final transformation matrix
    #end_effector_position = T[:3, 3]  # Extract x, y, z
    # Load the UR5 model
    ur5 = rtb.models.DH.UR5()

    # Convert joint angles to radians
    joint_angles_rad = np.radians(joint_angles_deg)

    # Compute forward kinematics
    T = ur5.fkine(joint_angles_rad)  # T is the end-effector pose as a SE(3) matrix

    # Extract the end-effector position (x, y, z)
    end_effector_position = T.t * 1000  # T.t gives the translational part of the SE(3) matrix

    return end_effector_position

def forcefield(w,v):
    j = 0
    pos = np.zeros(3)
    tar = np.zeros(3)
    res = np.zeros(3)
    for j in range(3):
        pos[j] = w[j]
        tar[j] = v[j]
        res[j] = KFORCE * (tar[j] - pos[j])
    ptr = res
    return ptr

def pmp(force, Jan):
    ff = np.zeros(3)
    Joint_Field = np.zeros(10)
    Jvel = np.zeros(6)

    for i in range(3):
        ff[i] = force[i]

    JacT =[]

    Jack = ur5_jacobian(Jan)

    for i in range(inputL):
        temp = []
        for j in range(outputL):
            temp.append(Jack[j,i])
        JacT.append(temp)

    Joint_Field[0]=(90-Jan[0])*1;# 45 25
    Joint_Field[1]=(-90-Jan[1])*J2H; # J2H = 1
    Joint_Field[2]=(-100-Jan[2])*J2H;
    Joint_Field[3]=(-100-Jan[3])*J2H*1; #50  5
    Joint_Field[4]=(-80-Jan[4])*J2H*1;#50  75
    Joint_Field[5]=(0-Jan[5])*J2H*1; #was 30   100

    for a in range(inputL):
        jvelo = 0
        for n in range(outputL):
            jvelo = jvelo+(JacT[a][n]*ff[n])
        Jvel[a] = 0.002*(jvelo+Joint_Field[a])
    foof = Jvel
    return foof


def GammaDisc(_Time):
    t_ramp=(_Time)*RAMP_KONSTANT
    t_init=0.1
    z=(t_ramp-t_init)/t_dur
    t_win=(t_init+t_dur)-t_ramp

    if t_win>0:
        t_window=1
    else:
        t_window=0
    csi=(6*pow(z,5))-(15*pow(z,4))+(10*pow(z,3)) #6z^5-15z^4+10z^3
    csi_dot=(30*pow(z,4))-(60*pow(z,3))+(30*pow(z,2)) #csi_dot=30z^4-60z^3+30z^2
    prod1=(1/(1.0001-(csi*t_window)))
    prod2=(csi_dot*0.3333*t_window)
    Gamma=prod1*prod2

    return Gamma

def Gamma_IntDisc(Gar, n):
    k = 1
    a = 0
    sum =Gar[0]
    c = 2
    h = 1
    while k <= (n-1):
        fk = Gar[k]
        c = 6-c
        sum = (sum + c*fk)
        k += 1
    sum=RAMP_KONSTANT*sum/3

    return sum

def MotCon(T1, T2, T3, time, Gam, Jan, q1, q2, q3, q4, q5, q6, janini0, janini1, janini2, janini3, janini4, janini5):
    ang = Jan
    nFK = forward_Kinematics(ang)
    X_pos = np.zeros(3)
    target = np.zeros(3)

    for i in range(3):
        X_pos[i] = nFK[i]
    po = X_pos

    target[0]=T1
    target[1]=T2
    target[2]=T3

    ta = target
    force = forcefield(po,ta)

    ffield = np.zeros(3)
    for i in range(3):
        ffield[i] = force[i]
    topmp= ffield
    Q_Dot=pmp(topmp, Jan)

    JoVel = np.zeros(6)
    for i in range(6):
        JoVel[i]=(Q_Dot[i])*Gam

    q1[time]=JoVel[0]
    j1=q1
    joi1=Gamma_IntDisc(j1,time)
    Jan[0]=joi1+janini0

    q2[time]=JoVel[1]
    j2=q2
    joi2=Gamma_IntDisc(j2,time)
    Jan[1]=joi2+janini1

    q3[time]=JoVel[2]
    j3=q3
    joi3=Gamma_IntDisc(j3,time)
    Jan[2]=joi3+janini2

    q4[time]=JoVel[3]
    j4=q4
    joi4=Gamma_IntDisc(j4,time)
    Jan[3]=joi4+janini3

    q5[time]=JoVel[4]
    j5=q5
    joi5=Gamma_IntDisc(j5,time)
    Jan[4]=joi5+janini4

    q6[time]=JoVel[5]
    j6=q6
    joi6=Gamma_IntDisc(j6,time)
    Jan[5]=joi6+janini5

    return Jan, X_pos, q1, q2, q3, q4, q5,q6

def VTGS(XT1, YT2, ZT3, XO1, YO2, ZO3, ChoiceAct, MentalSim, WristGraspPose):

    results = []

    Jan, janini0, janini1, janini2, janini3, janini4, janini5, x_iniIC, y_iniIC, z_iniIC, x_ini, y_ini, z_ini = InitializeJan()

    fin = np.zeros(3)
    n = 3
    retvalue=0

    if ChoiceAct==0:
        if XT1==0 and YT2==0 and ZT3==0: #for iCub
            XT1=-120
            YT2=-140
            ZT3=572

        fin = [XT1, YT2, ZT3]

        xoffs=0.0
        yoffs=0.0
        zoffs=0.0

        replan=0

        x_fin=fin[0]
        y_fin=fin[1]
        z_fin=fin[2]

        print(" Targets")
        print(x_fin,y_fin,z_fin)

        Gam_Arr = np.zeros(ITERATION)
        Gam_Arry = np.zeros(ITERATION)
        Gam_Arrz = np.zeros(ITERATION)

        q1 = np.zeros(ITERATION)
        q2 = np.zeros(ITERATION)
        q3 = np.zeros(ITERATION)
        q4 = np.zeros(ITERATION)
        q5 = np.zeros(ITERATION)
        q6 = np.zeros(ITERATION)

        for time in range(ITERATION):
            Gam=GammaDisc(time);

            #Target Generation

            inter_x=(x_fin-x_ini)*Gam
            Gam_Arr[time]=inter_x
            Gar=Gam_Arr
            x_ini=Gamma_IntDisc(Gar,time)+x_iniIC

            inter_y=(y_fin-y_ini)*Gam
            Gam_Arry[time]=inter_y
            Gary=Gam_Arry
            y_ini=Gamma_IntDisc(Gary,time)+y_iniIC

            inter_z=(z_fin-z_ini)*Gam
            Gam_Arrz[time]=inter_z
            Garz=Gam_Arrz
            z_ini=Gamma_IntDisc(Garz,time)+z_iniIC

            Jan, X_pos, q1, q2, q3, q4, q5, q6 = MotCon(x_ini, y_ini, z_ini, time, Gam, Jan, q1, q2, q3, q4, q5, q6, janini0, janini1, janini2, janini3, janini4, janini5)

            results.append([Jan[0],Jan[1],Jan[2],Jan[3],Jan[4],Jan[5],x_ini,y_ini,z_ini])

        konst=1
        ang1=konst*Jan[0]
        ang2=konst*Jan[1]
        ang3=konst*Jan[2]
        ang4=konst*Jan[3]
        ang5=konst*Jan[4]
        ang6=konst*Jan[5]
        print("\n Joint Angles: ",ang1,ang2,ang3,ang4,ang5,ang6)
        print("\n\n FINAL SOLUTION: ",X_pos[0],X_pos[1],X_pos[2])
        np.savetxt('results.txt', results, fmt='%f')
        #time.sleep(1)

def TargGenSMo():

    target_points = np.zeros(3)

    try:
        f = open('target_points.txt')
        matrix = f.read().split()
        target_points[0] = matrix[0]
        target_points[1] = matrix[1]
        target_points[2] = matrix[2]
    except:
        print("Oops!  Cannot find the target file.")

    VTGS(target_points[0], target_points[1], target_points[2], 0,0,0,0,0,0)

if __name__ == "__main__":

    TargGenSMo()
