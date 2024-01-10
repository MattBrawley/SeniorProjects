# CIRCLE ORBIT GENERATOR
# Created: Cameron King; 2/1/2023
# finalized: Cameron King; 2/2/2023


def CircleOrbitGenerator(center_pos,V,t_end,N):
    # this function will generate a [x,y,z,t] mat of length N in the transmission time period on a circular orbit.
    # this is done through the center point of the transmission period,
    # the current velocity on that trajectory, the total time of transmission, and the desired number of elements

    # imports
    import math
    import mpmath
    mpmath.dps = 20

    # simple magnitude function, used throughout
    def MagFunc(vec):
        return math.sqrt(vec[0]**2+vec[1]**2+vec[2]**2)

    orbit_radius = MagFunc(center_pos) # radius scalar

    t_step = t_end/N # time step
    d_step = V*t_step # distance step along orbit
    angle_step = d_step/orbit_radius # angle step
    inclination = mpmath.atan(center_pos[2]/(mpmath.sqrt(center_pos[0]**2+center_pos[1]**2)))

    # define center as (0,0), go from -N/2 to N/2 around the 0,0 to generate x,y,z and t
    new_i_range = range(round(-N/2),round(N/2))

    current_pos = []
    for i in range(0,N):

        new_i = new_i_range[i]
        t_current = i*t_step
        current_angle = new_i*angle_step # current angle between position, center of transmission, and moon center

        current_pos_onC = [orbit_radius*mpmath.cos(current_angle),orbit_radius*mpmath.sin(current_angle)] # used trig to get current position in circle plane

        current_pos.append([current_pos_onC[0]*mpmath.cos(inclination),current_pos_onC[1],current_pos_onC[0]*mpmath.sin(inclination),t_current]) # rotate current_pos_onC around y

    return current_pos # output the [x,y,z,t] vector of the transmission period


def StraitLineOrbitApprox(pos_vec,t_end):
    # This function takes position and velocity vectors, at least 2 points but any length,
    # and output the [x,y,z,t] vector of the transmission period, with AT LEAST 15 ELEMENTS
    import math
    def MagFunc(vec):
        return math.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)

    N_required = 20
    N_initial = len(pos_vec)

    new_pos_vec = []

    if len(pos_vec) > 2:

        # linearize the area between each given points, generate new points between, create new position vector
        t_current = 0
        add_between = round(N_required / N_initial) +1
        for gap in range(0,N_initial-1):

            vec_to_next = [pos_vec[gap+1][0]-pos_vec[gap][0],pos_vec[gap+1][1]-pos_vec[gap][1],pos_vec[gap+1][2]-pos_vec[gap][2]]
            d_mag_to_next = MagFunc(vec_to_next)
            unit_to_next = [vec_to_next[0]/d_mag_to_next,vec_to_next[1]/d_mag_to_next,vec_to_next[2]/d_mag_to_next]
            d_to_addition = d_mag_to_next/(add_between+1)
            this_t_step = pos_vec[gap+1][3]-pos_vec[gap][3]

            print(d_to_addition)

            new_pos_vec.append([pos_vec[gap][0],pos_vec[gap][1],pos_vec[gap][2],t_current])
            t_current = t_current + this_t_step

            for addition in range(0,add_between-1):
                new_pos_vec.append([unit_to_next[0]*d_to_addition*(addition+1),unit_to_next[1]*d_to_addition*(addition+1),unit_to_next[2]*d_to_addition*(addition+1),t_current])
                t_current = t_current + this_t_step

        new_pos_vec.append([pos_vec[gap+1][0],pos_vec[gap+1][1],pos_vec[gap+1][2],t_current])

    else:

        ref_vec = [pos_vec[1][0]-pos_vec[0][0],pos_vec[1][1]-pos_vec[0][1],pos_vec[1][2]-pos_vec[0][2]]
        center_pos_vec = pos_vec[0]
        ref_d = MagFunc(ref_vec)
        ref_unit = [ref_vec[0]/ref_d, ref_vec[1]/ref_d, ref_vec[2]/ref_d]
        V_trans = ref_d/pos_vec[1][3]
        t_step = t_end / 20
        d_between = V_trans*t_step
        count = 0

        this_pos = []
        for i in range(-20,20):

            this_time = count*t_step
            count = count + 1
            dist_fr_center = i*d_between
            this_x = center_pos_vec[0]+ref_unit[0]*dist_fr_center
            this_y = center_pos_vec[1]+ref_unit[1]*dist_fr_center
            this_z = center_pos_vec[2]+ref_unit[2]*dist_fr_center

            #this_pos  [center_pos_vec[0]+ref_unit[0]*dist_fr_center,center_pos_vec[1]+ref_unit[1]*dist_fr_center,center_pos_vec[2]+ref_unit[2]*dist_fr_center,this_time]
            #print('this_pos[0]:',center_pos_vec[0]+ref_unit[0]*dist_fr_center)
            #print('this_pos:',this_pos)
            new_pos_vec.append([this_x,this_y,this_z,this_time])

    lists = [[val for val in arr] for arr in new_pos_vec]

    return lists

