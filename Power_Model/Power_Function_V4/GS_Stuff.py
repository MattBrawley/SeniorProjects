def GroundStationLocator(PosnTime, numSats):
    import math
    # find transmitting orbit indices and pass over index
    LU = 389703000 # distance of Earth to Moon in m

    ###### find the transmitting orbit indices first:

    if sum(numSats) != max(numSats): # when all spacecraft are NOT on one orbit

        max_orbit_indices = []
        for i in range(0,len(numSats)):     # find all orbits that contain the max num of sc
            if numSats[i] == max(numSats):
                max_orbit_indices.append(i) # store the indices of the times we are at maximum

        if len(max_orbit_indices) == 1: # if the max is truly the max, we need to find the second most.

            second_most_sc = 0
            for i in range(0,len(numSats)):
                if i != max_orbit_indices:

                    if numSats[i] > second_most_sc:
                        second_most_sc = numSats[i]
                        second_most_index = i

            transmitting_orbits = [max_orbit_indices[0], second_most_index]

        else: # this is when there are more than 1 orbit with the max amount of sc, arbitrarily take the first two with the max

            transmitting_orbits = [max_orbit_indices[0], max_orbit_indices[1]]

    else: # when there is only one orbit with all sc

        transmitting_orbits = [numSats.index(max(numSats))]

    ##### now we need to find the passover indices of the two given orbits


    pass_index = []
    if len(transmitting_orbits) == 2:

        for i in range(0,len(transmitting_orbits)-1): # calculate the pass index for each orbit

            thisOrbit_index = transmitting_orbits[i]
            thisPosnTime = PosnTime[thisOrbit_index]
            x_vec = thisPosnTime[0]
            y_vec = thisPosnTime[1]
            z_vec = thisPosnTime[2]

            min_z_indices = []
            for j in range(0, len(z_vec)):  # find all orbits that contain the max num of sc
                if z_vec[j] == min(z_vec):
                    min_z_indices.append(j)  # store the indices of the times we are at maximum

            pass_index.append(min_z_indices[0])

            min_d_indices = []
            if len(min_z_indices) != 1: # this is when our min z isnt unique, and we need to find another time to transmit
                d = []
                for j in range(0,len(z_vec)):
                    d.append(math.sqrt((x_vec[j]+LU)**2+y_vec[j]**2+z_vec[j]**2))

                for j in range(0, len(d)):  # find all orbits that contain the max num of sc
                    if d[j] == min(d):
                        min_d_indices.append(j)  # store the indices of the times we are at maximum

                pass_index.append(min_d_indices[0])
    else:

        thisPosnTime = PosnTime[transmitting_orbits[0]]
        x_vec = thisPosnTime[0]
        y_vec = thisPosnTime[1]
        z_vec = thisPosnTime[2]

        min_z_indices = []
        for j in range(0, len(z_vec)):  # find all orbits that contain the max num of sc
            if z_vec[j] == min(z_vec):
                min_z_indices.append(j)  # store the indices of the times we are at maximum

        pass_index.append(min_z_indices[0])

        min_d_indices = []
        if len(min_z_indices) != 1:  # this is when our min z isnt unique, and we need to find another time to transmit
            d = []
            for j in range(0, len(z_vec)):
                d.append(math.sqrt((x_vec[j]+LU)** 2 + y_vec[j] ** 2 + z_vec[j] ** 2))

            for j in range(0, len(d)):  # find all orbits that contain the max num of sc
                if d[j] == min(d):
                    min_d_indices.append(j)  # store the indices of the times we are at maximum

            pass_index.append(min_d_indices[0])

    if len(transmitting_orbits) != len(pass_index): # the rare case where we dont find a second pass index fsr, so we set it to 0.
        if len(transmitting_orbits) == 1:
            pass_index = 0
        if len(transmitting_orbits) == 2:
            if len(pass_index) == 0:
                pass_index = [0,0]
            if len(pass_index) == 1:
                pass_index.append(0)


    return [transmitting_orbits, pass_index]


##### OLD METHOD OF FINDING GS IN FullPowerModelFunction
    #
    # # find the orbit with the most sc, and it's index
    # first_orbit_num = max(numSats)
    # first_orbit_ind = numSats.index(first_orbit_num)
    # second_orbit_num = 0
    #
    # # find all populated orbits, to search through those for the orbit with the second most sc
    #
    # # find indices of orbits with sc on them
    # populated_orbits = []
    # for i in range(0, NumOrbits):
    #     if numSats[i] != 0:
    #         populated_orbits.append(i)
    # populated_orbits_num = len(populated_orbits)
    #
    # if populated_orbits_num >= 2:  # if there is more than 1 populated orbit, find the second
    #     for i in populated_orbits:  # check all populated orbits
    #         if i != first_orbit_ind:  # check all but the most populated orbit
    #             if numSats[i] > second_orbit_num:  # if it userps, reassign and remember index
    #                 second_orbit_num = numSats[i]
    #                 second_orbit_ind = i
    #
    #     orbit_range = [first_orbit_ind,
    #                    second_orbit_ind]  # these are the indices of the two orbits we actually want to calculate power and groundstations for
    #
    # elif populated_orbits_num == 1:  # if there is only one populated orbit, then its not a range just that one index
    #
    #     orbit_range = [first_orbit_ind]
    #
    # # preallocation
    # GSLocations = []
    # E_rec_perT = []
    # E_rec_perD = []
    # Eff = []
    # numSats_transmitting = []
    # pos_rec = []
    # count = 0
    # this_loc = []
    # alpha_ave = []
    # d_ave = []
    # theta_r_transmission = []
    # periods = []
    #
    # for i in orbit_range:  # for these two orbits we care about...
    #     fullposmat_curr = PosnTime[i]  # current posntime matrix of this orbit
    #
    #     # now find optimal ground station for this orbit. Minimum z value is chosen, if no z is in southern hemisphere chooses closest surface approach
    #
    #     for j in range(0, len(fullposmat_curr[0])):
    #
    #         min_z = 0
    #         if fullposmat_curr[2][j] < min_z:  # find minimum z coordinate in orbit, and corresponding index
    #             min_z = fullposmat_curr[2][j]
    #             GS_index = j
    #
    #     if min_z == 0:  # this is the case where the orbit never has a neg z value, and is never over southern hem
    #         d_min = 10 ** 10
    #         for j in range(0, len(fullposmat_curr[0])):
    #
    #             d = math.sqrt(fullposmat_curr[0][j] ** 2 + fullposmat_curr[1][j] ** 2 + fullposmat_curr[2][
    #                 j] ** 2)  # find current distance from center of moon
    #             if d < d_min:  # find minimum z coordinate in orbit, and corresponding index
    #                 d_min = d
    #                 GS_index = j
    #
    #     # now find r_rec lat and lon
    #     lat = round(math.atan(fullposmat_curr[2][GS_index] / (
    #         math.sqrt(fullposmat_curr[0][GS_index] ** 2 + fullposmat_curr[1][GS_index] ** 2))) * 180 / numpy.pi, 3)
    #     lon = round(math.atan(fullposmat_curr[1][GS_index] / (fullposmat_curr[0][GS_index])) * 180 / numpy.pi, 3)
    #     this_loc.append([lat, lon])
    #
    #     # find receiver pos in vector form
    #     d_t = MagFunc([fullposmat_curr[0][GS_index], fullposmat_curr[1][GS_index],
    #                    fullposmat_curr[2][GS_index]])  # distance of transmission
    #     u_t = [fullposmat_curr[0][GS_index] / d_t, fullposmat_curr[1][GS_index] / d_t,
    #            fullposmat_curr[2][GS_index] / d_t]  # unit vector of transmission
    #     pos_rec.append([u_t[0] * r_m, u_t[1] * r_m,
    #                     u_t[2] * r_m])  # position of receiver, radius of moon * unit vector of transmission point
    #     # pos_rec.append([u_t[0]*d_t,u_t[1]*d_t,u_t[2]*d_t])
    #
    #     GSLocations.append(this_loc[count])  # save the location for later