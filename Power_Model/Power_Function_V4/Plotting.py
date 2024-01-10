def PlotThisDesign(PosnTime, TransPosnTime, PownComm, pos_rec, efficiency_profs, theta_r_profs, d_profs, t_profs):
    # this function plots the chosen design, with all artistic needs met to demonstrate
    # all necessary aspects of the full design

    # imports and constants
    import matplotlib.pyplot as plt
    import numpy
    ##### constants
    r_m = 1737400;  # radius of the moon, m
    LU = 389703000 # distance of Earth to Moon in m

    # 3D axis equal function, found online @  https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    import numpy as np

    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])





    ##### 3D ORBIT PLOT

    # define plot
    ax = plt.figure().add_subplot(projection='3d')

    u, v = numpy.mgrid[0:2*numpy.pi:20j, 0:numpy.pi:10j]

    # plot the moon, a grey sphere with a radius of, you guessed it, one lunar radius
    x = r_m * numpy.cos(u) * numpy.sin(v) + LU
    y = r_m * numpy.sin(u) * numpy.sin(v)
    z = r_m * numpy.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.Greys)
    plt.xlabel('m')
    # plotting of transmission orbits, and transmission periods


    transmitting_orbit_range = PownComm[0]
    comms_orbit_range = PownComm[1]

    count = 0
    for i in transmitting_orbit_range:
        print('Power Orbits Plotted')
        ax.plot(PosnTime[i][0], PosnTime[i][1], PosnTime[i][2], 'b') # plotting power orbits
        print(transmitting_orbit_range)
        print(count)
        if len(TransPosnTime[count]) == 2:
            thistransmission_x = []
            thistransmission_y = []
            thistransmission_z = []
            for j in range(0,len(TransPosnTime[count])-1):
                thistransmission_x.append(TransPosnTime[count][j][0])
                thistransmission_y.append(TransPosnTime[count][j][1])
                thistransmission_z.append(TransPosnTime[count][j][2])

            ax.plot(thistransmission_x,thistransmission_y,thistransmission_z, 'r')
        else:
            ax.plot(TransPosnTime[count][0],TransPosnTime[count][1],TransPosnTime[count][2])
        #ax.scatter(pos_rec[count][0], pos_rec[count][1], pos_rec[count][2], label='ground station') # this often outshines the moon

        count = count + 1


    # plotting of comms orbits
    for i in comms_orbit_range:
        print('Comms Orbits Plotted')
        ax.plot(PosnTime[i][0], PosnTime[i][1], PosnTime[i][2], linestyle='dashed', color='g') # plotting power orbits


    # plot labeling and definitions
    set_axes_equal(ax)
    ax.set_title("Chosen Design")
    plt.show()



    for i in [0,1]:

        ##### Plotting of efficiency over time
        print(i)
        fig, ax2 = plt.subplots()

        ax2.plot(t_profs[i][:-1],efficiency_profs[i])
        plt.ylim(numpy.mean(efficiency_profs)-0.20*numpy.mean(efficiency_profs),numpy.mean(efficiency_profs)+0.50*numpy.mean(efficiency_profs))
        plt.title('Efficiency vs Time')
        plt.xlabel('Transmission Time, [s]')
        plt.ylabel('Transmission Effiency, [%]')
        plt.show()


        ##### Plotting of angle over time

        fig, ax3 = plt.subplots()

        ax3.plot(t_profs[i], theta_r_profs[i])
        #plt.ylim(numpy.mean(theta_r_profs) - 0.20 * numpy.mean(theta_r_profs),
        #         numpy.mean(theta_r_profs) + 0.50 * numpy.mean(theta_r_profs))
        plt.title('Incident Angle vs Time')
        plt.xlabel('Transmission Time, [s]')
        plt.ylabel('Incident Angle, [deg]')
        plt.show()

        ##### Plotting of angle over time

        fig, ax4 = plt.subplots()

        ax4.plot(t_profs[i], d_profs[i])
        # plt.ylim(numpy.mean(theta_r_profs) - 0.20 * numpy.mean(theta_r_profs),
        #         numpy.mean(theta_r_profs) + 0.50 * numpy.mean(theta_r_profs))
        plt.title('Distance vs Time')
        plt.xlabel('Transmission Time, [s]')
        plt.ylabel('Distance to Groundstation, [m]')
        plt.show()



    return
