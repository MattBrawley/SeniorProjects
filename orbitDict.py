import pickle 
import os

class chosenOrbits:
    def __init__(self, family, x, y, z, vx, vy, vz, T, eclipse, stability):
        self.family =  family # orbit family 
        self.x = x # x position LU
        self.y = y # y position LU
        self.z = z # z position LU
        self.vx = vx # vx velocity LU/TU
        self.vy = vy # vy velocity LU/TU
        self.vz = vz # vz velocity LU/TU
        self.T = T # period in TU
        self.eclipse = eclipse # percent of an orbit period that the satellite is being eclipsed (no direct sun)
        self.stability = stability # stability index 
    
    def __repr__(self):
        return "Orbit: %s" % self.s
    def save(self, fileName):
        """Save thing to a file."""
        f = open(fileName,"wb")
        pickle.dump(self,f)
        f.close()
    def load(fileName):
        """Return a thing loaded from a file."""
        this_dir, this_filename = os.path.split(__file__)  # Get path of data.pkl
        data_path = os.path.join(this_dir, fileName)
        f = open(data_path, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj
    # make load a static method
    load = staticmethod(load)


if __name__ == "__main__":
    # code for standalone use
    foo = chosenOrbits("family", "x", "y", "z", "vx", "vy", "vz", "T", "eclipse", "stability")
    foo.save("foo.pickle")
