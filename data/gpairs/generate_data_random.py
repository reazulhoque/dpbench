import numpy as np

def generate_data(npoints, seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, 1, npoints * 4)

def dump_in_csv(npoints, data, names):
    Lbox = 500.

    start = 0
    end = npoints
    for idx, name in enumerate(names):
        start = idx * npoints
        end = start + npoints
        d = data[start:end] * Lbox
        np.savetxt(name + ".csv", d, delimiter=",")

def generate_bin(nbins, rmin, rmax, name):
    rbins = np.logspace(
        np.log10(rmin), np.log10(rmax), nbins).astype(
            np.float64)
    rbins_squared = (rbins**2).astype(np.float64)
    np.savetxt(name + ".csv", rbins_squared, delimiter=",")


def main():
    npoints = 10240
    nbins = 20
    rmin, rmax = 0.1, 50

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', required=False, default=npoints,  help="Number of points")
    parser.add_argument('--nbins', required=False, default=nbins,  help="Number of bins")
    parser.add_argument('--rmin', required=False, default=rmin,  help="Rmin")
    parser.add_argument('--rmax', required=False, default=rmax,  help="Rmax")

    args   = parser.parse_args()
    npoints = int(args.points)

    data = generate_data(npoints, 0)
    dump_in_csv(npoints, data, ["x1", "y1", "z1", "w1"])
    data = generate_data(npoints, 1)
    dump_in_csv(npoints, data, ["x2", "y2", "z2", "w2"])
    generate_bin(nbins, rmin, rmax, "rbins_squared")


if __name__ == "__main__":
    main()
