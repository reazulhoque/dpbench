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


def main():
    npoints = 10240
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--points', required=False, default=npoints,  help="Number of points")

    args   = parser.parse_args()
    npoints = int(args.points)

    data = generate_data(npoints, 0)
    dump_in_csv(npoints, data, ["x1", "y1", "z1", "w1"])
    data = generate_data(npoints, 1)
    dump_in_csv(npoints, data, ["x2", "y2", "z2", "w2"])


if __name__ == "__main__":
    main()
