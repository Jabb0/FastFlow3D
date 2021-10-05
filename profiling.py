import numpy as np
import torch

from models.Flow3DModel import Flow3DModel
from models.FastFlow3DModelScatter import FastFlow3DModelScatter

torch.manual_seed(0)


def create_random_data(n_points, n_features):
    pc_pos = torch.randint(low=-85, high=85, size=(1, n_points, 3)).cuda()
    pc_f = torch.randn(size=(1, n_points, n_features)).cuda()
    pc = torch.cat([pc_pos, pc_f], dim=-1)
    grid = torch.randint(low=0, high=10000, size=(1, n_points)).cuda()
    mask = torch.ones(size=(1, n_points)).long().cuda()
    return pc, grid, mask


def run():
    import time
    points = [10000, 50000, 100000, 250000, 1000000]
    architectures = ['FastFlowNet', 'FlowNet']
    n_forward_passes = 100

    f = open('timing.txt', 'w')
    for arch in architectures:
        s = "Time measurement for architecture {}:".format(arch)
        for n_points in points:
            print("starting profiling of {} with {} points".format(arch, n_points))
            if arch == 'FastFlowNet':
                model = FastFlow3DModelScatter(n_pillars_x=512, n_pillars_y=512).eval().cuda()
                n_features = 5
            elif arch == 'FlowNet':
                model = Flow3DModel().eval().cuda()
                n_features = 2
            else:
                raise ValueError("Unknown architecture {}".format(arch))

            times = list()
            try:
                for i in range(n_forward_passes):
                    prev_pc = create_random_data(n_points, n_features=n_features)
                    cur_pc = create_random_data(n_points, n_features=n_features)
                    x = (prev_pc, cur_pc)
                    torch.cuda.synchronize()
                    t = time.time()
                    with torch.no_grad():
                        model(x)
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - t
                    if i > 9:
                        times.append(elapsed_time * 1000)  # convert sec to ms
                mean_time = np.mean(np.array(times))
                s += "\n\t{}: ".format(n_points) + "{:.2f}ms".format((float(mean_time)))
            except RuntimeError:
                s += "\n\t{}: OOM".format(n_points)
            f.write(s)
        print(s)
    f.close()


if __name__ == '__main__':
    run()


