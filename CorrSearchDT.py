import sys
import numpy as np
import cv2
from skvideo.io import vread
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsRegressor
from scipy.signal import medfilt
from scipy.stats import scoreatpercentile
import os
from subprocess import getoutput
import json
import csv
import re
import warnings
from tqdm import tqdm


class misc():
    u"""Some helper functions
    """

    def medfilt_2d(self, array, twin=5):
        u"""Median filtering nd array
        # columns of array should be 2.
        """

        for d in range(array.shape[1]):
            array[:, d] = medfilt(array[:, d], twin)

        return array

    def calc_piecewise_constant(self, array, n_piece=8):
        u"""Piecewise constant approximation.
        # columns of array should be 2.
        # rows of array should be propotional to n.
        """

        array_normed = (array - array.mean(0)) / array.std(0)
        N = len(array_normed)
        w = N / n_piece
        pc = array_normed.reshape(n_piece, np.int(w), 2).mean(axis=1)
        pc_var = (pc ** 2).mean()

        return np.hstack((pc[:, 0], pc[:, 1], pc_var))

    def load_csv(self, csv_fname):
        csvdata = np.loadtxt(csv_fname, delimiter=',', dtype='str')
        return csvdata


class Trajectory(misc):
    u"""Trajectory class
    """

    def __init__(self, loc, first_frame, min_trajectory_length,
                 max_trajectory_length, n_piece, frame_shape,
                 video_dirname, traj_dirname):
        self.last_loc = loc
        self.first_frame = first_frame
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length
        self.n_piece = n_piece
        self.frame_shape = frame_shape
        self.video_dirname = video_dirname
        self.traj_dirname = traj_dirname
        self.traj_fname = '%04d_%04d_%04d.json' % (loc[0], loc[1], first_frame)
        self.margin = 2  # HACK

        self.loc = []
        self.hsv = []
        self.lm = []
        self.length = 0

    def append(self, loc, hsv, lm, e):
        u""" Appending tracking results
        """

        if(self.length > self.max_trajectory_length):
            self.finalize(True)
            return False

        W, H = self.frame_shape
        ind0 = loc[0] < self.margin
        ind1 = loc[1] < self.margin
        ind2 = (loc[0] >= W - self.margin)
        ind3 = (loc[1] >= H - self.margin)
        ind4 = ~e
        if(ind0 | ind1 | ind2 | ind3 | ind4):
            self.finalize(self.length >= self.min_trajectory_length)
            return False

        if(self.length != -1):
            loc_rounded = np.round(loc).astype('int')
            self.loc.append(loc_rounded)
            self.hsv.append(hsv)
            self.lm.append(lm)
            self.last_loc = loc_rounded
            self.length += 1

        return True

    def finalize(self, is_sufficient_length):
        u"""Finalize trajecotry data.
        Storing results if trajectory has sufficient length.
        """

        if(is_sufficient_length is False):
            return -1

        self.loc = np.array(self.loc)
        self.hsv = np.array(self.hsv)
        self.lm = self.medfilt_2d(np.array(self.lm))
        modval = np.mod(len(self.lm), np.max(self.n_piece))
        if(modval != 0):
            self.lm = self.lm[:-modval, :]
            self.loc = self.loc[:-modval, :]
            self.hsv = self.hsv[:-modval, :]
            self.length -= modval

        # Full data
        fname = '%s/%s.full' % (self.traj_dirname, self.traj_fname)
        lm_abs = np.sqrt(np.sum(self.lm ** 2, 1))
        feat = np.hstack((self.hsv.mean(0), self.hsv.std(0),
                          lm_abs.mean(0), lm_abs.std(0), self.length))
        dat = {'first_frame': np.int(self.first_frame), 'length': np.int(self.length),
               'loc': self.loc.astype('int32').tolist(), 'feat': feat.tolist(),
               'lm': self.lm.tolist(), 'video_dirname': self.video_dirname}
        json.dump(json.dumps(dat), open(fname, 'w'))

        # Motion data
        fname = '%s/%s' % (self.traj_dirname, self.traj_fname)
        dat = {'first_frame': np.int(self.first_frame), 'length': np.int(self.length),
               'feat': feat.tolist(), 'lm': self.lm.tolist()}
        json.dump(json.dumps(dat), open(fname, 'w'))

        # Piecewise constant
        for n_ in self.n_piece:
            piece_const = self.calc_piecewise_constant(self.lm, n_piece=n_)
            with open('%s_pc%02d.csv' % (self.traj_dirname, n_), 'a') as fp:
                writer = csv.writer(fp, delimiter=',')
                row1 = ['%s/%s' % (os.path.abspath(self.traj_dirname),
                                   self.traj_fname),
                        '%d' % self.first_frame, '%d' % self.length]
                row2 = ['%0.2f' % x for x in piece_const]
                writer.writerow(np.hstack((row1, row2)))


class CorrSearch(misc):
    def __init__(
            self,
            step=4,  # strides to sample trajectories
            t_step=4,  # temporal strides to sample trajectories
            min_trajectory_length=64,  # minimum length of trajectories
            max_trajectory_length=2048,  # maximum length of trajectories
            eigen_th=1e-4,  # eigenvalue threshold to find good feature to track
            n_piece=[64],  # number of pieces for approximation
            frame_shape=[320, 180]  # size of images
            ):
        self.step = step
        self.t_step = t_step
        self.min_trajectory_length = min_trajectory_length
        self.max_trajectory_length = max_trajectory_length
        self.eigen_th = eigen_th
        self.n_piece = n_piece
        self.frame_shape = frame_shape

    def localize_target(self, tar_pattern, obs_pattern,
                        dump_dir='/Users/yonetani/dump/example',
                        clsf_name='model.npy'):
        query = self.compute_query(tar_pattern)
        self.compute_trajectory(obs_pattern, dump_dir)
        result = self.estimate_trajectory_targetness(query,
                                                     '%s_pc%d.csv'
                                                     % (dump_dir,
                                                        self.n_piece[0]),
                                                     ub_th=10,
                                                     clsf_fname=clsf_name)
        tarmap, _, _ = self.estimate_pixel_targetness(result, [0])

        return tarmap

    def compute_trajectory(self, video_pattern, traj_dirname, verbose=False, force_overwrite=False):
        u""" Calculating dense trajectories and save them as npy file
        """

        if(force_overwrite):
            getoutput('rm -rf "%s"' % traj_dirname)
            getoutput('rm -rf %s*.csv' % traj_dirname)
            getoutput('mkdir -p "%s"' % traj_dirname)
        else:
            if(os.path.exists(traj_dirname)):
                print('Directory exists! If you want to overwrite the trajectory directory, change force_overwrite option to True')
                return 0

        imgs = vread(video_pattern)
        W, H = self.frame_shape
        w_grid, h_grid = np.meshgrid(range(W), range(H))
        p_grid = np.vstack((w_grid.flatten(), h_grid.flatten())).T
        video_dirname = os.path.dirname(video_pattern)

        trajlist = []
        img = cv2.resize(imgs[0], (W, H))
        img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for t in tqdm(range(len(imgs)), desc='Generating trajectories'):
            prev_img_g = img_g
            img = cv2.resize(imgs[t], (W, H))
            img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            flow = cv2.calcOpticalFlowFarneback(
                prev_img_g, img_g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = cv2.medianBlur(flow, 5)
            p_dst = p_grid + flow.reshape((W * H, 2))
            is_texture = cv2.cornerMinEigenVal(prev_img_g, 5) > self.eigen_th

            _, hom = self.__calc_gm(img_g, prev_img_g)
            lm = p_dst - cv2.perspectiveTransform(
                p_grid[np.newaxis].astype('float64'), hom)[0]
            lm[:, 1] *= -1

            # Initializing trajectories
            if(np.mod(t, self.t_step) == 0):
                init_mask = is_texture.copy()
                for x in trajlist:
                    init_mask[(x.last_loc[1] - self.step):
                              (x.last_loc[1] + self.step),
                              (x.last_loc[0] - self.step):
                              (x.last_loc[0] + self.step)] = False

                w_grid, h_grid = np.meshgrid(range(0, W, self.step),
                                             range(0, H, self.step))
                p0 = np.vstack((w_grid.flatten(), h_grid.flatten())).T
                p0 = p0[init_mask[p0[:, 1], p0[:, 0]], :]
                trajlist += [Trajectory(loc, t,
                                        self.min_trajectory_length,
                                        self.max_trajectory_length,
                                        self.n_piece,
                                        self.frame_shape,
                                        video_dirname, traj_dirname)
                             for loc in p0]

            # Updating trajectories
            is_alive = [x.append(p_dst[x.last_loc[0] + x.last_loc[1] * W],
                                 img_hsv[x.last_loc[1], x.last_loc[0], :],
                                 lm[x.last_loc[0] + x.last_loc[1] * W],
                                 is_texture[x.last_loc[1], x.last_loc[0]])
                        for x in trajlist]
            trajlist = [x for (x, f) in zip(trajlist, is_alive) if f is True]
            if(verbose):
                img_ = img.copy()
                for x in trajlist:
                    cv2.circle(img_, (np.int(x.last_loc[0]),
                                      np.int(x.last_loc[1])),
                               3, (255, 0, 0))
                cv2.imshow('Video', img_[:, :, ::-1])
                cv2.waitKey(1)

        [x.finalize(x.length >= x.min_trajectory_length) for x in trajlist]
        if(verbose):
            cv2.destroyAllWindows()

    def compute_query(self, video_pattern):
        u"""Generating global motion queries
        """

        imgs = vread(video_pattern)
        W, H = self.frame_shape

        query = []
        img = imgs[0]
        img = cv2.resize(imgs[0], (W, H))
        img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for t in tqdm(range(len(imgs)), desc='Generating query'):
            prev_img_g = img_g
            img = cv2.resize(imgs[t], (W, H))
            img_g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gm, _ = self.__calc_gm(img_g, prev_img_g)
            query.append(gm)

        query = self.medfilt_2d(np.array(query))

        return query

    def estimate_trajectory_targetness(self, query, pc_fname, ub_th=10,
                                       clsf_fname='model.npy'):
        u""" Estimating targetness for each trajectory.
        100 - ub_th percentile trajectories are fully evaluated.
        """

        clsf, ss = np.load(clsf_fname)

        pc_data = self.load_csv(pc_fname)
        end = pc_data[:, 1].astype('int') + pc_data[:, 2].astype('int')
        pc_data = pc_data[end < len(query), :]

        ub = self.__calc_upperbound(query, pc_data)
        ub = 1. / (1 + np.exp(-ub))
        ub[ub < scoreatpercentile(ub, 100 - ub_th)] = -1

        targetness_list = []
        traj_fname = pc_data[ub > -1, 0]
        if(len(traj_fname) == 0):
            return targetness_list

        for x in tqdm(traj_fname, desc='Estimating targetness'):
            traj_data = json.loads(json.load(open(x)))
            c = self.__calc_correlation(query, traj_data)
            c = 1. / (1 + np.exp(-c))
            p = clsf.predict_proba(ss.transform(np.array(traj_data['feat']).reshape(1, -1)))[0, 1]
            targetness_list.append(np.hstack([x, p * c, c, p, traj_data['length']]))
        targetness = np.array(targetness_list)

        return targetness

    def estimate_pixel_targetness(self, targetness, frame_list):
        W, H = self.frame_shape
        pts_array = []
        for x in tqdm(targetness, desc='Loading results'):
            traj_data = json.loads(json.load(open(x[0] + '.full')))
            loc = np.array(traj_data['loc'])
            fr = np.arange(traj_data['first_frame'],
                           traj_data['first_frame'] + traj_data['length']
                           )[:, np.newaxis]
            tar = np.ones(len(fr))[:, np.newaxis] * x[1:].astype('float')
            pts_array.append(np.hstack((loc, fr, tar)))
        pts_array = np.vstack(pts_array)
        x, y = np.meshgrid(range(W), range(H))
        pts_array_dense = np.vstack((x.flatten(), y.flatten())).T
        target_map = []
        corr_map = []
        prior_map = []

        for t in tqdm(frame_list, desc='Adopting KNN Regressor'):
            if(np.all(pts_array[:, 2] != t)):
                tar = np.zeros((H, W, 3))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    knn = KNeighborsRegressor(n_neighbors=1,
                                              weights=self.__distfunc)
                    knn.fit(pts_array[pts_array[:, 2] == t, :2],
                            pts_array[pts_array[:, 2] == t, 3:6])
                    tar = knn.predict(pts_array_dense).reshape(H, W, 3)
            tar[np.isnan(tar)] = 0
            target_map.append(tar[:, :, 0])
            corr_map.append(tar[:, :, 1])
            prior_map.append(tar[:, :, 2])
        target_map = np.dstack(target_map)
        corr_map = np.dstack(corr_map)
        prior_map = np.dstack(prior_map)

        return target_map # , corr_map, prior_map  # uncomment them if you want to see baseline results

    def __calc_gm(self, img_g, prev_img_g):
        lk_params = dict(winSize=(5, 5), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS
                                   | cv2.TERM_CRITERIA_COUNT, 5, 0.03))

        p0 = cv2.goodFeaturesToTrack(prev_img_g,
                                     150, 1e-3, 5).astype('float32').squeeze()

        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_img_g, img_g,
                                               p0, None, **lk_params)

        if(len(p0) < 10):
            return np.zeros(2)

        hom = cv2.findHomography(p0, p1)[0]
        p1_ = cv2.perspectiveTransform(np.array([p0]), hom)[0]
        gm = np.mean(p1_ - p0, 0)

        return gm, hom

    def __calc_correlation(self, query, traj_data):
        u""" Calculating correlations
        """

        gm = query[traj_data['first_frame']: (traj_data['first_frame'] +
                                              traj_data['length']), :]
        lm = np.array(traj_data['lm'])

        if(len(gm) != len(lm)):
            return 0

        corr = np.mean(np.array([np.corrcoef(gm[:, i], lm[:, i])[0, 1]
                                 for i in range(2)]))

        return corr

    def __calc_upperbound(self, query, pc_data):
        u""" Calculating upperbounds of correlations using piecewise constants
        """

        n_piece = np.int((pc_data.shape[1] - 4) / 2)
        begin = pc_data[:, 1].astype('int')
        end = begin + pc_data[:, 2].astype('int')
        _, idx, ridx = np.unique(begin * 1e5 + end,
                                 return_index=True, return_inverse=True)
        pc_g = np.array([self.calc_piecewise_constant(query[x:y, :],
                                                      n_piece)
                         for (x, y) in zip(begin[idx], end[idx])])

        corr = np.array([
            np.corrcoef(pc_g[ridx[i], 0:n_piece],
                        pc_data[i, 3:(3 + n_piece)].astype('float'))[0, 1] +
            np.corrcoef(pc_g[ridx[i], n_piece:n_piece * 2],
                        pc_data[i, (3 + n_piece):
                                (3 + n_piece * 2)].astype('float'))[0, 1]
            for i in range(len(pc_data))]) / 2.

        pc_var_g = pc_g[ridx, -1]
        pc_var_l = pc_data[:, -1].astype('float')
        ub = 1. - (pc_var_g + pc_var_l) / 2 + corr

        return ub

    def __distfunc(self, dist):
        th = self.step * 2
        ret = np.ones_like(dist, np.float)
        ret[dist > th] = 0.
        return ret
