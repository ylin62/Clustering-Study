import os
from functools import reduce
from collections import OrderedDict
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import seaborn as sns
from joblib import Parallel, delayed

LW=2
FONTSIZE=15
FIGSIZE=(10, 6)
MARKER = 10
ARROWLENGTH = 2.5
HW = 0.8
LIM = 50

def make_video(target, freq, path):
        
    cars = reduce(np.union1d, [item[:, -1] for item in target])
    colors = sns.hls_palette(len(cars), l=.3, s=.8)
    carsize = np.array([[4.93/2, -4.93/2, -4.93/2, 4.93/2, 4.93/2], 
                        [1.864/2, 1.864/2, -1.864/2, -1.864/2, 1.864/2]])
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.xaxis.offsetText.set_fontsize(FONTSIZE)
    ax.yaxis.offsetText.set_fontsize(FONTSIZE)
    
    def make_frame(t):
        i = int(t * freq)
        ax.clear()
        centerx, centery = [], []
        for s, item in enumerate(target[i]):
            h = np.deg2rad(item[4])
            rot = np.array([[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]])
            ax.plot(np.dot(rot, carsize)[0] + item[0], np.dot(rot, carsize)[1] + item[1], '-', lw=LW, color=colors[s], 
                    label=int(item[-1]))
            centerx += [item[0]]
            centery += [item[1]]
        ax.axis('equal')
        ax.set_xlabel("X [m]", fontsize=15)
        ax.set_ylabel("Y [m]", fontsize=15)
        fig.legend(fontsize=FONTSIZE)
        ax.set_xlim([np.average(centerx)-LIM, np.average(centerx)+LIM])
        ax.set_ylim([np.average(centery)-LIM, np.average(centery)+LIM])
        
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=len(target)/freq)    
    # animation.write_videofile(path+'.mp4', fps=20)
    animation.write_gif(path+'.gif', fps=20)
    

class ClusterTrack(AgglomerativeClustering):
    """docstring for ClusterTrack"""
    def __init__(self, data, freq, distance, duration=1., sv=None, linkage='complete'):
        super(ClusterTrack, self).__init__(n_clusters=None, compute_full_tree=True, 
                                           linkage=linkage, distance_threshold=distance)
        
        self.data = data
        self.freq = freq
        self.duration = duration
        self.sv = sv
        
        self.n_veh = data['n_veh']
        self.raw_data = data['data']
        
    def main(self, path=os.path.expanduser('~/Documents/Clustering-Study/imgs/')):
        
        tracked = self.tracking()
        if not os.path.exists(path):
            os.makedirs(path)
            
        Parallel(n_jobs=os.cpu_count())(delayed(make_video)(target, self.freq, path+'cluster_'+str(i)) 
                                        for i, target in enumerate(tracked))
    
    def clusters(self, idx):
        
        clusters = OrderedDict()
        
        if self.sv is None:
            points = np.array([np.append(item[0:2], item[4]) for item in self.raw_data[idx]])
            self.fit(points)
            
            for i in range(self.n_clusters_):
                points_in_cluster = np.where(self.labels_==i)[0]
                if points_in_cluster.size > 1:
                    cars = np.array([self.raw_data[idx][j] for j in points_in_cluster])
                    clusters[tuple(cars[:, -1])] = cars
                    
        else:
            raise NotImplementedError('not developed yet\n')
                    
        return clusters
    
    def tracking(self):
        
        tracking, tracked = [], []
        
        for i in range(len(self.raw_data)):
            print('progress: {:.1f}%'.format(i/len(self.raw_data)*100), end='\r')
            new_tracking = []
            targets_pool = list(range(len(tracking)))
            cluster = self.clusters(i)
            for key, values in cluster.items():
                found = False
                for j, target in enumerate(tracking):
                    if np.count_nonzero(np.in1d(key, reduce(np.union1d, [item[:, -1] for item in target]))==0) <= 1 \
                        and np.count_nonzero(np.in1d(reduce(np.union1d, [item[:, -1] for item in target]), key)==0) <= 1:
                        tracking[j] += [values]
                        targets_pool.remove(j)
                        found = True
                        break
                if not found:
                    new_tracking += [[values]]
                    
            temp = tracking.copy()
            for s in targets_pool:
                if len(tracking[s]) >= self.duration*self.freq:
                    tracked += [tracking[s]]
            for s in targets_pool:
                tracking.remove(temp[s])
            tracking.extend(new_tracking)
        for item in tracking:
            if len(item) >= self.duration*self.freq:
                tracked.append(item)
        
        return tracked
    
if __name__ == "__main__":
    import pickle
    
    with open(os.getcwd() + '/data.pkl', 'rb') as p:
        data = pickle.load(p)
        
    Demo = ClusterTrack(data, freq=25, distance=100., duration=2., linkage='complete')
    
    Demo.main()