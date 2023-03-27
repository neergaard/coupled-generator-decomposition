
import numpy as np
import torch
import matplotlib.pyplot as plt

def moving_average(a, n=3) :
    return np.convolve(a, np.ones(n), 'same') / n

def visualize_AA_trajectory(S,type='full'):

    times = torch.load('data/MEEGtimes.pt')
    K = S.shape[-2]

    t0 = np.zeros(8,dtype=np.int16)
    for ticks in range(8):
        t0[ticks] = int(np.argmin(abs(times-ticks/10)))
    
    smooth_val  = 10;
    deltaPhi    = 360/K;
    orig        = [0,0];

    #leg = ['Famous, EEG','Unfamiliar, EEG','Scrambled, EEG','Famous, MEG','Unfamiliar, MEG','Scrambled, MEG'...
    #,'Start point','Stimulus','100ms markers','End point'

    colors = np.array([[134/256,   203/256,    146/256,    .75],[0,         0,          0,          .75],[203/256,   134/256,    146/256,    .75]])
    style  = ['-', '--']
    
    cors = np.zeros((K,2))
    for idx,j in enumerate(np.arange(0,360-deltaPhi+1,deltaPhi)):
        cors[idx] = np.array((np.sin(j*np.pi/180),np.cos(j*np.pi/180)))

    plt.figure()
    plt.plot(np.append(cors[:,0],0),np.append(cors[:,1],1),'k-')

    count = 1
    condstartsS = [0,180,360,540]

    
    for m in range(S.shape[0]):
        for l in range(3):
            if type=='full':
                s = torch.mean(S[m,:,l],axis=0)
            elif type=='concatenated':
                s = torch.mean(S[m,:,:,condstartsS[l]:condstartsS[l+1]],axis=0);
            
            s_smooth = np.zeros(s.shape)
            for k in range(K):
                s_smooth[k] = moving_average(s[k],n=smooth_val)
            # circshift?
            s_smooth = np.roll(s_smooth,2,axis=0)
            projection = s_smooth.T@cors

            # plot start point
            plt.plot(projection[0,0],projection[0,1],marker='.',color=colors[l])
            plt.plot(projection[-1,0],projection[-1,1],marker='x',color=colors[l])
            plt.plot(projection[:,0],projection[:,1],linestyle=style[m],color=colors[l])

            # baseline as triangle
            plt.plot(projection[t0[0],0],projection[t0[0],1],marker='2',color=colors[l])

            for tick in range(7):
                p1 = np.array((projection[t0[tick+1],0],projection[t0[tick+1],1]))
                p2 = np.array((projection[t0[tick+1]+1,0],projection[t0[tick+1]+1,1]))
                v = p2-p1
                x = p1[0]+0.5*v[0]
                y = p1[1]+0.5*v[1]
                v = v/np.linalg.norm(v)/50
            plt.plot(np.array((x+v[1], x-v[1])),np.array((y-v[0], y+v[0])),color=colors[l])
plt.axis('off')
