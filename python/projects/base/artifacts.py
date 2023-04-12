import mne
import numpy as np

def detect_squid_jumps(raw, time=0.1, threshold=4, description = "bad_jump"):
    """

    """

    samps = int(time*raw.info["sfreq"])
    if samps % 2 == 1:
        samps += 1

    # Detection kernel
    kernel = np.concatenate((np.ones(1), -np.ones(1)))

    assert len(kernel) % 2 == 0
    hkl = int(len(kernel)/2) # half kernel length

    meg = mne.pick_types(raw.info, meg=True)
   # data = raw.get_data()[meg]
   # data -= data.mean(1)[:,None]
   # data /= data.std(1)[:,None]

    all_jumps = np.zeros(len(raw.times),dtype=bool)

    for x in raw.get_data()[meg]:

        x -= x.mean()
        x /= x.std()

        # Filter data
        #y = np.correlate(x, kernel, "valid")
        y = np.convolve(x, kernel, "valid")

        # zero pad in pre and post
        y = np.concatenate((np.zeros(hkl), y, np.zeros(hkl-1)))

        # Threshold the filtered signal
        thr1, thr2 = np.percentile(y, [1, 99])
        thr1 *= threshold
        thr2 *= threshold
        y = (y < thr1) | (y > thr2)

        # Expansion kernel
        # Convolve to cover the requested samples around the artifact
        kernel2 = np.ones(samps+1)
        assert len(kernel2) % 2 == 1

        y = np.convolve(y, kernel2, "valid") >= 1

        if any(y):
            hlk2 = int((len(kernel2)-1)/2)

            # zero pad pre and post
            y = np.concatenate((np.zeros(hlk2), y, np.zeros(hlk2) )).astype(bool)

            all_jumps = all_jumps | y

    # Annotate raw
    jumps = np.where(np.concatenate(([0], np.diff(all_jumps.astype(int)))))[0]
    assert len(jumps)%2 == 0

    jumps = (jumps+raw.first_samp) / raw.info["sfreq"]
    for onset, offset in jumps.reshape(-1,2):
        skip = False
        # check if segment is already included in an annotation
        for a in raw.annotations:
            if onset >= a['onset'] and onset <= a['onset']+a['duration'] or \
               offset >= a['onset'] and offset <= a['onset']+a['duration']:
               skip = True
               break
        # onset : time in seconds relative to first_samp (!)
        # duration : time in seconds
        #if raw.annotations is None:
        #    raw.annotations = mne.Annotations(onset, offset-onset, description)
        #else:
        if not skip:
            raw.annotations.append(onset, offset-onset, description)
