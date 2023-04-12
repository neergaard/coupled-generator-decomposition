import sys
import itertools

import mne

from projects.anateeg.utils import parse_args

from projects.facerecognition_dtu import utils

def inverse_subject(subject_id):
    io = utils.SubjectIO(subject_id)
    io.data.update(stage="inverse")
    io.data.path.ensure_exists()

    conditions = ("famous", "unfamiliar", "scrambled")
    splits = (0, 1)
    for condition, i in itertools.product(conditions, splits):
        io.data.update(condition=condition, split=str(i))

        evo_kwargs = dict(stage="preprocess", processing="p", suffix="evo")
        cov_kwargs = dict(stage="preprocess", processing="p", suffix="cov")
        fwd_kwargs = dict(condition=None, split=None, stage="forward", forward="mne", suffix="fwd")
        morph_kwargs = dict(condition=None, split=None, stage="forward", forward="mne", suffix="morph", extension="h5")

        evoked = mne.read_evokeds(io.data.get_filename(**evo_kwargs))[0]
        noise_cov = mne.read_cov(io.data.get_filename(**cov_kwargs, space="noise"))
        data_cov = mne.read_cov(io.data.get_filename(**cov_kwargs, space="data"))
        fwd = mne.read_forward_solution(io.data.get_filename(**fwd_kwargs))
        morph = mne.read_source_morph(io.data.get_filename(**morph_kwargs))

        ch_types = dict(
            meg=[evoked.info["ch_names"][ch] for ch in mne.pick_types(evoked.info, meg=True)],
            eeg=[evoked.info["ch_names"][ch] for ch in mne.pick_types(evoked.info, eeg=True)],
        )

        for ch_type, channels in ch_types.items():
            evoked_ch = evoked.copy()
            filters = mne.beamformer.make_lcmv(
                evoked_ch.pick_channels(channels).info,
                fwd.copy().pick_channels(channels),
                data_cov.copy().pick_channels(channels),
                noise_cov=noise_cov.copy().pick_channels(channels),
                # pick_ori=None, # max-power, normal
                # weight_norm="nai",
                rank=None,
            )
            stc = mne.beamformer.apply_lcmv(evoked, filters)
            stc.save(io.data.get_filename(forward="mne", channel=ch_type, suffix="stc", extension=None), overwrite=True)

            stc_morphed = morph.apply(stc)
            stc_morphed.save(io.data.get_filename(forward="mne", channel=ch_type, space="fsaverage", suffix="stc", extension=None), overwrite=True)

s = mne.setup_source_space("fsaverage", spacing="ico5", add_dist="patch", subjects_dir=Config.path.FREESURFER)
root = Path("/mrhome/jesperdn/INN_JESPER/projects/facerecognition_dtu/data")


sub = "sub-08"
stc = mne.read_source_estimate(root / f"{sub}/ses-meg/stage-inverse/task-facerecognition_space-fsaverage_cond-famous_fwd-mne_ch-eeg_split-0_stc")
# stc = mne.read_source_estimate(root / f"{sub}/ses-meg/stage-inverse/task-facerecognition_space-fsaverage_cond-famous_fwd-mne_ch-meg_split-0_stc")

m0 = pv.make_tri_mesh(s[0]["rr"][s[0]["inuse"].astype(bool)], s[0]["use_tris"])
m1 = pv.make_tri_mesh(s[1]["rr"][s[1]["inuse"].astype(bool)], s[1]["use_tris"])

idx1, idx2 = np.where(stc.data==stc.data.max())
print(f"{stc.times[idx2][0]*1e3} ms")
p = pv.Plotter(notebook=False)
p.add_mesh(m0, scalars=stc.lh_data[:,idx2])
p.add_mesh(m1, scalars=stc.rh_data[:,idx2])
p.show()

if __name__ == "__main__":
    args = parse_args(sys.argv)
    subject_id = getattr(args, "subject-id")
    inverse_subject(subject_id)