"""
Process Ninapro original data -> per-subject+per-exercise static-hold postures.

Improvements over existing pre_processing.py:
1. Uses 'glove' field (not 'angles') - no NaN
2. Keeps ALL movement labels
3. Extracts static-hold phase (last 25% frames)
4. Saves per-subject-per-exercise files
5. Correctly handles variable-length trials via interpolation to 100 frames
6. Preserves exercise info for group separation
"""
import os, glob
import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import argparse

TARGET_FRAMES = 100
LAST_FRAMES_FRAC = 0.25
OUTPUT_DIR = 'Ninapro Data/preprocessed'

# Exercise codes for restimulus label ranges
# E1: labels 1-12 (basic finger movements), subjects S28-S67
# E2: labels 1-17 (hand + wrist), subjects S1-S27, S68-S77
# E3: labels 1-23 (grasping+functional), same subjects as E2


def get_exercise_from_filepath(fpath):
    """Determine exercise from filename (E1, E2, E3)."""
    basename = os.path.basename(fpath)
    part = basename.split('_')[1]  # e.g., E1
    return part


def extract_trials_from_raw(mat):
    """Extract trials from raw Ninapro .mat, return resampled (N, 100, 22)."""
    glove_raw = mat['glove']
    restimulus = mat['restimulus'].flatten()

    diff = np.diff(restimulus, prepend=0)
    starts = np.where(diff > 0)[0]
    ends = np.where(diff < 0)[0]

    if len(starts) > len(ends):
        ends = np.append(ends, len(restimulus))
    elif len(ends) > len(starts):
        starts = np.insert(starts, 0, 0)

    trials, labels = [], []
    for s, e in zip(starts, ends):
        if e - s < 5:
            continue
        label = restimulus[s]
        if label == 0:
            continue
        segment = glove_raw[s:e, :].astype(np.float64)
        x_old = np.linspace(0, 1, len(segment))
        x_new = np.linspace(0, 1, TARGET_FRAMES)
        f_interp = interp1d(x_old, segment, axis=0, kind='linear',
                            bounds_error=False, fill_value=(segment[0], segment[-1]))
        resampled = f_interp(x_new)
        trials.append(resampled)
        labels.append(label)

    if not trials:
        return np.zeros((0, TARGET_FRAMES, 22)), np.array([])
    return np.stack(trials), np.array(labels, dtype=int)


def extract_static_postures(trial_data, last_frac=LAST_FRAMES_FRAC):
    """Mean of last fraction as static-hold posture."""
    n_last = max(5, int(TARGET_FRAMES * last_frac))
    return np.mean(trial_data[:, -n_last:, :], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exercises', nargs='+', default=['E1', 'E2', 'E3'])
    parser.add_argument('--subjects', nargs='+', default=None)
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find files
    all_files = []
    for ex in args.exercises:
        files = sorted(glob.glob(f'Ninapro Data/original/S*_{ex}_*.mat'))
        all_files.extend(files)

    if args.subjects:
        subjects_set = set(args.subjects)
        all_files = [f for f in all_files
                     if os.path.basename(f).split('_')[0] in subjects_set]

    print(f"Processing {len(all_files)} files...")

    # Process each file individually
    per_file_data = {}  # key: (subject, exercise) -> (angles, labels)

    for fpath in sorted(all_files):
        basename = os.path.basename(fpath)
        subject = basename.split('_')[0]
        exercise = basename.split('_')[1]

        mat = sio.loadmat(fpath)
        trial_data, trial_labels = extract_trials_from_raw(mat)

        if len(trial_data) == 0:
            print(f"  {basename}: no valid trials, skipping")
            continue

        static_poses = extract_static_postures(trial_data)

        key = (subject, exercise)
        if key in per_file_data:
            per_file_data[key] = (
                np.vstack([per_file_data[key][0], static_poses]),
                np.concatenate([per_file_data[key][1], trial_labels])
            )
        else:
            per_file_data[key] = (static_poses, trial_labels)

        print(f"  {basename}: {len(trial_data)} trials -> {len(static_poses)} static")

    # Save files
    # 1. Per-subject-per-exercise
    # 2. Combined per-subject (all exercises)
    # 3. Combined across all subjects

    # Per-subject/per-exercise
    for (subj, ex), (angles, labels) in sorted(per_file_data.items()):
        fname = f'{subj}_{ex}_preprocessed.mat'
        save_path = os.path.join(OUTPUT_DIR, fname)
        sio.savemat(save_path, {
            'angles': angles,
            'trial_tasks': labels,
            'exercise': ex,
            'num_trials': len(angles),
            'num_tasks': len(np.unique(labels)),
        })
        print(f"  Saved: {fname}")

    # Per-subject (all exercises combined)
    by_subject = {}
    for (subj, ex), (angles, labels) in per_file_data.items():
        if subj not in by_subject:
            by_subject[subj] = []
        by_subject[subj].append((angles, labels))

    for subj, data_list in sorted(by_subject.items()):
        all_angles = np.vstack([d[0] for d in data_list])
        all_labels = np.concatenate([d[1] for d in data_list])
        fname = f'{subj}_preprocessed.mat'
        save_path = os.path.join(OUTPUT_DIR, fname)
        sio.savemat(save_path, {
            'angles': all_angles,
            'trial_tasks': all_labels,
            'num_trials': len(all_angles),
            'num_tasks': len(np.unique(all_labels)),
        })

    # Combined across all subjects
    all_angles_list, all_labels_list = [], []
    for (subj, ex), (angles, labels) in per_file_data.items():
        all_angles_list.append(angles)
        all_labels_list.append(labels)

    if all_angles_list:
        combined_angles = np.vstack(all_angles_list)
        combined_labels = np.concatenate(all_labels_list)
        save_path = os.path.join(OUTPUT_DIR, 'all_participants_preprocessed.mat')
        sio.savemat(save_path, {
            'angles': combined_angles,
            'trial_tasks': combined_labels,
            'num_trials': len(combined_angles),
            'num_tasks': len(np.unique(combined_labels)),
        })
        print(f"\nSaved combined: {save_path} ({len(combined_angles)} total, "
              f"{len(np.unique(combined_labels))} unique tasks)")

    print("Done!")


if __name__ == '__main__':
    main()
