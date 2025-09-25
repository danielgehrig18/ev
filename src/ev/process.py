import numpy as np


def compute_frame(output):
    rotation_frames = np.stack([
        output["d1f"][:, :3],
        output["d2f"][:, :3],
        output["d3f"][:, :3]
    ], axis=1)

    translation_frames = np.stack([
        output["d1f"][:, 3:],
        output["d2f"][:, 3:],
        output["d3f"][:, 3:]
    ], axis=1)

    rotation_frames_orth = gram_schmidt_orthogonalization(rotation_frames)
    translation_frames_orth = gram_schmidt_orthogonalization(translation_frames)

    return rotation_frames_orth, translation_frames_orth


def gram_schmidt_orthogonalization(frames, eps=1e-9):
    # ... x n x n
    n = frames.shape[-1]
    for i in range(n):
        f = frames[...,i,:]
        f_u = frames[...,:i,:]
        f -= np.einsum("...ji,...jk,...k->...i", f_u, f_u, f)
        f /= (np.linalg.norm(f, axis=-1, keepdims=True) + eps)
        frames[...,i,:] = f
    return frames
