import numpy as np
import torch as th
import rdel.lbvh_bind as lb
from tetutils import compute_cc, Grid
from intersect import ti_init, upload_bvh_and_mesh, query_segments_any_hit
import trimesh
import time
import logging
from scipy.spatial import Delaunay
import igl

def run(input_verts: np.ndarray, input_faces: np.ndarray, verbose: bool = False, orient: bool = False):
    """
    Run Restricted Delaunay Triangulation on the input mesh and save the result to the output path.
    
    @ param orient: Whether to orient face normals of the output faces.
    """
    # Initialize Taichi
    ti_init()

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load mesh
    verts = input_verts
    faces = input_faces
    logging.info(f"[Mesh Loaded] #V: {verts.shape[0]}, #F: {faces.shape[0]}")

    # Build BVH
    start_time = time.time()
    bvh = lb.build_lbvh(verts, faces)
    end_time = time.time()
    logging.info(f"[BVH Build] {end_time - start_time} sec | #Internal Nodes: {bvh['n_internal']}, #Leaves: {bvh['n_leaves']}")


    # Upload BVH and mesh
    start_time = time.time()
    upload_bvh_and_mesh(bvh, verts, faces)
    end_time = time.time()
    logging.info(f"[BVH Upload] {end_time - start_time} sec")


    # Delaunay Tetrahedralization
    start_time = time.time()
    _verts = th.from_numpy(verts).to(dtype=th.float32)
    dt = Delaunay(_verts)
    _tets = th.from_numpy(dt.simplices).to(dtype=th.long)
    end_time = time.time()
    logging.info(f"[Delaunay Tetrahedralization] {end_time - start_time} sec | #T: {_tets.shape[0]}")
    
    # Build Grid
    start_time = time.time()
    _verts, _tets = _verts.to(device=th.device('cuda')), _tets.to(device=th.device('cuda'))
    grid = Grid(_verts, _tets)
    faces_is_valid = grid.face_tet[:, 1] >= 0
    face_tet = grid.face_tet[faces_is_valid].clone()
    faces = grid.faces[faces_is_valid].clone()
    del grid
    th.cuda.empty_cache()
    end_time = time.time()
    logging.info(f"[Grid] {end_time - start_time} sec | #F: {len(faces)}, #T: {len(_tets)}")


    # Compute Circumcenter
    start_time = time.time()
    tets_cc = compute_cc(_verts, _tets)
    end_time = time.time()
    th.cuda.empty_cache()
    logging.info(f"[Circumcenter] {end_time - start_time} sec | #CC: {tets_cc.shape[0]}")

    # Intersect
    start_time = time.time()
    face_cc_0 = tets_cc[face_tet[:, 0]]
    face_cc_1 = tets_cc[face_tet[:, 1]]
    hit = query_segments_any_hit(face_cc_0, face_cc_1)
    end_time = time.time()
    logging.info(f"[Intersect] {end_time - start_time} sec | #H: {hit.sum()}")

    # Export result
    start_time = time.time()
    hit_faces = faces[hit].cpu().numpy()
    if orient:
        hit_faces = np.array(igl.bfs_orient(hit_faces)[0])
    end_time = time.time()
    logging.info(f"[Get OUTPUT Faces] {end_time - start_time} sec | #H: {hit_faces.shape[0]}")
    
    return hit_faces