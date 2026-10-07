import torch as th

def compute_cc(points: th.Tensor, simp: th.Tensor, batch_size: int = 2_000_000):
    """
    Compute circumcenter of simplexes.

    Args:
        points (th.Tensor): Points coordinates.
        simp (th.Tensor): Simplexes.

    Returns:
        th.Tensor: Circumcenter coordinates.
    """
    
    num_points = points.shape[0]
    num_simplex = simp.shape[0]
    dimension = points.shape[1]
    circumcenters = []
    
    weights = th.zeros((num_points), dtype=th.float32, device=points.device)
    
    s = 0
    while s < num_simplex:
        e = min(num_simplex, s + batch_size)
        curr_simp = simp[s:e]
        curr_num_simplex = curr_simp.shape[0]
    
        '''
        1. Gather point coordinates for each simplex.
        '''
        # [# simplex, # dim + 1, # dim]
        simplex_points = points[curr_simp]

        # [# simplex, # dim + 1]
        simplex_weights = weights[curr_simp]

        '''
        2. Change points in [# dim] dimension to hyperplanes in [# dim + 1] dimension
        '''
        # [# simplex, # dim + 1, # dim + 2]
        hyperplanes0 = th.ones_like(simplex_points[:, :, [0]]) * -1.
        hyperplanes1 = simplex_weights.unsqueeze(-1) - \
            th.sum(simplex_points * simplex_points, dim=-1, keepdim=True)
        hyperplanes = th.cat([simplex_points * 2., hyperplanes0, hyperplanes1], dim=-1)

        '''
        3. Find intersection of hyperplanes above to get circumcenter.
        '''
        mats = []
        for dim in range(dimension + 2):
            cols = list(range(dimension + 2))
            cols = cols[:dim] + cols[(dim + 1):]

            # [# simplex, # dim + 1, # dim + 1]
            mat = hyperplanes[:, :, cols]
            mats.append(mat)

        # [# simplex * (# dim + 2), # dim + 1, # dim + 1]
        detmat = th.cat(mats, dim=0)

        # [# simplex * (# dim + 2)]
        det = th.det(detmat)

        # [# simplex, # dim + 2]
        hyperplane_intersections0 = det.reshape((dimension + 2, curr_num_simplex))
        hyperplane_intersections0 = th.transpose(hyperplane_intersections0.clone(), 0, 1)
        sign = 1.
        for dim in range(dimension + 2):
            hyperplane_intersections0[:, dim] = hyperplane_intersections0[:, dim] * sign
            sign *= -1.
            
        # [# simplex, # dim + 2]
        eps = 1e-6
        last_dim = hyperplane_intersections0[:, [-1]]
        is_stable = th.abs(last_dim) > eps
        last_dim = th.sign(last_dim) * th.clamp(th.abs(last_dim), min=eps)
        last_dim = th.where(last_dim == 0., th.ones_like(last_dim) * eps, last_dim)
        hyperplane_intersections = hyperplane_intersections0[:, :] / \
                                        last_dim

        '''
        Projection
        '''
        # [# simplex, # dim]
        curr_circumcenters = hyperplane_intersections[:, :-2]
        if th.any(th.isnan(curr_circumcenters)) or th.any(th.isinf(curr_circumcenters)):
            raise ValueError()
        
        circumcenters.append(curr_circumcenters)
        s = e
        
    circumcenters = th.cat(circumcenters, dim=0)
    assert circumcenters.shape[0] == num_simplex, "Invalid circumcenter shape."
    
    return circumcenters

def add_ordinal_axis(mat: th.Tensor):
    '''
    Add an ordinal axis to 2-D tensor's last axis.

    @ mat: (num_row, num_col)
    @ return: (num_row, num_col + 1)
    '''

    num_row = mat.shape[0]
    new_col = th.arange(num_row, device=mat.device).reshape((-1, 1))
    return th.cat([mat, new_col], dim=1)

class Grid:

    def __init__(self, 
                verts: th.Tensor, 
                tets: th.Tensor):
        '''
        @ verts: (# vert, 3)
        @ tets: (# tet, 4)
        '''
        self.device = verts.device
        self.verts = verts
        self.tets = tets

        assert th.min(self.tets) >= 0 and th.max(self.tets) < self.num_verts, \
            'Invalid tetrahedron indices.'

        # faces;
        self.faces = None           # (# face, 3)
        self.face_tet = None        # (# face, 2), membership of each face to two tets
        self.init_faces()

    @property
    def num_verts(self):
        return self.verts.shape[0]

    @property
    def num_faces(self):
        return self.faces.shape[0]

    @property
    def num_tets(self):
        return self.tets.shape[0]
    
    def init_faces(self):
        '''
        Initialize face information.
        '''
        # faces;
        faces = []
        face_tet = []
        for comb in [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 3, 1], [1, 2, 3, 0]]:
            faces.append(self.tets[:, comb[:3]])
            face_tet.append(th.arange(self.num_tets, device=self.device))
        faces = th.cat(faces, dim=0)        # (# dup face, 3), duplicates are possible
        face_tet = th.cat(face_tet, dim=0)  # (# dup face)

        # sort;
        tmp = th.cat([faces, face_tet.unsqueeze(-1)], dim=1)      # (# dup face, 4)
        tmp[:, :3] = th.sort(tmp[:, :3], dim=1)[0]
        tmp = th.unique(tmp, dim=0)
        
        # remove duplicate faces;
        u_faces, u_faces_cnt = th.unique(tmp[:, :3], dim=0, return_counts=True)
        assert th.all(u_faces_cnt <= 2), "A face can be shared by at most 2 tets."
        u_faces_first_id = th.cumsum(u_faces_cnt, dim=0)[:-1]
        u_faces_first_id = th.cat([th.zeros(1, dtype=u_faces_first_id.dtype, device=self.device), 
                                    u_faces_first_id], 
                                    dim=0)
        
        self.faces = u_faces
        self.face_tet = th.zeros((len(u_faces), 2), dtype=u_faces.dtype, device=self.device) - 1
        for i in range(2):
            valid_i = i < u_faces_cnt
            self.face_tet[valid_i, i] = \
                tmp[u_faces_first_id[valid_i] + i, -1]
