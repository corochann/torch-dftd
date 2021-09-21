from time import perf_counter
from typing import Optional

import torch
from torch import Tensor


def calc_distances(
    pos: Tensor,
    edge_index: Tensor,
    cell: Optional[Tensor] = None,
    shift: Optional[Tensor] = None,
    batch_edge: Optional[Tensor] = None,
    eps=1e-20,
) -> Tensor:
    print("einsum...")
    torch.cuda.synchronize()
    t0 = perf_counter()

    idx_i, idx_j = edge_index
    # calculate interatomic distances
    Ri = pos[idx_i]
    Rj = pos[idx_j]
    if cell is not None:
        if batch_edge is None:
            # shift (n_edges, 3), cell (3, 3) -> offsets (n_edges, 3)
            offsets = torch.mm(shift, cell)
        else:
            # shift (n_edges, 3), cell[batch] (n_atoms, 3, 3) -> offsets (n_edges, 3)
            # offsets = torch.bmm(shift[:, None, :], cell[batch_edge])[:, 0]
            uniq, counts = batch_edge.unique(return_counts=True)

            clist = [0] + counts.tolist()
            offsets_list = [torch.mm(shift[clist[i]:clist[i+1]], cell[i])
                            for i in range(len(counts))]
            offsets = torch.cat(offsets_list, dim=0)
            # import IPython; IPython.embed()

        Rj += offsets
    # eps is to avoid Nan in backward when Dij = 0 with sqrt.
    Dij = torch.sqrt(torch.sum((Ri - Rj) ** 2, dim=-1) + eps)

    torch.cuda.synchronize()
    t1 = perf_counter()
    duration = t1 - t0
    print(f"einsum done!... {duration} sec")
    return Dij
