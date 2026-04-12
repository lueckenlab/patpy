import gudhi as gd
from scipy.sparse import coo_matrix, triu


def connectivities_to_edge_list(connectivities, mutal_nbhs=False):
    """Helper function to construct a list of edges from a neighbourhood graph.

    Parameters
    ----------
    connectivities:
        An adjacency or connectivity matrix. crs matrix
    mutual_nbhs:
        If TRUE, a mutual nbh graph is constructed, that is an edge (u,v) is only added when (v,u) is also in the nbh matrix.
        If FALSE, an edge between vertex u and v is added whenever (u,v) or (v,u) are in the nbh matrix.

    Returns
    -------
    A list of edges as given by the indices of the indices of their associated vertices i.e. [(u_1, v_1), (u_2, v_2), ..., (u_n, v_n)]
    """
    if mutal_nbhs:
        g = triu(connectivities).minimum(triu(connectivities.transpose()))  ## mutal nbh
    else:
        g = triu(connectivities).maximum(triu(connectivities.transpose()))  ## max ==> not mutal nbhs

    cx = coo_matrix(g)
    edge_list = [(a, b) for a, b in zip(cx.row, cx.col, strict=False)]
    return edge_list


### Use gudhi to calculate PH ###
def calculate_persistent_homology(
    vertex_feature, edge_list, k=2, order="sublevel", infinity_value="max", min_persistence=0.0
):
    """Calculates the persistent homology of a graph clique complex.

    In particular, this function calculates sublevel-set persistent homology
    taking the values of one specific vertex feature as a filtration function
    over a pre-constructed spatial graph.

    Parameters
    ----------
    vertex_feature:
        A list of numeric lables associated with each vertex, that is [f_1, f_2, ..., f_n]
        where f_i can be seen as the value of a scalar valued function evaluated at vertex i.
    edge list:
        A list of undirected edges (u,v) between vertex u and vertex v.
    k:
        Expands the simplex tree containing only its one skeleton until dimension k.
        The expanded simplicial complex until dimension k attached to a graph is
        the maximal simplicial complex of dimension at most k admitting the graph as 1-skeleton.
        If k is set to 3, the simplex will be expanded up to 3-cliques (i.e. faces).
        The filtration value assigned to a simplex is the maximal filtration value of one of its vertices.
    order:
        The order of the filtration. Either "subevel" or "superlevel".
    infinity_values:
        The value that the maximium death coordinates in the persistence diagram should be set to.
        For example, it could be +inf or any appropriate real number higher than
        the maximum value of the filtration function.
        By default it is equal to "max" and automatically sets the value of infinite coordinates
        to the maximum value of the filtration function.
    min_persistence:
        The minimum persistence value to take into account (strictly greater than min_persistence).
        Default value is 0.0. Set min_persistence to -1.0 to see all values
        (that is also include coordinates on the diagonal of the persistence diagram).

    Returns
    -------
    A list of persistence diagrams [PD0, PD1].
    PD0 is the persistence diagram of dimension 0 tracking the birth and death
    of 0-dimensional simplicial complexes (i.e. connected components) across the filtration.
    PD1 is the persitence diagram of dimension 1 recording the appearance and disappearance
    of 1-dimensional simplicial complexes (i.e. loops or circles).
    Each persistence diagram is a list of persistence coordinates.

    References
    ----------
    [1] Boissonnat and Maria (2024): https://gudhi.inria.fr/python/latest/simplex_tree_ref.html
    [2] Limbeck and Rieck (2024): https://arxiv.org/abs/2409.03575v1
    [3] Rieck et al. (2017): https://ieeexplore.ieee.org/document/8017588
    """
    eval_fn = max

    # print(infinity_values)
    if order == "sublevel":
        vertex_weights = vertex_feature
    elif order == "superlevel":
        vertex_weights = -(vertex_feature)

    if infinity_value == "max":
        infinity_value = max(vertex_weights)
    else:
        # assert type(infinity_values) == int or float
        assert infinity_value >= max(vertex_weights)

    st = gd.SimplexTree()

    for v in range(len(vertex_weights)):
        w = vertex_weights[v]
        st.insert([v], filtration=w)

    ## only iterate over unique edges, direction doesn't matter!
    for u, v in edge_list:
        f_u = vertex_weights[u]
        f_v = vertex_weights[v]
        w = eval_fn(f_u, f_v)
        st.insert([u, v], filtration=w)

    st.make_filtration_non_decreasing()
    st.expansion(k)
    persistence_pairs = st.persistence(min_persistence=min_persistence)

    diagrams = []

    for dimension in range(k - 1):
        diagram = [[c, min(d, infinity_value)] for dim, (c, d) in persistence_pairs if dim == dimension]

        diagrams.append(diagram)

    return diagrams
