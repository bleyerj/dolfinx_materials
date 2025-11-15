import numpy as np
import numpy.typing as npt
import dolfinx


def get_entity_map(
    entity_map: dolfinx.mesh.EntityMap, inverse: bool = False
) -> npt.NDArray[np.int32]:
    """Get an entity map from the sub-topology to the topology.

    Args:
        entity_map: An `EntityMap` object or a numpy array representing the mapping.
        inverse: If `True`, return the inverse mapping.
    Returns:
        Mapped indices of entities.
    """
    sub_top = entity_map.sub_topology
    assert isinstance(sub_top, dolfinx.mesh.Topology)
    sub_map = sub_top.index_map(entity_map.dim)
    indices = np.arange(sub_map.size_local + sub_map.num_ghosts, dtype=np.int32)
    return entity_map.sub_topology_to_topology(indices, inverse=inverse)


def transfer_meshtags_to_submesh(
    mesh: dolfinx.mesh.Mesh,
    entity_tag: dolfinx.mesh.MeshTags,
    submesh: dolfinx.mesh.Mesh,
    vertex_entity_map: dolfinx.mesh.EntityMap,
    cell_entity_map: dolfinx.mesh.EntityMap,
) -> tuple[dolfinx.mesh.MeshTags, npt.NDArray[np.int32]]:
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.

    Args:
        mesh: Mesh containing the meshtags
        entity_tag: The meshtags object to transfer
        submesh: The submesh to transfer the `entity_tag` to
        sub_to_vertex_map: Map from each vertex in `submesh` to the corresponding
            vertex in the `mesh`
        sub_cell_to_parent: Map from each cell in the `submesh` to the corresponding
            entity in the `mesh`
    Returns:
        The entity tag defined on the submesh, and a map from the entities in the
        `submesh` to the entities in the `mesh`.

    """

    sub_cell_to_parent = get_entity_map(cell_entity_map, inverse=False)
    sub_vertex_to_parent = get_entity_map(vertex_entity_map, inverse=False)

    num_cells = (
        mesh.topology.index_map(mesh.topology.dim).size_local
        + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    )
    mesh_to_submesh = np.full(num_cells, -1, dtype=np.int32)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32
    )

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)
                    ).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet
    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name
    return tags, sub_to_parent_entity_map


def interpolate_submesh_to_parent(u_parent, u_sub, sub_to_parent_cells):
    """
    Interpolate results from functions defined in submeshes to a function defined on the parent mesh.

    Parameters
    ----------
    u_parent : dolfinx.fem.Function
        Parent function to interpolate to.
    u_sub : list[dolfinx.fem.Function]
        _description_
    sub_to_parent_cells : list
        Submesh cells to parent cells mapping
    """
    V_parent = u_parent.function_space
    V_sub = u_sub.function_space
    for i, cell in enumerate(sub_to_parent_cells):
        bs = V_parent.dofmap.bs
        bs_sub = V_sub.dofmap.bs
        assert bs == bs_sub
        parent_dofs = V_parent.dofmap.cell_dofs(cell)
        sub_dofs = V_sub.dofmap.cell_dofs(i)
        for p_dof, s_dof in zip(parent_dofs, sub_dofs):
            for j in range(bs):
                u_parent.x.array[p_dof * bs + j] = u_sub.x.array[s_dof * bs + j]


def interface_int_entities(
    msh,
    interface_facets,
    domain_to_domain_0,
    domain_to_domain_1,
):
    """
    This function computes the integration entities (as a list of pairs of
    (cell, local facet index) pairs) required to assemble mixed domain forms
    over the interface. It assumes there is a domain with two sub-domains,
    domain_0 and domain_1, that have a common interface. Cells in domain_0
    correspond to the "+" restriction and cells in domain_1 correspond to
    the "-" restriction.

    Parameters:
        interface_facets: A list of facets on the interface
        domain_0_cells: A list of cells in domain_0
        domain_1_cells: A list of cells in domain_1
        c_to_f: The cell to facet connectivity for the domain mesh
        f_to_c: the facet to cell connectivity for the domain mesh
        facet_imap: The facet index_map for the domain mesh
        domain_to_domain_0: A map from cells in domain to cells in domain_0
        domain_to_domain_1: A map from cells in domain to cells in domain_1

    Returns:
        A tuple containing:
            1) The integration entities
            2) A modified map (see HACK below)
            3) A modified map (see HACK below)
    """
    # Create measure for integration. Assign the first (cell, local facet)
    # pair to the cell in domain_0, corresponding to the "+" restriction.
    # Assign the second pair to the domain_1 cell, corresponding to the "-"
    # restriction.
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    facet_imap = msh.topology.index_map(fdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    # FIXME This can be done more efficiently
    interface_entities = []
    domain_to_domain_0_new = np.array(domain_to_domain_0)
    domain_to_domain_1_new = np.array(domain_to_domain_1)
    for facet in interface_facets:
        # Check if this facet is owned
        if facet < facet_imap.size_local:
            cells = f_to_c.links(facet)
            assert len(cells) == 2
            if domain_to_domain_0[cells[0]] > 0:
                cell_plus = cells[0]
                cell_minus = cells[1]
            else:
                cell_plus = cells[1]
                cell_minus = cells[0]
            assert (
                domain_to_domain_0[cell_plus] >= 0
                and domain_to_domain_0[cell_minus] < 0
            )
            assert (
                domain_to_domain_1[cell_minus] >= 0
                and domain_to_domain_1[cell_plus] < 0
            )

            local_facet_plus = np.where(c_to_f.links(cell_plus) == facet)[0][0]
            local_facet_minus = np.where(c_to_f.links(cell_minus) == facet)[0][0]

            interface_entities.extend(
                [cell_plus, local_facet_plus, cell_minus, local_facet_minus]
            )

            # FIXME HACK cell_minus does not exist in the left submesh, so it
            # will be mapped to index -1. This is problematic for the
            # assembler, which assumes it is possible to get the full macro
            # dofmap for the trial and test functions, despite the restriction
            # meaning we don't need the non-existant dofs. To fix this, we just
            # map cell_minus to the cell corresponding to cell plus. This will
            # just add zeros to the assembled system, since there are no
            # u("-") terms. Could map this to any cell in the submesh, but
            # I think using the cell on the other side of the facet means a
            # facet space coefficient could be used
            domain_to_domain_0_new[cell_minus] = domain_to_domain_0[cell_plus]
            # Same hack for the right submesh
            domain_to_domain_1_new[cell_plus] = domain_to_domain_1[cell_minus]

    return interface_entities, domain_to_domain_0_new, domain_to_domain_1_new
