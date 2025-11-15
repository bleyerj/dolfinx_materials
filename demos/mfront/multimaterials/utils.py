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


def interpolate_submesh_to_parent(
    u_parent: dolfinx.fem.Function,
    u_sub: dolfinx.fem.Function,
    entity_map: dolfinx.mesh.EntityMap,
):
    """
    Copy DOFs from a function on a submesh into a function on the parent mesh,
    assuming identical function spaces and identical local DOF layout.

    Parameters
    ----------
    u_parent : dolfinx.fem.Function
        Function on the parent mesh (output).
    u_sub : dolfinx.fem.Function
        Function on the submesh (input).
    entity_map : dolfinx.mesh.EntityMap
        Maps submesh cells → parent mesh cells.
    """

    Vp = u_parent.function_space
    Vs = u_sub.function_space

    assert Vp.dofmap.bs == Vs.dofmap.bs, "Block sizes must match."

    bs = Vp.dofmap.bs

    # Number of local submesh cells
    tdim = Vs.mesh.topology.dim
    nc_sub = Vs.mesh.topology.index_map(tdim).size_local

    # Submesh cell indices: 0 .. nc_sub-1
    sub_cells = np.arange(nc_sub, dtype=np.int32)

    # Map to parent cells
    # entity_map outputs parent cell indices for each sub cell
    parent_cells = entity_map.sub_topology_to_topology(sub_cells, inverse=False)

    # Copy DOFs cell-wise
    for s_cell, p_cell in zip(sub_cells, parent_cells):

        # Could be -1 in parallel if parent cell is remote; skip those.
        if p_cell < 0:
            continue

        sub_dofs = Vs.dofmap.cell_dofs(s_cell)
        parent_dofs = Vp.dofmap.cell_dofs(p_cell)

        # Cell layout is guaranteed identical → direct copy OK
        if bs == 1:
            u_parent.x.array[parent_dofs] = u_sub.x.array[sub_dofs]
        else:
            for pd, sd in zip(parent_dofs, sub_dofs):
                u_parent.x.array[pd * bs : (pd + 1) * bs] = u_sub.x.array[
                    sd * bs : (sd + 1) * bs
                ]


def interface_int_entities(
    msh: dolfinx.mesh.Mesh, interface_facets: np.ndarray, marker: np.ndarray
):
    """
    This helper function computes the integration entities for
    interior facet integrals (i.e. a list of (cell_0, local_facet_0,
    cell_1, local_facet_1)) over an interface. The integration
    entities are ordered consistently such that cells for which
    `marker[cell] != 0` correspond to the "+" restriction, and cells
    for which `marker[cell] == 0` correspond to the "-" restriction.

    Parameters:
        msh: the mesh
        interface_facets: Facet indices of interior facets on an
            interface
        marker: If `marker[cell] != 0`, then that cell corresponds
            to a "+" restriction. Otherwise, it corresponds to a
            negative restriction.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    facet_imap = msh.topology.index_map(fdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)

    interface_entities = []
    for facet in interface_facets:
        # Check if this facet is owned
        if facet < facet_imap.size_local:
            cells = f_to_c.links(facet)
            assert len(cells) == 2
            if marker[cells[0]] == 0:
                cell_plus, cell_minus = cells[1], cells[0]
            else:
                cell_plus, cell_minus = cells[0], cells[1]

            local_facet_plus = np.where(c_to_f.links(cell_plus) == facet)[0][0]
            local_facet_minus = np.where(c_to_f.links(cell_minus) == facet)[0][0]

            interface_entities.extend(
                [cell_plus, local_facet_plus, cell_minus, local_facet_minus]
            )

    return interface_entities
