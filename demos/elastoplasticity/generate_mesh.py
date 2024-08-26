import gmsh
from mpi4py import MPI
from dolfinx.io.gmshio import model_to_mesh


def generate_perforated_plate(W, H, R, mesh_sizes):
    """Generate a perforated rectangular plate of dimensions W x H and hole radius R."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info

    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, W, H, tag=1)
        hole = gmsh.model.occ.addDisk(
            W / 2,
            H / 2,
            0,
            R,
            R,
            zAxis=[0, 0, 1],
            xAxis=[0.0, 1.0, 0.0],
        )
        gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, hole)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(gdim, [volumes[0][1]], 1, name="Plate")
        gmsh.model.addPhysicalGroup(gdim - 1, [1], 1, name="Bottom")
        gmsh.model.addPhysicalGroup(gdim - 1, [4], 2, name="Top")

        try:
            field_tag = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(field_tag, "VIn", min(mesh_sizes))
            gmsh.model.mesh.field.setNumber(field_tag, "VOut", max(mesh_sizes))
            gmsh.model.mesh.field.setNumber(field_tag, "XMin", 0)
            gmsh.model.mesh.field.setNumber(field_tag, "XMax", W)
            gmsh.model.mesh.field.setNumber(field_tag, "YMin", H / 2 - 1.2 * R)
            gmsh.model.mesh.field.setNumber(field_tag, "YMax", H / 2 + 1.2 * R)
            gmsh.model.mesh.field.setNumber(field_tag, "Thickness", R)
            gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)
        except:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_sizes)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_sizes)

        gmsh.model.mesh.generate(gdim)

        domain, markers, facets = model_to_mesh(
            gmsh.model,
            mesh_comm,
            model_rank,
            gdim=gdim,
        )

    gmsh.finalize()
    return domain, markers, facets
