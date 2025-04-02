import trimesh

def load_scene(filepath):
    mesh = trimesh.load(filepath)
    if isinstance(mesh, trimesh.Scene):
        return trimesh.util.concatenate(mesh.dump())
    return mesh
