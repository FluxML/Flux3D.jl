export laplacian_loss, edge_loss

function laplacian_loss(m::TriMesh)
    L = @ignore get_laplacian_packed(m)
    verts = get_verts_packed(m)
    L = L * transpose(verts)
    L = _norm(L; dims = 2)
    return mean(L)
end

function edge_loss(m::TriMesh, target_length::Number=0.0)
    verts = get_verts_packed(m)
    edges = @ignore get_edges_packed(m)
    v1 = verts[:, edges[:,1]]
    v2 = verts[:, edges[:,2]]
    el =  (_norm(v1-v2; dims=1) .- Float32(target_length)) .^ 2
    loss = mean(el)
    return loss
end

function chamfer_distance(m1::TriMesh, m2::TriMesh, num_samples::Int = 5000; w1::Number=1.0, w2::Number=1.0)
    A = sample_points(m1, num_samples)
    B = sample_points(m2, num_samples)
    return _chamfer_distance(A, B, Float32(w1), Float32(w2))
end
