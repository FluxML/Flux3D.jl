export laplacian_loss, edge_loss

function laplacian_loss(m::TriMesh)
    # TODO: There will be some changes when migrating to batched format
    my_ignore() do
        L = get_laplacian_sparse(m)
    end
    L = L * m.vertices
    L = _norm(L; dims = 2)
    return sum(L)
end

function _edge_loss2(m::TriMesh, target_length::Number=0.0f0)
    #TODO: This is different approach to calculate edge_loss, remove one approach
    p1 = m.vertices[m.faces[:,1],:]
    p2 = m.vertices[m.faces[:,2],:]
    p3 = m.vertices[m.faces[:,3],:]

    e1 = p2-p1
    e2 = p3-p2
    e3 = p1-p2

    el1 = (_norm(e1; dims=2) .- Float32(target_length)) .^ 2
    el2 = (_norm(e1; dims=2) .- Float32(target_length)) .^ 2
    el3 = (_norm(e1; dims=2) .- Float32(target_length)) .^ 2

    # assuming homogenous mesh, each edge is shared between faces
    loss = (mean(el1) + mean(el2) + mean(el3)) / 6
    return loss
end

function edge_loss(m::TriMesh, target_length::Number=0.0f0)
    #TODO: will change changing to batched format
    edges = Zygote.nograd(get_edges(m))
    v1 = m.vertices[edges[:,1],:]
    v2 = m.vertices[edges[:,2],:]
    el =  (_norm(v1-v2; dims=2) .- Float32(target_length)) .^ 2
    loss = mean(el)
    return loss
end

function chamfer_distance(m1::TriMesh, m2::TriMesh, num_samples::Int = 5000; w1::Number=1.0, w2::Number=1.0)
    A = sample_points(m1, num_samples)
    B = sample_points(m2, num_samples)
    return chamfer_distance(A,B;w1=w1, w2=w2)
end

function normal_consistency_loss(m::TriMesh)
#     edges = get_edges(m)
    error("Not implemented")
end

function cloud_surface_distance(m::TriMesh)
    error("Not implemented")
end
