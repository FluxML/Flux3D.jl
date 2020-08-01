export DGCNN, EdgeConv

function CreateSingleKNNGraph(X::AbstractArray, K::Int)
    F, N = size(X)
    kdtree = KDTree(X)
    cat([X[:, (knn(kdtree, X[:, i], K + 1, true)[1][2:K+1])] for i = 1:N]..., dims = 3)
end

@nograd CreateSingleKNNGraph

struct EdgeConv
    layers::AbstractArray
    K::Int
    mlp::Any
    maxpool_K::Any
end

function EdgeConv(layers::AbstractArray, K::Int)
    m = []

    for i = 1:(length(layers)-1)
        if i == 1
            push!(m, conv_bn_block(2 * layers[i], layers[i+1], (1,)))
        else
            push!(m, conv_bn_block(layers[i], layers[i+1], (1,)))
        end
    end

    EdgeConv(layers, K, Chain(m...), MaxPool((K,)))
end

function (m::EdgeConv)(X)
    F, N, B = size(X)
    # X: [F, N, B]

    KNNGraph = cat([CreateSingleKNNGraph(X[:, :, i], m.K) for i = 1:B]..., dims = 4)
    # KNNGraph: [F, K, N, B]

    X = reshape(X, F, 1, N, B)
    # X: [F, 1, N, B]

    X = cat([X for i = 1:m.K]..., dims = 2)
    # X: [F, K, N, B]

    X = cat(X, KNNGraph - X, dims = 1)
    # X: [2*F, K, N, B]

    X = PermutedDimsArray(X, (2, 3, 1, 4))
    # X: [K, N, 2*F, B]

    X = reshape(X, N * m.K, 2 * F, B)
    # X: [K*N, 2*F, B]

    X = m.mlp(X)
    # X: [K*N, an, B]

    an = size(X)[2]
    X = reshape(X, m.K, an * N, B)
    # X: [K, N*an, B]

    X = m.maxpool_K(X)
    # X: [1, N*an, B]

    X = reshape(X, N, an, B)
    # X: [N, an, B]

    X = PermutedDimsArray(X, (2, 1, 3))
    # X: [an, N, B]

    return X
end

@functor EdgeConv

struct DGCNN
    EdgeConv1::EdgeConv
    EdgeConv2::EdgeConv
    conv_3::Any
    maxpool_3::Any
    fc_4::Any
    drop_4::Any
    fc_5::Any
    drop_5::Any
    fc_6::Any
end

function DGCNN(num_classes::Int = 10, K::Int = 10, npoints::Int = 1024)
    DGCNN(
        EdgeConv([3, 32, 64, 64], K),
        EdgeConv([64, 128, 256], K),
        conv_bn_block(256, 1024, (1,)),
        MaxPool((npoints,)),
        fc_bn_block(1024, 512),
        Dropout(0.5),
        fc_bn_block(512, 256),
        Dropout(0.5),
        Dense(256, num_classes),
    )
end

function (m::DGCNN)(X)

    # X: [3, N, B]

    C, N, B = size(X)
    X = m.EdgeConv1(X)
    # X: [64, N, B]

    X = m.EdgeConv2(X)
    # X: [128, N, B]

    X = PermutedDimsArray(X, (2, 1, 3))
    # X: [N, 128, B]

    X = m.conv_3(X)
    # X: [N, 1024, B]

    X = m.maxpool_3(X)
    # X: [1, 1024, B]

    X = reshape(X, :, B)
    # X: [1024, B]

    X = m.drop_4(m.fc_4(X))
    # X: [512, B]

    X = m.drop_5(m.fc_5(X))
    # X: [256, B]

    X = m.fc_6(X)
    # X: [num_classes, B]

    return softmax(X, dims = 1)

end

@functor DGCNN
