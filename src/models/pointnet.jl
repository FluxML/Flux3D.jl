export stnKD, PointNet

stnKD(K::Int) = Chain(
    Conv((1,), K => 64, relu),
    BatchNorm(64),
    Conv((1,), 64 => 128, relu),
    BatchNorm(128),
    Conv((1,), 128 => 1024, relu),
    BatchNorm(1024),
    x -> maximum(x, dims = 1),
    x -> reshape(x, :, size(x, 3)),
    Dense(1024, 512, relu),
    #     BatchNorm(512),
    Dense(512, 256, relu),
    BatchNorm(256),
    Dense(256, K * K),
    x -> reshape(x, K, K, size(x, 2)),
    # x -> x .+ I, #TODO: add identity matrix compatible with gpu
    x -> PermutedDimsArray(x, (2,1,3)),
)

"""
    PointNet(num_classes::Int=10, hidden_dims::Int=64)

Flux implementation of PointNet classification model.

### Fields:

- `num_classes` - Number of classes in dataset.
- `hidden_dims` - Hiddem dimension in PointNet model.

"""
struct PointNet
    stn::Any
    fstn::Any
    conv_block1::Any
    feat::Any
    cls::Any
end

function PointNet(num_classes::Int = 10, K::Int = 64)
    stn = stnKD(3)
    fstn = stnKD(K)
    conv_block1 = conv_bn_block(3, 64, (1,))
    feat = Chain(
        Conv((1,), 64 => 128, relu),
        BatchNorm(128),
        Conv((1,), 128 => 1024),
        BatchNorm(1024),
        x -> maximum(x, dims = 1),
        x -> reshape(x, 1024, :),
        Dense(1024, 512, relu),
        BatchNorm(512),
        Dense(512, 256, relu),
        Dropout(0.4),
        BatchNorm(256),
    )
    cls = Dense(256, num_classes, relu)
    PointNet(stn, fstn, conv_block1, feat, cls)
end

function (m::PointNet)(X)

    # X: [3, N, B]

    X = permutedims(X, (2, 1, 3))
    # X: [N, 3, B]

    X = Flux.batched_mul(X, m.stn(X))
    # X: [3, 3, B]

    X = m.conv_block1(X)
    # X: [3, 64, B]

    X = batched_mul(X, m.fstn(X))
    # X: [3, 64, B]

    X = m.feat(X)
    # X: [256, B]

    X = m.cls(X)
    # X: [num_classes, B]

    return softmax(X, dims = 1)
end

@functor PointNet
