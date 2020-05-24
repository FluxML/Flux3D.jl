using Flux3D, Flux, Zygote
using Parameters: @with_kw
using Flux: onehotbatch, onecold, onehot, crossentropy
using Statistics: mean
using Base.Iterators: partition

using CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw mutable struct Args
    K::Int = 10 # k nearest-neighbors
    batch_size::Int = 32
    lr::Float64 = 3e-4
    epochs::Int = 50
    num_classes::Int = 10 #possible values {10,40}
    npoints::Int = 1024
    cuda::Bool = true
    device = cpu
end

function get_processed_data(args)
    # Fetching the train and validation data and getting them into proper shape	
    if args.num_classes == 10
        dset = ModelNet10.dataset(;mode=:pointcloud, npoints=args.npoints, transforms=NormalizePointCloud())
    elseif args.num_classes == 40
        dset = ModelNet40.dataset(;mode=:pointcloud, npoints=args.npoints, transforms=NormalizePointCloud())
    else
        error("ModelNet dataset with $(args.num_classes) is not supported.
                Currently supported num_classes for ModelNet dataset is {10,40}")
    end

    data = [dset[i].data.points for i in 1:length(dset)]
    labels = onehotbatch([dset[i].ground_truth for i in 1:length(dset)],1:args.num_classes)

    #onehot encode labels of batch
    train = [(cat(data[i]..., dims = 3), labels[:,i]) for i in partition(1:length(data), args.batch_size)] .|> args.device
    
    if args.num_classes == 10
        VAL = ModelNet10.dataset(;mode=:pointcloud, train=false, npoints=args.npoints, transforms=NormalizePointCloud())
    elseif args.num_classes == 40
        VAL = ModelNet40.dataset(;mode=:pointcloud, train=false, npoints=args.npoints, transforms=NormalizePointCloud())
    else
        error("ModelNet dataset with $(args.num_classes) is not supported.
                Currently supported num_classes for ModelNet dataset is {10,40}")
    end

    valX = cat([VAL[i].data.points for i in 1:length(VAL)]..., dims=3) |> args.device
    valY = onehotbatch([VAL[i].ground_truth for i in 1:length(VAL)], 1:num_classes) |> args.device

    val = (valX,valY)	
    return train, val
end

function train(; kwargs...)
    # Initialize the hyperparameters
    args = Args(; kwargs...)

    # GPU config
    if args.cuda && has_cuda()
        args.device = gpu
        @info("Training on GPU")
    else
        args.device = cpu
        @info("Training on CPU")
    end
    
    @info("Loading Data...")
    # Load the train, validation data 
    train,val = get_processed_data(args)

    @info("Initializing Model...")	
    # Defining the loss and accuracy functions
    m = PointNet(args.num_classes) |> args.device

    loss(x, y) = crossentropy(m(x), y)
    accuracy(x, y) = mean(onecold(cpu(m(x)), 1:args.num_classes) .== onecold(cpu(y), 1:args.num_classes))

    ## Training
    opt = ADAM(args.lr)
    @info("Training...")
    # Starting to train models
    custom_train!(loss, params(m), train, opt, args.epochs, val, accuracy)

    return m
end

function custom_train!(loss, ps, data, opt, epochs, (valX, valY), accuracy)
    ps = Zygote.Params(ps)
    for epoch in 1:epochs
        running_loss = 0
        for d in data
        gs = gradient(ps) do
            training_loss = loss(d...)
            running_loss += training_loss
            return training_loss
        end
        Flux.update!(opt, ps, gs)
        end
        print("Epoch: $(epoch), epoch_loss: $(running_loss), accuracy: $(accuracy(valX, valY))\n")
    end
end

m = train()