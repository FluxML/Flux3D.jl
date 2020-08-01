function conv_bn_block(input, output, kernel_size)
    return Chain(Conv((kernel_size...,), input => output), BatchNorm(output), x -> relu.(x))
end

function fc_bn_block(input, output)
    return Chain(Dense(input, output), BatchNorm(output), x -> relu.(x))
end
