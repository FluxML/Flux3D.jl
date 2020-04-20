function find_datasets(root::String, query::String)
    dataset_folder = root
    for entry in readdir(dataset_folder)
        path = joinpath(dataset_folder, entry)
        isdir(path) || continue
        if entry == query
            return path
        end
    end
    return nothing
end

"""
Get the appropriate dataset from anywhere we can find.
Available options: ModelNet10, ModelNet40
"""
function dataset(which::String, root::String)
    if which === "ModelNet10PCloud"
        path = find_datasets(root, "modelnet40_normal_resampled")
        if path == nothing
            download("ModelNet", root)
            path = find_datasets(root, "modelnet40_normal_resampled")
        end
        return path
    elseif which == "ModelNet40PCloud"
        path = find_datasets(root, "modelnet40_normal_resampled")
        if path == nothing
            download("ModelNet", root)
            path = find_datasets(root, "modelnet40_normal_resampled")
        end
        return path
    else
        error("Autodetection not supported for $(which)")
    end
end

function download(which::String, root::String)
    if which == "ModelNet"
        local_path = joinpath(root, "modelnet40_normal_resampled.zip")
        dir_path = joinpath(root) #TODO check if dir_path exists and if not then create one.
        if(!isdir(joinpath(dir_path, "modelnet40_normal_resampled")))
            if(!isfile(local_path))
                # dataset prepared by authors of pointnet2
                Base.download("https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip", local_path)
            end
            run(`unzip $local_path -d $dir_path`)
        end
    else
        error("Download not supported for $(which)")
    end
end
