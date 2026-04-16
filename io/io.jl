"""
subj_id is a string (e.g. "1")
"""
function read_raw(subj_id)
    hdr_path = joinpath(raw_path, subj_id, "$(subj_id)_P3.set")
    data_path = joinpath(raw_path, subj_id, "$(subj_id)_P3.fdt")
    hdr = mat.matread(hdr_path)["EEG"]

    n_channels = Int(hdr["nbchan"])
    n_tp = Int(hdr["pnts"])


    # fdt is based on floatwrite, which is based on fwrite
    # https://viewer.mathworks.com/addons/56415/2024.0/files/functions/sigprocfunc/floatwrite.m
    # https://www.mathworks.com/help/matlab/ref/fwrite.html
    data = open(data_path, "r") do io
        data = read!(io, Array{Float32}(undef, n_channels, n_tp))
        return data
    end

    return hdr, data
end

