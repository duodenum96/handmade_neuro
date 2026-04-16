import MAT as mat

hdr_path = "data/raw_data/P3/1/1_P3.set"
data_path = "data/raw_data/P3/1/1_P3.fdt"
zaa = mat.matread(hdr_path)["EEG"]

n_channels = Int(zaa["nbchan"])
n_tp = Int(zaa["pnts"])

# fdt is based on floatwrite, which is based on fwrite
# https://viewer.mathworks.com/addons/56415/2024.0/files/functions/sigprocfunc/floatwrite.m
# https://www.mathworks.com/help/matlab/ref/fwrite.html
data = open(data_path, "r") do io
    data = read!(io, Array{Float32}(undef, n_channels, n_tp))
    return data
end

