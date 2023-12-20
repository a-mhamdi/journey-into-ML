using StatsBase

## Horsepower
hp = [200, 300, 150, 250, 180, 280, 160, 240]
uhp, shp = mean(hp), std(hp)
## Acceleration
a = [8, 6, 9, 7, 8.5, 6.5, 9.5, 7.5]
ua, sa = mean(a), std(a)
## Miles per gallon
mpg = [25, 20, 30, 15, 22, 18, 28, 16]
umpg, smpg = mean(mpg), std(mpg)
## Weight
weight = [1200, 1500, 1400, 2000, 1300, 1600, 1500, 1800]
uweight, sweight = mean(weight), std(weight)

# Construct a dataframe object
using DataFrames
df = DataFrame("hp" => vec(hp), "a" => vec(a), "mpg" => vec(mpg), "weight" => vec(weight))

# Change scitypes of cols in df
using MLJ
df_c = coerce(df, :hp => Continuous, :a => Continuous, :mpg => Continuous, :weight => Continuous)

# Standardization
## 1st method
data_1 = DataFrame(zeros(Float64, size(df)), names(df))
foreach((x, y) -> y .= (x .- mean(x)) ./ std(x, corrected=true), eachcol(df_c), eachcol(data_1))
## 2nd method
data_2 = mapcols(zscore, df)
## 3rd method
sc_ = Standardizer()
sc = machine(sc_, df_c) |> fit!
data_3 = MLJ.transform(sc, df_c)

# New observation -> Standardization
obs = [220 7.2 24 1400]
obsn = [(220-uhp)/shp (7.2-ua)/sa (24-umpg)/smpg (1400-uweight)/sweight]

@info "Can Be Done Also As: `obsn = MLJ.transform(sc, DataFrame(obs, names(df)))`"

dist = round.(sqrt.(sum((obsn .- hcat(data_1.hp, data_1.a, data_1.mpg, data_1.weight)).^2, dims=2)), digits=2)

