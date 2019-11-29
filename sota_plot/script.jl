
# packages
#= 
using Pkg
Pkg.add("Cascadia")
Pkg.add("Gumbo")
Pkg.add("HTTP")
Pkg.add("JSON")
Pkg.add("DataFrames")
Pkg.add("Dates")
Pkg.add("Plots")
=#

using Cascadia
using Gumbo
using HTTP
using JSON
using DataFrames
using Dates
using Plots


# scraping
r = HTTP.request("GET", "https://paperswithcode.com/sota/image-classification-on-imagenet")                      
body = String(r.body)
h = parsehtml(convert(String, body))
eva_tb_data = eachmatch(Selector("#evaluation-table-data"),h.root)
eva_tb_json = JSON.parse(nodeText(eva_tb_data[1]))

df_l = []
for row in eva_tb_json
    rank = row["rank"]
    method_short = row["method_short"]

    metrics = row["metrics"]
    num_params = metrics["Number of params"]
    if num_params != nothing
        num_params = replace(String(num_params), "M" => "")
        num_params = parse(Float64, num_params)
    else
        num_params = NaN
    end
    top1_acc = metrics["Top 1 Accuracy"]
    if top1_acc != nothing
        top1_acc = replace(String(top1_acc), "%" => "")
        top1_acc = parse(Float64, top1_acc)
    else
        top1_acc = NaN
    end
    top5_acc = metrics["Top 5 Accuracy"]
    if top5_acc != nothing
        top5_acc = replace(String(top5_acc), "%" => "")
        top5_acc = parse(Float64, top5_acc)
    else
        top5_acc = NaN
    end

    method = row["method"]
    method_details = row["method_details"]
    evaluation_date = row["evaluation_date"]
    evaluation_date = Date(evaluation_date, DateFormat("y-m-d"))
    uses_additional_data = row["uses_additional_data"]

    tmp = Dict(
        "rank" => rank,
        "method_short" => method_short,
        "Number_of_params" => num_params,
        "Top_1_Accuracy" => top1_acc,
        "Top_5_Accuracy" => top5_acc,
        "method" => method,
        "method_details" => method_details,
        "evaluation_date" => evaluation_date,
        "uses_additional_data" => uses_additional_data
    )
    push!(df_l, tmp)
end

# dataframe
df = reduce(vcat, DataFrame.(df_l)) 

# plot
scatter(df.evaluation_date , df.Number_of_params, ylabel = "Number of params(M)", legend = false)
savefig("fig.png")
