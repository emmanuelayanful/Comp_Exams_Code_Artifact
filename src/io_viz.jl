# ============================================================
# io_viz.jl
#
# This file handles:
# 1. Reading saved experiment results from Excel
# 2. Choosing a good legend position
# 3. Plotting summary result curves
# 4. Saving a picture of the raw dataset
#
# Unlike core.jl, this file is not the "math engine".
# It is the presentation layer.
# ============================================================


# ============================================================
# 1. READ ONE SHEET OF RESULTS FROM THE EXCEL WORKBOOK
# ============================================================
#
# Inputs:
#   filepath  = folder containing the workbook
#   filename  = workbook name, e.g. "blobs.xlsx"
#   sheetname = sheet to read, e.g. "disjoint"
#
# Output:
#   a DataFrame containing the contents of that sheet
#
# What it does:
#   - opens the Excel workbook
#   - reads one sheet
#   - converts it to a DataFrame
#   - renames all columns to lowercase Symbols
#
# Why rename columns?
#   Because then later code can reliably use names like
#   :method, :nedges, :accuracy instead of worrying about
#   capitalization differences from Excel.
function read_results(filepath::String, filename::String, sheetname::String)::DataFrame
    # Open the workbook
    xf = XLSX.readxlsx(joinpath(filepath, filename))

    # Read the selected sheet and convert it to a DataFrame
    df = DataFrame(XLSX.gettable(xf[sheetname]))

    # Standardize column names:
    # Example: "Method" -> :method, "Nedges" -> :nedges
    rename!(df, Symbol.(lowercase.(string.(names(df)))))

    return df
end


# ============================================================
# 2. CHOOSE A GOOD LEGEND POSITION
# ============================================================
#
# Goal:
#   Put the legend where it blocks as little of the data as possible.
#
# Inputs:
#   xs_list = list of x-coordinate vectors, one per plotted curve
#   ys_list = list of y-coordinate vectors, one per plotted curve
#
# Method:
#   - Combine all points from all curves
#   - Split the plot region into four quadrants:
#       :lt = left top
#       :rt = right top
#       :lb = left bottom
#       :rb = right bottom
#   - Count how many data points fall into each quadrant
#   - Return the quadrant with the fewest points
#
# This is a simple heuristic, not a perfect algorithm,
# but it usually gives a reasonable legend location.
function best_legend_position(xs_list, ys_list)
    # Combine all x-values into one long vector
    xs = vcat(xs_list...)

    # Combine all y-values into one long vector
    ys = vcat(ys_list...)

    # Overall bounds of all plotted data
    xmin, xmax = minimum(xs), maximum(xs)
    ymin, ymax = minimum(ys), maximum(ys)

    # Midpoint of plotting region
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2

    # Count how many points fall in each quadrant
    counts = Dict(:lt => 0, :rt => 0, :lb => 0, :rb => 0)

    for (x, y) in zip(xs, ys)
        if x <= xmid && y >= ymid
            counts[:lt] += 1
        elseif x > xmid && y >= ymid
            counts[:rt] += 1
        elseif x <= xmid && y < ymid
            counts[:lb] += 1
        else
            counts[:rb] += 1
        end
    end

    # Return the quadrant with the smallest count
    return argmin(counts)
end


# ============================================================
# 3. PLOT SUMMARY RESULTS
# ============================================================
#
# This function reads one results sheet and produces a line plot of:
#
#   metric  vs  number of edges
#
# Example:
#   accuracy vs nedges
#   l1 vs nedges
#   accuracy_error vs nedges
#
# It separates the results into method families:
#   - Elliptic Gabriel
#   - Double Cone
#   - KNN
#   - Epsilon Ball
#
# and plots each method as its own curve.
function plot_results(;
    filepath::String,
    filename::String,
    sheetname::String,
    metric::Symbol=:accuracy,
    savepath::Union{String,Nothing}=nothing,
    show::Bool=true,
    xticklabelsize::Int=18,
    yticklabelsize::Int=18,
    linewidth::Int=4,
    savefilename::Union{String,Nothing}=nothing,
    figsize::Tuple{Int,Int}=(900, 650)
)

    # --------------------------------------------------------
    # Step 1: read the results from Excel
    # --------------------------------------------------------
    df = read_results(filepath, filename, sheetname)

    # --------------------------------------------------------
    # Step 2: identify which rows belong to which method
    # --------------------------------------------------------
    #
    # The workbook stores all methods in one table.
    # We separate them by checking the text in df.method.
    method_strings = string.(df.method)

    egab_mask = occursin.("Elliptic Gabriel", method_strings)
    dc_mask = occursin.("Double Cone", method_strings)
    knn_mask = occursin.("KNN", method_strings)
    epsb_mask = occursin.("Epsilon Ball", method_strings)

    # --------------------------------------------------------
    # Step 3: keep only the columns we need and sort by nedges
    # --------------------------------------------------------
    #
    # We only need:
    #   :nedges
    #   metric
    #
    # Sorting by :nedges ensures each line is drawn left to right.
    egab = sort(df[egab_mask, [:nedges, metric]], :nedges)
    dc = sort(df[dc_mask, [:nedges, metric]], :nedges)
    knn = sort(df[knn_mask, [:nedges, metric]], :nedges)
    epsb = sort(df[epsb_mask, [:nedges, metric]], :nedges)

    # --------------------------------------------------------
    # Step 4: determine y-axis limits from all methods combined
    # --------------------------------------------------------
    yall = Float64[]

    nrow(egab) > 0 && append!(yall, Float64.(egab[!, metric]))
    nrow(dc) > 0 && append!(yall, Float64.(dc[!, metric]))
    nrow(knn) > 0 && append!(yall, Float64.(knn[!, metric]))
    nrow(epsb) > 0 && append!(yall, Float64.(epsb[!, metric]))

    y_min = minimum(yall)
    y_max = maximum(yall)

    # Add a small padding so the curves do not sit right on the axis bounds
    pad = max(0.002, 0.08 * max(y_max - y_min, 1e-12))

    # --------------------------------------------------------
    # Step 5: create the figure and axis
    # --------------------------------------------------------
    fig = Figure(size=figsize)

    ax = Axis(
        fig[1, 1],
        xlabel="Number of Edges",
        ylabel=String(metric),
        xticklabelsize=xticklabelsize,
        yticklabelsize=yticklabelsize,
    )

    ylims!(ax, y_min - pad, y_max + pad)

    # --------------------------------------------------------
    # Step 6: prepare legend handles and colors
    # --------------------------------------------------------
    handles, labels = Any[], String[]

    # Chosen colors for each method family
    c_egab, c_dc, c_knn, c_epsb = :blue, :red, :black, :green

    # --------------------------------------------------------
    # Step 7: draw each curve if that method exists
    # --------------------------------------------------------
    #
    # For each method:
    #   - draw line
    #   - add a matching legend entry
    if nrow(knn) > 0
        lines!(
            ax,
            Float64.(knn.nedges),
            Float64.(knn[!, metric]),
            color=c_knn,
            linewidth=linewidth
        )
        push!(handles, LineElement(color=c_knn, linewidth=linewidth))
        push!(labels, "k-NN")
    end

    if nrow(epsb) > 0
        lines!(
            ax,
            Float64.(epsb.nedges),
            Float64.(epsb[!, metric]),
            color=c_epsb,
            linewidth=linewidth
        )
        push!(handles, LineElement(color=c_epsb, linewidth=linewidth))
        push!(labels, "Epsilon Ball")
    end

    if nrow(dc) > 0
        lines!(
            ax,
            Float64.(dc.nedges),
            Float64.(dc[!, metric]),
            color=c_dc,
            linewidth=linewidth
        )
        push!(handles, LineElement(color=c_dc, linewidth=linewidth))
        push!(labels, "Double Cone")
    end

    if nrow(egab) > 0
        lines!(
            ax,
            Float64.(egab.nedges),
            Float64.(egab[!, metric]),
            color=c_egab,
            linewidth=linewidth
        )
        push!(handles, LineElement(color=c_egab, linewidth=linewidth))
        push!(labels, "Elliptic Gabriel")
    end

    # --------------------------------------------------------
    # Step 8: collect x/y coordinates to choose legend location
    # --------------------------------------------------------
    xs_list = Vector{Vector{Float64}}()
    ys_list = Vector{Vector{Float64}}()

    nrow(knn) > 0 && push!(xs_list, Float64.(knn.nedges))
    nrow(knn) > 0 && push!(ys_list, Float64.(knn[!, metric]))

    nrow(epsb) > 0 && push!(xs_list, Float64.(epsb.nedges))
    nrow(epsb) > 0 && push!(ys_list, Float64.(epsb[!, metric]))

    nrow(dc) > 0 && push!(xs_list, Float64.(dc.nedges))
    nrow(dc) > 0 && push!(ys_list, Float64.(dc[!, metric]))

    nrow(egab) > 0 && push!(xs_list, Float64.(egab.nedges))
    nrow(egab) > 0 && push!(ys_list, Float64.(egab[!, metric]))

    position = best_legend_position(xs_list, ys_list)

    # --------------------------------------------------------
    # Step 9: add legend
    # --------------------------------------------------------
    axislegend(
        ax,
        handles,
        labels;
        position=position,
        framevisible=true,
        backgroundcolor=(:white, 0.85),
        padding=(8, 8, 8, 8)
    )

    # Tighten layout so figure elements fit nicely
    resize_to_layout!(fig)

    # --------------------------------------------------------
    # Step 10: optionally save the plot
    # --------------------------------------------------------
    if savepath !== nothing && savefilename !== nothing
        mkpath(savepath)
        save(joinpath(savepath, savefilename), fig, px_per_unit=6)
    end

    # --------------------------------------------------------
    # Step 11: optionally display the plot in the current session
    # --------------------------------------------------------
    show && display(fig)

    return fig
end


# ============================================================
# 4. SAVE A PICTURE OF THE RAW DATASET
# ============================================================
#
# This function plots the original problem itself, before graph building.
#
# For classification:
#   - points are split by class
#
# For regression:
#   - points are colored by the response value y
#
# This is useful when running experiments because you can save:
#   - the raw dataset picture
#   - the summary curves later
#
# and inspect both.
function saveProblemPlot(
    X::AbstractMatrix{T},
    y::AbstractVector;
    filepath::String,
    title_str::String="",
    pblm_type::Symbol=:classification,
    figsize::Tuple{Int,Int}=(800, 800),
    show_legend::Bool=false,
    legend_pos::Symbol=:rt,
    markersize::Int=24,
) where {T<:Real}

    # Create a new figure and axis
    fig = Figure(size=figsize)

    ax = Axis(
        fig[1, 1],
        title=isempty(title_str) ? "" : title_str,

        # Hide ticks and tick labels because this figure is mostly
        # for visual pattern inspection, not precise coordinate reading
        xticksvisible=false,
        yticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false,

        # Preserve geometric shape
        aspect=DataAspect()
    )

    # --------------------------------------------------------
    # Classification case
    # --------------------------------------------------------
    if pblm_type == :classification
        classes = sort(unique(y))

        @inbounds for c in classes
            idx = y .== c
            scatter!(ax, X[idx, 1], X[idx, 2], markersize=markersize)
        end

        show_legend && axislegend(ax, position=legend_pos)

        # --------------------------------------------------------
        # Regression case
        # --------------------------------------------------------
    elseif pblm_type == :regression
        scatter!(
            ax,
            X[:, 1],
            X[:, 2],
            color=Float64.(y),
            markersize=markersize
        )

        # Add colorbar so color values are interpretable
        Colorbar(fig[1, 2], ax.scene.plots[end])

    else
        throw(ArgumentError("pblm_type must be :classification or :regression"))
    end

    # Make sure the folder exists
    mkpath(dirname(filepath))

    # Save figure
    save(filepath, fig, px_per_unit=6)

    return fig
end