# ============================================================
# run_blobs.jl
#
# This is the top-level experiment script.
#
# Think of it as the "director" of the experiment.
#
# It does not define the graph rules or datasets itself.
# Instead, it calls the functions already defined in:
#   - ObstructionGraphs.jl
#   - datasets.jl
#   - core.jl
#   - io_viz.jl
#
# Main job of this script:
#   1. Load the project module
#   2. Define the parameter grids to test
#   3. Define the blob classification problems
#   4. Run the experiments
#   5. Save workbook + figures
# ============================================================


# ------------------------------------------------------------
# Load the project module
# ------------------------------------------------------------
#
# @__DIR__ is the directory of THIS script.
#
# If this file is in:
#   myproject/test/run_blobs.jl
#
# then:
#   joinpath(@__DIR__, "..", "src", "ObstructionGraphs.jl")
#
# points to:
#   myproject/src/ObstructionGraphs.jl
#
# include(...) reads that file and evaluates it here.
# using .ObstructionGraphs then brings the exported names
# from that module into scope.
include(joinpath(@__DIR__, "..", "src", "ObstructionGraphs.jl"))
using .ObstructionGraphs
using Random


# ------------------------------------------------------------
# Main experiment function
# ------------------------------------------------------------
#
# Putting everything inside main() is good practice because:
#   - it keeps code out of global scope
#   - it is faster in Julia
#   - it makes the script easier to rerun and profile
function main()

    # ========================================================
    # 1. DEFINE PARAMETER GRIDS
    # ========================================================
    #
    # These are the parameter values that will be tested
    # for the different graph families.
    #
    # temp:
    #   small values from just above 0 up to just below 0.5
    #
    # ratios_tmp:
    #   values for Elliptic Gabriel ratio parameter
    #
    # alphas_tmp:
    #   values for Double Cone alpha parameter
    #
    # eps_tmp:
    #   values for Epsilon Ball radius
    #
    # k_vals:
    #   values for KNN
    #
    # Why split temp and the other grids?
    # Because later you concatenate them with vcat(...)
    # to create a richer sweep of small-to-large parameter values.
    temp = collect(range(0.001, 0.5 - 0.001, length=10))
    ratios_tmp = collect(range(0.5, sqrt(2), length=10))
    alphas_tmp = collect(range(0.5, sqrt(3), length=10))
    eps_tmp = collect(range(0.5, 2, length=10))

    # Candidate k values for KNN
    k_vals = [1, 3, 5, 7, 9, 11, 13, 15, 17, 23, 31, 43, 59, 81, 111, 151, 207, 283, 535, 1023]


    # ========================================================
    # 2. DEFINE THE BLOB CENTERS
    # ========================================================
    #
    # This is the 3 x 2 matrix of class centers used by make_blobs.
    #
    # Each row is one class center:
    #   class 1 -> [-0.65,  0.50]
    #   class 2 -> [ 0.65,  0.60]
    #   class 3 -> [ 0.00, -0.55]
    #
    # Then make_blobs will scale these inward depending on the level.
    centers = [
        -0.65 0.50
        0.65 0.60
        0.0 -0.55
    ]


    # ========================================================
    # 3. MAP INTEGER LEVELS TO HUMAN-READABLE NAMES
    # ========================================================
    #
    # The dataset generator uses numeric levels 1,2,3,4,
    # but for workbook sheet names and figure names,
    # it is nicer to use text labels.
    #
    # So:
    #   1 -> "disjoint"
    #   2 -> "close"
    #   3 -> "touch"
    #   4 -> "overlap"
    level_map = Dict(
        1 => "disjoint",
        2 => "close",
        3 => "touch",
        4 => "overlap"
    )

    base_seed = 1234

    # ========================================================
    # 4. BUILD THE LIST OF PROBLEMS
    # ========================================================
    #
    # Each entry in problems is a NamedTuple with:
    #   name      = name of the problem / sheet
    #   make_data = function that generates X, y
    #   make_pdf  = function that returns classes, classpdf, priors
    #
    # Why store functions instead of raw data directly?
    # Because the experiment driver expects each problem to know
    # how to generate its own dataset and matching PDF model.
    #
    # This makes the experiment framework flexible:
    # you can swap in circles, moons, spirals, etc. using
    # the same structure.
    #
    # For each level:
    #   - name is "disjoint", "close", "touch", or "overlap"
    #   - make_data generates the sampled blob dataset
    #   - make_pdf generates the corresponding true density model
    problems = [
        (
            name=level_map[level],

            # Dataset generator closure:
            # when called, it generates X, y for this level
            make_data=() -> make_blobs(
                centers,
                npts=500,
                level=level,
                rng=MersenneTwister(base_seed + level)
            ),

            # Matching "ground truth" PDF model for posterior experiments
            make_pdf=() -> make_blobs_pdf(
                centers=centers,
                level=level
            )
        )
        for level in 1:4
    ]


    # ========================================================
    # 5. BUILD THE METHOD LIST
    # ========================================================
    #
    # makeDefaultMethods returns a vector of method families:
    #   - Elliptic Gabriel
    #   - Double Cone
    #   - Epsilon Ball
    #   - KNN
    #
    # Each method family contains:
    #   name   = method name
    #   params = vector of parameters to try
    #   rule   = function turning one parameter into a graph rule
    #
    # Here you are specifying the parameter grids for each family.
    methods = makeDefaultMethods(
        ratios=vcat(temp, ratios_tmp),
        alphas=vcat(temp, alphas_tmp),
        ks=k_vals,
        epsilons=vcat(temp, eps_tmp)
    )


    # ========================================================
    # 6. DEFINE OUTPUT PATHS
    # ========================================================
    #
    # project_root points to the top-level project directory.
    #
    # If this script is in:
    #   myproject/test/run_blobs.jl
    #
    # then:
    #   normpath(joinpath(@__DIR__, ".."))
    #
    # gives:
    #   myproject/
    project_root = normpath(joinpath(@__DIR__, ".."))

    # Folder for workbook output
    data_path = mkpath(joinpath(project_root, "saves", "data", "classification", "blobs"))

    # Folder for figure output
    fig_path = mkpath(joinpath(project_root, "saves", "figs", "classification", "blobs"))

    # Full path to workbook file
    workbook_path = joinpath(data_path, "blobs.xlsx")


    # ========================================================
    # 7. RUN THE PDF-BASED EXPERIMENTS
    # ========================================================
    #
    # posterior_metric = :l1 means:
    #   compare graph probability vector vhat to true posterior lambda
    #   using mean absolute error
    #
    # You could also use :l2 for mean squared error in probability space.
    posterior_metric = :l1

    runGraphExperimentsWithPDF(
        problems;

        # Methods to evaluate
        methods=methods,

        # Save all numeric results into this Excel workbook
        workbook_path=workbook_path,

        # Save figures into this directory
        figs_dir=fig_path,

        # Which posterior error metric to use
        metric=posterior_metric,

        # Power in inverse-distance weighting:
        # weight = d^(-p)
        p=2,

        # If a point gets no neighbors, use the nearest point
        fallback=:nearest,

        # Save pictures of the raw datasets too
        save_problem_plots=true,

        # Treat graph as undirected when counting edges
        directed=false,

        # summary_plotter:
        # This is a callback function.
        #
        # After each sheet is written, the experiment driver calls this
        # function so you can immediately make a plot from that sheet.
        #
        # Inputs:
        #   sheet_name    = e.g. "disjoint"
        #   workbook_path = path to blobs.xlsx
        #   fig_dir       = where plots should be saved
        #   metric_sym    = which metric column to plot
        #
        # Here it plots the posterior metric (l1) vs number of edges.
        summary_plotter=(sheet_name, workbook_path, fig_dir, metric_sym) -> begin
            plot_results(
                filepath=dirname(workbook_path),
                filename=basename(workbook_path),
                sheetname=sheet_name,
                savepath=fig_dir,
                savefilename="$(posterior_metric)_blobs_$(sheet_name).png",
                show=false,
                metric=metric_sym
            )
        end
    )


    # ========================================================
    # 8. MAKE ADDITIONAL PLOTS FROM THE SAVED WORKBOOK
    # ========================================================
    #
    # The PDF experiment already saved the workbook.
    # Now you loop back over the sheets and create more summary plots
    # for other columns in the workbook.
    #
    # These metrics are:
    #   :accuracy
    #   :accuracy_error
    #
    # For each metric and each problem level,
    # you call plot_results and save a PNG figure.
    for metric in (:accuracy, :accuracy_error)
        for level in 1:4
            sheet_name = level_map[level]

            plot_results(
                filepath=dirname(workbook_path),
                filename=basename(workbook_path),
                sheetname=sheet_name,
                savepath=fig_path,
                savefilename="$(metric)_blobs_$(sheet_name).png",
                show=false,
                metric=metric
            )
        end
    end


    # ========================================================
    # 9. PRINT OUTPUT LOCATIONS
    # ========================================================
    #
    # This makes it easy to see where the workbook and figures ended up.
    println("Done.")
    println("Workbook: ", workbook_path)
    println("Figures:  ", fig_path)
end


# ============================================================
# 10. RUN AND TIME THE WHOLE SCRIPT
# ============================================================
#
# @time runs main() and prints:
#   - elapsed time
#   - memory allocations
#
# This is useful for quick profiling during development.
@time main()