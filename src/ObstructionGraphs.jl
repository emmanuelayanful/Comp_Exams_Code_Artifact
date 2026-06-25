# ============================================================
# ObstructionGraphs.jl
#
# This is the top-level module file for the project.
#
# Think of it as the "front door" of your codebase.
#
# Its job is NOT to implement the algorithms directly.
# Instead, its job is to:
#
# 1. Create a module namespace
# 2. Load external packages needed by the project
# 3. Pull in the project source files
# 4. Decide which names should be publicly visible
#
# So this file is about project organization, not heavy computation.
# ============================================================

module ObstructionGraphs

# ------------------------------------------------------------
# External package imports
# ------------------------------------------------------------
#
# These are the Julia packages used somewhere in the project.
#
# They become available to all code included inside this module.

# Linear algebra tools:
# norms, symmetric matrices, diagonal matrices, etc.
using LinearAlgebra

# Statistical helpers:
# mean, etc.
using Statistics

# Random number generation:
# used by the dataset generators
using Random

# Threading tools:
# used in core.jl for parallel graph building and prediction
using Base.Threads

# DataFrames:
# used when saving / loading experiment results tables
using DataFrames

# XLSX:
# used to read and write Excel workbooks containing experiment results
using XLSX

# CairoMakie:
# used for plotting datasets and result curves
using CairoMakie

# Distributions:
# used for the "true" PDFs in classification experiments
using Distributions


# ------------------------------------------------------------
# Include the project source files
# ------------------------------------------------------------
#
# include("...") literally reads and evaluates those files
# inside the current module.
#
# So after these lines run, all the functions defined in:
#   datasets.jl
#   core.jl
#   io_viz.jl
# become part of the ObstructionGraphs module.
#
# Order matters:
# - datasets.jl defines data generators and PDFs
# - core.jl defines graph rules, graph building, prediction, experiments
# - io_viz.jl defines reading, plotting, and figure saving
include("datasets.jl")
include("core.jl")
include("io_viz.jl")


# ------------------------------------------------------------
# Export list
# ------------------------------------------------------------
#
# export decides which names become available automatically
# when someone writes:
#
#   using .ObstructionGraphs
#
# or
#
#   using ObstructionGraphs
#
# Exported names are the "public API" of the module.
#
# That means:
# - these are the functions/types you expect users to call directly
# - anything not exported is still available, but the user would need
#   to access it explicitly as:
#       ObstructionGraphs.some_name
#
# The exports are grouped naturally into:
#   - dataset generation
#   - true PDF generation
#   - regression dataset helpers
#   - graph rule functions
#   - graph rule types / workspace types
#   - graph builders and experiment runners
#   - I/O and plotting tools
export gaussian_ellipse,

    # Classification datasets
    make_blobs, make_circles, make_spirals, make_moons,

    # Matching classification PDFs
    make_blobs_pdf, make_circles_pdf,

    # Regression datasets
    sample_points, make_sinesine, make_peaks, make_cosinesine, make_radial_sinc,

    # Obstruction / graph rule helper functions
    gabriel, ellipticGabriel, bow, LoS, doubleCone,

    # Rule and workspace types
    ObstructionRule, EpsilonBall, KNN,
    NbrWorkspace,

    # Core graph tools
    build_sparseGraph_threaded, count_edges,
    makeDefaultMethods,

    # Experiment runners
    runGraphExperiments, runGraphExperimentsWithPDF,

    # I/O and plotting tools
    read_results, plot_results, saveProblemPlot

end