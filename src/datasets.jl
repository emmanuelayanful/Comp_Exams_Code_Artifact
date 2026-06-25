# ============================================================
# datasets.jl
#
# This file is the "problem factory" for the project.
#
# It creates the synthetic datasets that your graph methods
# will be tested on.
#
# There are three main parts:
#
# 1. Classification dataset generators
#    - blobs
#    - circles
#    - spirals
#    - moons
#
# 2. Matching classification PDF generators
#    - these define the "true" class densities used when
#      comparing graph-based probabilities to Bayes posteriors
#
# 3. Regression dataset generators
#    - smooth functions sampled on points in [-1,1]^2
#
# Output convention:
#   Every generator returns:
#       X, y
#   where
#       X = n x 2 matrix of input points
#       y = response vector
#
# For classification:
#       y contains class labels like 1, 2, 3
#
# For regression:
#       y contains real-valued function outputs
# ============================================================


# ============================================================
# 1. CLASSIFICATION DATASET GENERATION
# ============================================================


# ------------------------------------------------------------
# gaussian_ellipse
# ------------------------------------------------------------
#
# Purpose:
#   Generate a cloud of 2D Gaussian points shaped like an ellipse.
#
# Inputs:
#   center = location of ellipse center, e.g. [0.5, -0.2]
#   scales = standard deviations along the ellipse axes
#   theta  = rotation angle of the ellipse
#   npts   = number of sampled points
#   rng    = random number generator
#
# Idea:
#   Start with standard Gaussian points in 2D
#   -> stretch them by scales
#   -> rotate them by theta
#   -> shift them to the chosen center
#
# Output:
#   X = npts x 2 matrix
#
# Why permutedims at the end?
#   randn(rng, 2, npts) gives a 2 x npts matrix
#   but the rest of your project expects one point per row,
#   so we transpose/permutedims it into npts x 2
function gaussian_ellipse(
    center::AbstractVector,
    scales::AbstractVector;
    theta::AbstractFloat=0.0,
    npts::Int=100,
    rng=Random.default_rng()
)
    # Rotation matrix
    R = [cos(theta) -sin(theta);
        sin(theta) cos(theta)]

    # Generate Gaussian points, stretch, rotate, then shift to center
    X = reshape(center, 2, 1) .+
        R * Diagonal(scales) * randn(rng, 2, npts)

    # Return one point per row
    return permutedims(X)
end


# ------------------------------------------------------------
# make_blobs
# ------------------------------------------------------------
#
# Purpose:
#   Create a 3-class Gaussian blob classification problem.
#
# Inputs:
#   centers = a 3 x 2 matrix of blob centers
#   npts    = number of points PER class
#   level   = difficulty / overlap level
#
# How level works:
#   The "width" of each blob is fixed here,
#   but the centers are scaled toward the origin.
#
#   level 1 -> centers stay far apart
#   level 4 -> centers move much closer together
#
# So the classes become harder to separate as level increases.
#
# Output:
#   X = (3*npts) x 2 matrix
#   y = labels [1,1,...,2,2,...,3,3,...]
function make_blobs(
    centers::AbstractMatrix;
    npts::Int=100,
    level::Int=1,
    rng=Random.default_rng()
)
    # Blob widths by level.
    # In this version they are all the same, but keeping them
    # in a Dict makes it easy to vary later if you want.
    widths = Dict(
        1 => 0.70,
        2 => 0.70,
        3 => 0.70,
        4 => 0.70,
    )

    # Scale factor applied to the centers.
    # Smaller scale = centers move closer to (0,0)
    center_scales = Dict(
        1 => 1.0,
        2 => 0.7,
        3 => 0.5,
        4 => 0.3
    )

    # Rotation angle for each blob
    thetas = [0.25, 0.25, 0.25]

    width = get(widths, level, 0.70)
    scale = get(center_scales, level, 1.0)

    # Ellipse axis lengths for each class
    w1 = [0.17 * width, 0.20 * width]
    w2 = [0.17 * width, 0.20 * width]
    w3 = [0.17 * width, 0.20 * width]

    # Generate one blob per class
    X1 = gaussian_ellipse(centers[1, :] .* scale, w1, theta=thetas[1]; npts=npts, rng=rng)
    X2 = gaussian_ellipse(centers[2, :] .* scale, w2, theta=thetas[2]; npts=npts, rng=rng)
    X3 = gaussian_ellipse(centers[3, :] .* scale, w3, theta=thetas[3]; npts=npts, rng=rng)

    # Stack all class points vertically
    X = vcat(X1, X2, X3)

    # Labels: class 1 for X1, class 2 for X2, class 3 for X3
    y = vcat(fill(1, npts), fill(2, npts), fill(3, npts))

    return X, y
end


# ------------------------------------------------------------
# make_circles
# ------------------------------------------------------------
#
# Purpose:
#   Create a 2-class concentric-circle classification problem.
#
# Inputs:
#   npts  = number of points per class
#   level = controls how separated the circles are
#
# Idea:
#   - Class 1 lives on an inner noisy circle
#   - Class 2 lives on an outer noisy circle
#
# As the inner radius changes, the problem becomes easier/harder.
#
# Output:
#   X = (2*npts) x 2 matrix
#   y = class labels
function make_circles(;
    npts::Int=100,
    level::Int=1,
    rng=Random.default_rng()
)
    # Radius of the inner circle by level
    inner_radii = Dict(
        1 => 1.20,
        2 => 1.80,
        3 => 2.50,
        4 => 2.75
    )

    # Random angles around the circle
    inner = 2π .* rand(rng, npts)
    outer = 2π .* rand(rng, npts)

    R_inner = get(inner_radii, level, 1.20)
    R_outer = 3.00

    # Add radial noise
    r_inner = R_inner .+ 0.15 .* randn(rng, npts)
    r_outer = R_outer .+ 0.15 .* randn(rng, npts)

    # Convert polar -> Cartesian
    X1 = [r_inner .* cos.(inner) r_inner .* sin.(inner)]
    X2 = [r_outer .* cos.(outer) r_outer .* sin.(outer)]

    X = vcat(X1, X2)
    y = vcat(fill(1, npts), fill(2, npts))

    return X, y
end


# ------------------------------------------------------------
# make_spirals
# ------------------------------------------------------------
#
# Purpose:
#   Create a 2-class spiral problem.
#
# Inputs:
#   level controls the amount of Gaussian noise
#
# Idea:
#   - Create one spiral
#   - Create another spiral rotated by π
#   - Add noise
#
# Why normalize at the end?
#   To keep the entire dataset on a consistent scale.
function make_spirals(;
    npts::Int=100,
    level::Int=1,
    rng=Random.default_rng()
)
    widths = Dict(
        1 => 0.03,
        2 => 0.05,
        3 => 0.08,
        4 => 0.12
    )

    width = get(widths, level, 0.03)

    # Spiral parameter
    t = range(0.4, 4.2π; length=npts)
    r = 0.1 .* t

    # Two opposite spirals
    X1 = [r .* cos.(t) r .* sin.(t)]
    X2 = [r .* cos.(t .+ π) r .* sin.(t .+ π)]

    # Add noise
    X1 .+= width .* randn(rng, npts, 2)
    X2 .+= width .* randn(rng, npts, 2)

    X = vcat(X1, X2)

    # Normalize so the largest absolute coordinate becomes 1
    X ./= maximum(abs.(X))

    y = vcat(fill(1, npts), fill(2, npts))

    return X, y
end


# ------------------------------------------------------------
# make_moons
# ------------------------------------------------------------
#
# Purpose:
#   Create the classic 2-class two-moons problem.
#
# Inputs:
#   level controls noise width
#
# Idea:
#   - First moon: upper arc
#   - Second moon: shifted lower arc
#   - Add noise
#   - Center and normalize
function make_moons(;
    npts::Int=100,
    level::Int=1,
    rng=Random.default_rng()
)
    widths = Dict(
        1 => 0.03,
        2 => 0.10,
        3 => 0.18,
        4 => 0.30
    )

    width = get(widths, level, 0.03)

    # Random angles on [0, π]
    t1 = π .* rand(rng, npts)
    t2 = π .* rand(rng, npts)

    # First moon
    X1 = [cos.(t1) sin.(t1)]

    # Second moon shifted downward/rightward
    X2 = [1 .- cos.(t2) 0.45 .- sin.(t2)]

    # Add noise
    X1 .+= width .* randn(rng, npts, 2)
    X2 .+= width .* randn(rng, npts, 2)

    X = vcat(X1, X2)

    # Center the dataset
    X[:, 1] .-= mean(X[:, 1])
    X[:, 2] .-= mean(X[:, 2])

    # Normalize scale
    X ./= maximum(abs.(X))

    y = vcat(fill(1, npts), fill(2, npts))

    return X, y
end


# ============================================================
# 2. CLASSIFICATION PDF GENERATORS
# ============================================================
#
# These functions do NOT generate sample points directly.
#
# Instead, they return the "true" probabilistic model behind
# the classification problem:
#
#   classes, classpdf, priors
#
# where:
#   classes = class labels
#   classpdf(c, x) = density of class c at point x
#   priors = prior class probabilities
#
# These are used in your posterior-error experiments:
#   graph probabilities vs true Bayes posterior


# ------------------------------------------------------------
# make_blobs_pdf
# ------------------------------------------------------------
#
# Purpose:
#   Return the Gaussian mixture model underlying make_blobs.
#
# Important:
#   This should match the geometry used in make_blobs as closely as possible,
#   because this is your "ground truth" class-density model.
function make_blobs_pdf(; centers::AbstractMatrix{<:Real}, level::Int=1)
    widths = Dict(
        1 => 0.70,
        2 => 0.70,
        3 => 0.70,
        4 => 0.70,
    )

    center_scales = Dict(
        1 => 1.0,
        2 => 0.7,
        3 => 0.5,
        4 => 0.3
    )

    thetas = [0.25, 0.25, 0.25]

    width = get(widths, level, 0.70)
    scale = get(center_scales, level, 1.0)

    # Same ellipse widths used in make_blobs
    w = [0.17 * width, 0.20 * width]

    classes = collect(1:size(centers, 1))

    # Store one multivariate normal per class
    mvns = Dict{Int,MvNormal}()

    for c in classes
        # Mean of class c
        mu = Float64.(centers[c, :] .* scale)

        # Rotation angle of class c ellipse
        theta = thetas[c]

        # Rotation matrix
        R = [cos(theta) -sin(theta);
            sin(theta) cos(theta)]

        # Covariance matrix of rotated ellipse
        sigma = Symmetric(R * Diagonal(w .^ 2) * R')

        mvns[c] = MvNormal(mu, sigma)
    end

    # class-conditional density function
    classpdf(c, x) = pdf(mvns[c], x)

    # Uniform priors across classes
    priors = fill(1 / length(classes), length(classes))

    return classes, classpdf, priors
end


# ------------------------------------------------------------
# make_circles_pdf
# ------------------------------------------------------------
#
# Purpose:
#   Return a radial-density model for the circles problem.
#
# Idea:
#   For each class, radius is modeled as a 1D Normal distribution.
#
# Since the data lives in 2D, the density at x depends on norm(x),
# and the function divides by approximately 2πr to convert a radial
# density into a planar density.
function make_circles_pdf(; level::Int=1)
    inner_radii = Dict(
        1 => 1.20,
        2 => 1.80,
        3 => 2.50,
        4 => 2.75
    )

    R_inner = get(inner_radii, level, 1.20)
    R_outer = 3.00
    sigma_r = 0.15

    # One radial Normal per class
    normals = Dict(
        1 => Normal(R_inner, sigma_r),
        2 => Normal(R_outer, sigma_r)
    )

    classes = [1, 2]

    # Density at x depends only on radius norm(x)
    classpdf(c, x) = pdf(normals[c], norm(x)) / max(2π * norm(x), 1e-12)

    priors = fill(0.5, 2)

    return classes, classpdf, priors
end


# ============================================================
# 3. REGRESSION DATASET GENERATION
# ============================================================
#
# These functions generate points in [-1,1]^2 and evaluate
# a smooth target function on them.
#
# The main helper is sample_points, which determines where
# input locations come from.


# ------------------------------------------------------------
# clamp_to_square!
# ------------------------------------------------------------
#
# Purpose:
#   Force all coordinates into the square [lo, hi]^2
#
# Why needed?
#   Clustered Gaussian sampling can produce values outside [-1,1],
#   so we clamp them back into the intended domain.
#
# The ! means this modifies X in-place.
function clamp_to_square!(
    X::AbstractMatrix; lo::AbstractFloat=-1.0, hi::AbstractFloat=1.0
)
    X[:, 1] .= clamp.(X[:, 1], lo, hi)
    X[:, 2] .= clamp.(X[:, 2], lo, hi)

    return X
end


# ------------------------------------------------------------
# sample_points
# ------------------------------------------------------------
#
# Purpose:
#   Generate input points for regression experiments.
#
# Inputs:
#   density can be:
#     "uniform"   -> spread evenly in the square
#     "mild"      -> clustered at four corners, loosely
#     "moderate"  -> more clustered
#     "dense"     -> very tightly clustered
#
# Why this matters:
#   It lets you test how graph methods behave under different
#   sampling patterns, not just different target functions.
function sample_points(;
    npts::Int=100,
    density::String="uniform",
    rng=Random.default_rng()
)
    # Uniform sampling in [-1,1]^2
    if density == "uniform"
        return 2 .* rand(rng, npts, 2) .- 1
    end

    # Cluster width by density type
    scale_map = Dict(
        "mild" => 0.35,
        "moderate" => 0.22,
        "dense" => 0.12
    )

    # Four cluster centers near the corners
    centers = [
        -0.65 -0.65;
        -0.65 0.65;
        0.65 -0.65;
        0.65 0.65
    ]

    scale = get(scale_map, density, 0.35)

    # Number of clusters
    m = size(centers, 1)

    # Split npts roughly evenly across the clusters
    counts = fill(npts ÷ m, m)
    for i in 1:(npts-sum(counts))
        counts[i] += 1
    end

    X = Matrix{Float64}(undef, npts, 2)

    idx = 1
    for j in axes(centers, 1)
        # Sample Gaussian cluster around center j
        X[idx:idx+counts[j]-1, :] .=
            scale .* randn(rng, counts[j], 2) .+ centers[j, :]'
        idx += counts[j]
    end

    # Ensure all points remain inside the square
    return clamp_to_square!(X)
end


# ------------------------------------------------------------
# make_sinesine
# ------------------------------------------------------------
#
# Target function:
#   y = sin(2π x1) * sin(2π x2)
#
# This is a smooth oscillatory surface.
function make_sinesine(; npts::Int=100, density::String="uniform", rng=Random.default_rng())
    X = sample_points(npts=npts, density=density, rng=rng)
    y = sin.(2π * X[:, 1]) .* sin.(2π * X[:, 2])
    return X, y
end


# ------------------------------------------------------------
# make_peaks
# ------------------------------------------------------------
#
# Target function:
#   a standard multi-peak smooth benchmark surface
#
# This one has several hills and valleys and is harder than
# simple sinusoidal surfaces.
function make_peaks(; npts::Int=100, density::String="uniform", rng=Random.default_rng())
    X = sample_points(npts=npts, density=density, rng=rng)

    y = 3 * (1 .- X[:, 1]) .^ 2 .* exp.(-X[:, 1] .^ 2 .- (X[:, 2] .+ 1) .^ 2) .-
        10 * (X[:, 1] ./ 5 .- X[:, 1] .^ 3 .- X[:, 2] .^ 5) .* exp.(-X[:, 1] .^ 2 .- X[:, 2] .^ 2) .-
        (1 / 3) * exp.(-(X[:, 1] .+ 1) .^ 2 .- X[:, 2] .^ 2)

    return X, y
end


# ------------------------------------------------------------
# make_cosinesine
# ------------------------------------------------------------
#
# Another smooth oscillatory benchmark with mixed polynomial and
# trigonometric structure.
function make_cosinesine(; npts::Int=100, density::String="uniform", rng=Random.default_rng())
    X = sample_points(npts=npts, density=density, rng=rng)

    y = X[:, 1] .* (1 .- X[:, 1]) .* cos.(4π .* X[:, 1]) .* (sin.(4π .* X[:, 2] .^ 2)) .^ 2

    return X, y
end


# ------------------------------------------------------------
# make_radial_sinc
# ------------------------------------------------------------
#
# Radially symmetric target function:
#   y = sin(6π r)/(6π r)
#
# where r = distance from origin.
#
# Important detail:
#   At r = 0, the expression sin(z)/z should be interpreted
#   as 1, so the code handles that special case explicitly.
function make_radial_sinc(; npts::Int=100, density::String="uniform", rng=Random.default_rng())
    X = sample_points(npts=npts, density=density, rng=rng)

    r = sqrt.(X[:, 1] .^ 2 .+ X[:, 2] .^ 2)
    z = π * 6 .* r

    y = similar(z)

    @inbounds for i in eachindex(z)
        y[i] = iszero(z[i]) ? 1.0 : sin(z[i]) / z[i]
    end

    return X, y
end