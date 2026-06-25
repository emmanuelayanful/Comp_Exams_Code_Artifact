# ============================================================
# core.jl
#
# Teaching version with detailed comments.
#
# Big picture:
# 1. Define graph rules (KNN, epsilon ball, obstruction-based)
# 2. Build a sparse graph once for a dataset
# 3. Reuse that graph for prediction
# 4. Compute accuracy / regression loss / posterior error
# 5. Loop over many methods and parameters in experiments
# ============================================================

using Base.Threads

# ============================================================
# 1. OBSTRUCTION / NEIGHBORHOOD RULES
# ============================================================
#
# Each rule tells us whether a candidate point should be accepted
# as a neighbor of a host point.
#
# The obstruction-based rules depend on a function g(R), where:
# - R is the normalized position of an already-accepted neighbor
#   along the line from host to candidate
# - g(R) controls the width of the forbidden / obstruction region
#
# These are small scalar helper functions, so @inline is reasonable.

# Standard Gabriel-style width
@inline gabriel(R::T) where {T<:Real} = sqrt(max(zero(T), R * (one(T) - R)))

# Elliptic Gabriel is just a scaled Gabriel function
@inline ellipticGabriel(R::T, ratio::T=0.75) where {T<:Real} = ratio * gabriel(R)

# "Bow" obstruction:
# - one side grows like R * alpha
# - the other limit comes from staying inside the unit circle
@inline bow(R::T, alpha::T=1.0) where {T<:Real} = min(R * alpha, sqrt(max(zero(T), one(T) - R^2)))

# Line-of-sight baseline: zero obstruction width everywhere
@inline LoS(R::T) where {T<:Real} = zero(T)

# Double-cone with separate host-side and candidate-side opening factors
@inline doubleCone(R::T, alpha_h::T=1.0, alpha_c::T=1.0) where {T<:Real} = min(R * alpha_h, (one(T) - R) * alpha_c)

# Symmetric double-cone version
@inline doubleCone(R::T, alpha::T=1.0) where {T<:Real} = min(R * alpha, (one(T) - R) * alpha)

# ObstructionRule stores the function g that defines the obstruction shape.
# Example:
#   ObstructionRule(R -> ellipticGabriel(R, 0.8))
struct ObstructionRule{F}
    g::F
end

# Epsilon-ball rule stores epsilon^2 rather than epsilon itself.
# Why?
# Because most distance checks naturally happen in squared distance,
# so storing eps^2 avoids repeated squaring.
struct EpsilonBall{T<:Real}
    eps2::T
end

# Constructor from epsilon to eps^2
EpsilonBall(epsilon::T) where {T<:Real} = EpsilonBall{T}(epsilon^2)

# KNN rule only needs k
struct KNN
    k::Int
end

# ============================================================
# 2. WORKSPACE
# ============================================================
#
# A workspace holds temporary arrays used while building neighbor lists.
# This is important for performance:
# - we avoid allocating new arrays inside tight loops
# - each thread gets its own workspace
#
# Fields:
# d2      : squared distances from current host/query to every point
# perm    : permutation that sorts points by distance
# nbr_ids : temporary storage of accepted neighbor indices
# dir     : unit direction vector from host to current candidate
mutable struct NbrWorkspace{T<:Real}
    d2::Vector{T}
    perm::Vector{Int}
    nbr_ids::Vector{Int}
    dir::Vector{T}
end

function NbrWorkspace(X::AbstractMatrix{T}) where {T<:Real}
    N, D = size(X)

    return NbrWorkspace{T}(
        Vector{T}(undef, N),   # one distance per point
        collect(1:N),          # indices 1,2,...,N
        Vector{Int}(undef, N), # worst case: all points accepted
        Vector{T}(undef, D)    # direction vector in ambient dimension D
    )
end

# ============================================================
# 3. DISTANCE HELPERS
# ============================================================
#
# These fill a preallocated vector d2 with squared distances.
#
# There are two versions:
# - one for a host point already inside X
# - one for a free query point x
#
# The key idea:
#   squared distance = sum_k (x_i[k] - x_j[k])^2
#
# Using squared distance avoids an unnecessary sqrt until we actually
# need the final Euclidean distance.
@inline function fill_sqdist!(d2::Vector{T}, X::AbstractMatrix{T}, host::Int) where {T<:Real}
    N, D = size(X)

    @inbounds for i in 1:N
        s = zero(T)

        @simd for k in 1:D
            diff = X[i, k] - X[host, k]
            s = muladd(diff, diff, s)  # s += diff^2, using fused multiply-add when available
        end

        d2[i] = s
    end

    return d2
end

@inline function fill_sqdist!(d2::Vector{T}, X::AbstractMatrix{T}, x::AbstractVector{T}) where {T<:Real}
    N, D = size(X)

    @inbounds for i in 1:N
        s = zero(T)

        @simd for k in 1:D
            diff = X[i, k] - x[k]
            s = muladd(diff, diff, s)
        end

        d2[i] = s
    end

    return d2
end

# Sort indices by distance, not the distances themselves.
# After this, perm[1] is the closest point, perm[2] the next closest, etc.
@inline function sort_indices!(perm::Vector{Int}, d2::Vector{T}) where {T<:Real}
    sortperm!(perm, d2)
    return perm
end

# ============================================================
# 4. NEIGHBOR ACCEPTANCE TESTS
# ============================================================
#
# These functions answer:
#   "Should candidate cand be accepted as a neighbor?"
#
# We separate host-based and query-based versions because:
# - host version assumes the host is X[host, :]
# - query version assumes the host is an external vector x_host
#
# The KNN and epsilon-ball rules are simple.
# The obstruction rule is more involved:
#   a candidate is rejected if an already-accepted neighbor lies
#   inside the obstruction region between the host and candidate.

# ---------------- Epsilon Ball ----------------
@inline function _accept_host!(
    rule::EpsilonBall{T},
    X::AbstractMatrix{T},
    host::Int,
    cand::Int,
    nbr_ids::Vector{Int},
    nnbr::Int,
    d2_cand::T,
    dir::Vector{T}
) where {T<:Real}
    return d2_cand > zero(T) && d2_cand < rule.eps2
end

@inline function _accept_query!(
    rule::EpsilonBall{T},
    x_host::AbstractVector{T},
    X::AbstractMatrix{T},
    cand::Int,
    nbr_ids::Vector{Int},
    nnbr::Int,
    d2_cand::T,
    dir::Vector{T}
) where {T<:Real}
    return d2_cand > zero(T) && d2_cand < rule.eps2
end

# ---------------- KNN ----------------
#
# Since candidates are scanned in sorted distance order,
# "accept while nnbr < k" is enough.
@inline function _accept_host!(
    rule::KNN,
    X::AbstractMatrix{T},
    host::Int,
    cand::Int,
    nbr_ids::Vector{Int},
    nnbr::Int,
    d2_cand::T,
    dir::Vector{T}
) where {T<:Real}
    return d2_cand > zero(T) && nnbr < rule.k
end

@inline function _accept_query!(
    rule::KNN,
    x_host::AbstractVector{T},
    X::AbstractMatrix{T},
    cand::Int,
    nbr_ids::Vector{Int},
    nnbr::Int,
    d2_cand::T,
    dir::Vector{T}
) where {T<:Real}
    return d2_cand > zero(T) && nnbr < rule.k
end

# ---------------- Obstruction Rule: host version ----------------
#
# Idea:
# 1. Consider candidate cand for host point host.
# 2. Compute the unit direction from host to candidate.
# 3. For each already-accepted neighbor nbr:
#    - project nbr onto the host->candidate line
#    - compute how far nbr lies off that line
#    - compare that off-line distance to the obstruction width g(R)
# 4. If nbr lies inside the obstruction region, reject cand.
#
# Important:
# - neighbors are scanned in order of increasing distance
# - only already-accepted neighbors can obstruct later candidates
@inline function _accept_host!(
    rule::ObstructionRule{F},
    X::AbstractMatrix{T},
    host::Int,
    cand::Int,
    nbr_ids::Vector{Int},
    nnbr::Int,
    d2_cand::T,
    dir::Vector{T}
) where {T<:Real,F}

    # Do not accept zero-distance points
    d2_cand > zero(T) || return false

    # Actual distance host -> candidate
    R0 = sqrt(d2_cand)
    invR0 = inv(R0)
    D = size(X, 2)

    # dir = unit vector from host to candidate
    @inbounds @simd for k in 1:D
        dir[k] = (X[cand, k] - X[host, k]) * invR0
    end

    # Check all previously accepted neighbors
    @inbounds for t in 1:nnbr
        nbr = nbr_ids[t]

        proj = zero(T)  # projection of nbr onto candidate direction
        v2 = zero(T)  # squared distance host -> nbr

        @simd for k in 1:D
            vk = X[nbr, k] - X[host, k]
            proj = muladd(dir[k], vk, proj)
            v2 = muladd(vk, vk, v2)
        end

        # proj in (0, R0) means neighbor lies somewhere "between"
        # the host and candidate along that direction
        if proj > zero(T) && proj < R0
            R = proj * invR0      # normalize to [0,1]
            gR = rule.g(R)        # obstruction width at normalized location R

            # Allowed squared orthogonal distance from centerline
            thresh2 = d2_cand * gR * gR

            # orth2 = squared perpendicular distance from nbr to the host->cand line
            orth2 = max(zero(T), v2 - proj * proj)

            # If nbr lies inside obstruction tube/cone/etc., reject candidate
            orth2 < thresh2 && return false
        end
    end

    return true
end

# ---------------- Obstruction Rule: query version ----------------
#
# Same logic as above, except the host is an external query vector x_host
@inline function _accept_query!(
    rule::ObstructionRule{F},
    x_host::AbstractVector{T},
    X::AbstractMatrix{T},
    cand::Int,
    nbr_ids::Vector{Int},
    nnbr::Int,
    d2_cand::T,
    dir::Vector{T}
) where {T<:Real,F}

    d2_cand > zero(T) || return false

    R0 = sqrt(d2_cand)
    invR0 = inv(R0)
    D = size(X, 2)

    # dir = unit vector from query point to candidate
    @inbounds @simd for k in 1:D
        dir[k] = (X[cand, k] - x_host[k]) * invR0
    end

    @inbounds for t in 1:nnbr
        nbr = nbr_ids[t]

        proj = zero(T)
        v2 = zero(T)

        @simd for k in 1:D
            vk = X[nbr, k] - x_host[k]
            proj = muladd(dir[k], vk, proj)
            v2 = muladd(vk, vk, v2)
        end

        if proj > zero(T) && proj < R0
            R = proj * invR0
            gR = rule.g(R)
            thresh2 = d2_cand * gR * gR
            orth2 = max(zero(T), v2 - proj * proj)

            orth2 < thresh2 && return false
        end
    end

    return true
end

# ============================================================
# 5. BUILD THE SPARSE GRAPH
# ============================================================
#
# Output:
#   nbr_idx[i]    = indices of neighbors of host i
#   nbr_dist[i]   = distances from host i to those neighbors
#   nearest_idx[i]= nearest non-self point to host i
#
# Why store nearest_idx?
# Because some rules can produce zero accepted neighbors.
# Then prediction still needs a fallback.
#
# Why store both nbr_idx and nbr_dist?
# Because:
# - nbr_idx tells us WHO the neighbors are
# - nbr_dist tells us HOW MUCH weight each neighbor gets
#
# Why threaded?
# Because each host point can be processed independently.
function build_sparseGraph_threaded(
    X::AbstractMatrix{T},
    rule::Union{ObstructionRule,EpsilonBall,KNN}
) where {T<:Real}

    N = size(X, 1)

    # For each host i, store a vector of accepted neighbor indices
    nbr_idx = Vector{Vector{Int}}(undef, N)

    # For each host i, store the matching distances to those neighbors
    nbr_dist = Vector{Vector{T}}(undef, N)

    # For each host i, store the nearest non-self point
    nearest_idx = fill(0, N)

    # One workspace per possible thread id
    # maxthreadid() is safer than nthreads() because thread ids can be offset
    workspaces = [NbrWorkspace(X) for _ in 1:Threads.maxthreadid()]

    Threads.@threads :static for host in 1:N
        ws = workspaces[Threads.threadid()]

        # Distances from host to all points
        fill_sqdist!(ws.d2, X, host)

        # Make sure the point does not select itself
        ws.d2[host] = T(Inf)

        # Sort candidate indices by increasing distance
        sort_indices!(ws.perm, ws.d2)

        # Closest non-self point is stored for fallback prediction later
        nearest_idx[host] = ws.perm[1]

        nnbr = 0

        # Sweep through candidates from nearest to farthest
        @inbounds for p in 1:N
            cand = ws.perm[p]
            cand == host && continue

            d2c = ws.d2[cand]

            if _accept_host!(rule, X, host, cand, ws.nbr_ids, nnbr, d2c, ws.dir)
                nnbr += 1
                ws.nbr_ids[nnbr] = cand
            end
        end

        # Copy accepted neighbors into exactly-sized output vectors
        ids = Vector{Int}(undef, nnbr)
        ds = Vector{T}(undef, nnbr)

        @inbounds for t in 1:nnbr
            j = ws.nbr_ids[t]
            ids[t] = j
            ds[t] = sqrt(ws.d2[j])   # convert squared distance to actual distance
        end

        # Sort by index for stable storage / reproducible ordering
        p = sortperm(ids)
        nbr_idx[host] = ids[p]
        nbr_dist[host] = ds[p]
    end

    return nbr_idx, nbr_dist, nearest_idx
end

# Count edges in the sparse graph representation.
# If directed=true, count all stored directed arcs.
# If directed=false, collapse (i,j) and (j,i) into one undirected edge.
function count_edges(nbr_idx::Vector{Vector{Int}}; directed::Bool=true)
    if directed
        return sum(length(v) for v in nbr_idx)
    else
        seen = Set{Tuple{Int,Int}}()

        @inbounds for i in eachindex(nbr_idx)
            for j in nbr_idx[i]
                i == j && continue
                a, b = minmax(i, j)
                push!(seen, (a, b))
            end
        end

        return length(seen)
    end
end

# ============================================================
# 6. PREDICTION HELPERS
# ============================================================

# Turn a list of labels into class probabilities.
# Example:
#   y_local = [1,1,3,3,3], classes=[1,2,3]
# gives:
#   [2/5, 0/5, 3/5]
function classProbFromLabels(
    y_local::AbstractVector{T},
    classes::AbstractVector{T}
) where {T<:Real}
    counts = [sum(y_local .== c) for c in classes]
    total = sum(counts)

    total == 0 && return fill(inv(length(classes)), length(classes))

    return Float64.(counts) ./ total
end

# Make a one-hot vector in-place:
# everything is zero except the chosen class index
function onehot_index!(out::AbstractVector{Float64}, idx::Int)
    fill!(out, 0.0)
    out[idx] = 1.0
    return out
end

# Map class label -> position in probability vector
# Example:
#   classes = [1,3,7]
# returns Dict(1=>1, 3=>2, 7=>3)
function class_index_map(classes)
    return Dict(c => j for (j, c) in pairs(classes))
end

# ============================================================
# 7. CLASSIFICATION PREDICTION FROM A PREBUILT GRAPH
# ============================================================
#
# This function predicts for the SAME points whose graph we built.
#
# Inputs:
#   nbr_idx[i]     = neighbors of point i
#   nbr_dist[i]    = distances to those neighbors
#   nearest_idx[i] = nearest fallback point for i
#   y              = training labels
#
# Output:
#   vhat    = predicted probability vector for each point
#   yhat    = predicted class label for each point
#   classes = list of class labels
#
# Why do we pass nbr_idx, nbr_dist, nearest_idx instead of X and rule?
# Because the graph has already been built.
# We do not want to recompute neighbors again.
function predictProbaFromGraph_threaded(
    nbr_idx::Vector{Vector{Int}},
    nbr_dist::Vector{Vector{T}},
    nearest_idx::Vector{Int},
    y::AbstractVector{S};
    p::Int=2,
    fallback::Symbol=:nearest
) where {T<:Real,S<:Real}

    N = length(y)

    # Sorted class labels
    classes = sort(unique(y))
    class_to_pos = class_index_map(classes)
    K = length(classes)

    # One row per point, one column per class
    vhat = Matrix{Float64}(undef, N, K)

    # Predicted hard label for each point
    yhat = Vector{S}(undef, N)

    # Used if fallback == :majority
    majority_probs = classProbFromLabels(y, classes)
    default_label = classes[argmax(majority_probs)]

    # One scratch score vector per thread
    scratch = [zeros(Float64, K) for _ in 1:Threads.maxthreadid()]

    Threads.@threads :static for i in 1:N
        scores = scratch[Threads.threadid()]
        fill!(scores, 0.0)

        # Neighbor indices and distances for point i
        ids = nbr_idx[i]
        ds = nbr_dist[i]

        # --------------------------------------------------------
        # Case A: no graph neighbors were found
        # --------------------------------------------------------
        if isempty(ids)
            if fallback == :nearest && nearest_idx[i] != 0
                lbl = y[nearest_idx[i]]

                # One-hot probability on nearest label
                onehot_index!(scores, class_to_pos[lbl])
                @views vhat[i, :] .= scores
                yhat[i] = lbl
            else
                # Use global majority probabilities
                @views vhat[i, :] .= majority_probs
                yhat[i] = default_label
            end

            continue
        end

        # --------------------------------------------------------
        # Case B: one or more neighbor distances are exactly zero
        # --------------------------------------------------------
        #
        # This matters because inverse-distance weights d^(-p)
        # would blow up when d = 0.
        #
        # So if there are exact matches, we only use those.
        has_zero = false
        @inbounds for d in ds
            if d == 0
                has_zero = true
                break
            end
        end

        if has_zero
            # Count exact matches only
            @inbounds for t in eachindex(ids)
                if ds[t] == 0
                    scores[class_to_pos[y[ids[t]]]] += 1.0
                end
            end
        else
            # ----------------------------------------------------
            # Case C: normal weighted voting
            # ----------------------------------------------------
            total_w = 0.0

            @inbounds for t in eachindex(ids)
                w = ds[t]^(-p)   # inverse-distance weight
                scores[class_to_pos[y[ids[t]]]] += w
                total_w += w
            end

            # Normalize to probabilities
            scores ./= total_w
        end

        # Store probability vector
        total = sum(scores)

        if total > 0 && !has_zero
            @views vhat[i, :] .= scores
        else
            # In the zero-distance case, normalize after counting
            @views vhat[i, :] .= scores ./ max(total, 1.0)
        end

        # Predicted class = class with largest score/probability
        yhat[i] = classes[argmax(scores)]
    end

    return vhat, yhat, classes
end

# ============================================================
# 8. REGRESSION PREDICTION FROM A PREBUILT GRAPH
# ============================================================
#
# Same idea as classification, but now we compute a weighted average
# of numeric response values instead of a class probability vector.
function predictValuesFromGraph_threaded(
    nbr_idx::Vector{Vector{Int}},
    nbr_dist::Vector{Vector{T}},
    nearest_idx::Vector{Int},
    y::AbstractVector{S};
    p::Int=2,
    fallback::Symbol=:nearest
) where {T<:Real,S<:Real}

    N = length(y)
    ypred = Vector{Float64}(undef, N)

    # Used for fallback == :majority (here meaning global mean)
    ymean = mean(Float64.(y))

    Threads.@threads :static for i in 1:N
        ids = nbr_idx[i]
        ds = nbr_dist[i]

        # No neighbors -> fallback
        if isempty(ids)
            if fallback == :nearest && nearest_idx[i] != 0
                ypred[i] = y[nearest_idx[i]]
            else
                ypred[i] = ymean
            end
            continue
        end

        # Exact-match protection
        has_zero = false
        @inbounds for d in ds
            if d == 0
                has_zero = true
                break
            end
        end

        if has_zero
            # Average only exact matches
            acc = 0.0
            cnt = 0

            @inbounds for t in eachindex(ids)
                if ds[t] == 0
                    acc += y[ids[t]]
                    cnt += 1
                end
            end

            ypred[i] = acc / cnt
            continue
        end

        # Standard inverse-distance weighted regression
        num = 0.0
        den = 0.0

        @inbounds for t in eachindex(ids)
            w = ds[t]^(-p)
            num += w * y[ids[t]]
            den += w
        end

        ypred[i] = num / den
    end

    return ypred
end

# ============================================================
# 9. METRIC COMPUTATION
# ============================================================
#
# A simple helper so the experiment code can ask for:
# - classification accuracy
# - regression mse, mae, rmse, r2
function computeMetric(
    y_true::AbstractVector,
    y_pred::AbstractVector;
    pblm_type::Symbol=:classification,
    metric::Union{Symbol,Nothing}=nothing
)
    length(y_true) == length(y_pred) ||
        throw(DimensionMismatch("y_true and y_pred must have the same length"))

    metric === nothing &&
        (metric = pblm_type == :classification ? :accuracy : :mse)

    if pblm_type == :classification
        metric == :accuracy ||
            throw(ArgumentError("metric must be :accuracy for classification"))

        return mean(y_true .== y_pred), metric

    elseif pblm_type == :regression
        yt = Float64.(y_true)
        yp = Float64.(y_pred)

        if metric == :mse
            return mean((yt .- yp) .^ 2), metric
        elseif metric == :mae
            return mean(abs.(yt .- yp)), metric
        elseif metric == :rmse
            return sqrt(mean((yt .- yp) .^ 2)), metric
        elseif metric == :r2
            return 1.0 - sum((yt .- yp) .^ 2) / sum((yt .- mean(yt)) .^ 2), metric
        else
            throw(ArgumentError("metric must be :mse, :mae, :rmse, or :r2 for regression"))
        end
    else
        throw(ArgumentError("pblm_type must be :classification or :regression"))
    end
end


# ============================================================
# 10. TRUE POSTERIOR / PDF HELPERS
# ============================================================
#
# These are used in your PDF-based classification experiments.
#
# The idea:
# - vhat = graph-based predicted class probabilities
# - lambda = "true" posterior probabilities from the known class PDFs
# - posteriorError(vhat, lambda) measures how close the graph prediction
#   is to the ideal posterior


# Compute Bayes posterior from known class densities and priors
function posteriorFromPDFs(
    x_query::AbstractVector{T},
    classes::AbstractVector;
    class_pdf::Function,
    priors::Union{Nothing,AbstractVector}=nothing
) where {T<:Real}

    # Use uniform priors if none supplied
    π = priors === nothing ?
        fill(1.0 / length(classes), length(classes)) :
        Float64.(priors)

    numerator = [π[j] * class_pdf(c, x_query) for (j, c) in pairs(classes)]
    denominator = sum(numerator)

    # If densities are degenerate or non-finite, fall back to uniform posterior
    if denominator <= 0 || !isfinite(denominator)
        return fill(1.0 / length(classes), length(classes))
    else
        return numerator ./ denominator
    end
end

# Measure discrepancy between graph posterior and true posterior
function posteriorError(
    vhat::AbstractVector{<:Real},
    lambda::AbstractVector{<:Real};
    metric::Symbol=:l2
)
    length(vhat) == length(lambda) ||
        throw(DimensionMismatch("vhat and lambda must have the same length"))

    if metric == :l1
        return mean(abs.(vhat .- lambda))
    elseif metric == :l2
        return mean((vhat .- lambda) .^ 2)
    else
        throw(ArgumentError("metric must be :l1 or :l2"))
    end
end

# Precompute the true posterior for every point in the dataset.
# This is cheaper than recomputing it for every graph parameter setting.
function precompute_true_posteriors(
    X::AbstractMatrix{T},
    y::AbstractVector;
    class_pdf::Function,
    priors::Union{Nothing,AbstractVector}=nothing
) where {T<:Real}

    N = size(X, 1)
    classes = sort(unique(y))
    K = length(classes)

    lambda = Matrix{Float64}(undef, N, K)
    y_bayes = Vector{eltype(y)}(undef, N)

    Threads.@threads :static for i in 1:N
        probs = posteriorFromPDFs(@view(X[i, :]), classes; class_pdf=class_pdf, priors=priors)
        @views lambda[i, :] .= probs
        y_bayes[i] = classes[argmax(probs)]
    end

    return lambda, y_bayes, classes
end

# Compare graph-based class probabilities against precomputed true posteriors
function scoreGraphPredictionWithPDF_threaded(
    nbr_idx::Vector{Vector{Int}},
    nbr_dist::Vector{Vector{T}},
    nearest_idx::Vector{Int},
    y::AbstractVector,
    lambda_true::AbstractMatrix{<:Real},
    y_bayes::AbstractVector;
    p::Int=2,
    fallback::Symbol=:nearest,
    metric::Symbol=:l2
) where {T<:Real}

    N = length(y)

    # Graph-based probability predictions
    vhat, yhat, _ = predictProbaFromGraph_threaded(
        nbr_idx, nbr_dist, nearest_idx, y; p=p, fallback=fallback
    )

    errs = Vector{Float64}(undef, N)

    Threads.@threads :static for i in 1:N
        errs[i] = posteriorError(@view(vhat[i, :]), @view(lambda_true[i, :]); metric=metric)
    end

    return (
        accuracy=mean(yhat .== y),
        bayes_accuracy=mean(yhat .== y_bayes),
        error=mean(errs)
    )
end

# ============================================================
# 11. METHOD FACTORY
# ============================================================
#
# This builds the list of method families and parameter grids
# used in experiment sweeps.
function makeDefaultMethods(;
    ratios=collect(range(0.05, sqrt(2), length=10)),
    alphas=collect(range(0.01, sqrt(3), length=10)),
    ks=collect(2 .^ (1:10) .- 1),
    epsilons=collect(range(0.01, 2.0, length=10))
)
    return [
        (
            name="Elliptic Gabriel",
            params=ratios,
            rule=r -> ObstructionRule(R -> ellipticGabriel(R, r))
        ),
        (
            name="Double Cone",
            params=alphas,
            rule=a -> ObstructionRule(R -> doubleCone(R, a))
        ),
        (
            name="Epsilon Ball",
            params=epsilons,
            rule=epsilon -> EpsilonBall(epsilon)
        ),
        (
            name="KNN",
            params=ks,
            rule=k -> KNN(Int(k))
        )
    ]
end

# ============================================================
# 12. EXPERIMENT DRIVER: STANDARD METRICS
# ============================================================
#
# Workflow for each problem:
# 1. Generate dataset X, y
# 2. Optionally save a picture of the problem
# 3. For each method and parameter:
#    a. Build sparse graph once
#    b. Count its edges
#    c. Predict from the graph
#    d. Score prediction
# 4. Save results to workbook
# 5. Optionally make a summary plot
function runGraphExperiments(
    problems::Vector;
    methods::Vector,
    pblm_type::Symbol=:classification,
    workbook_path::Union{String,Nothing}=nothing,
    figs_dir::Union{String,Nothing}=nothing,
    metric::Union{Symbol,Nothing}=nothing,
    p::Int=2,
    fallback::Symbol=:nearest,
    save_problem_plots::Bool=true,
    summary_plotter::Union{Function,Nothing}=nothing,
    directed::Bool=false
)
    pblm_type in (:classification, :regression) ||
        throw(ArgumentError("pblm_type must be :classification or :regression"))

    if workbook_path !== nothing
        mkpath(dirname(workbook_path))
        isfile(workbook_path) && rm(workbook_path; force=true)
    end

    figs_dir !== nothing && mkpath(figs_dir)

    first_sheet = true

    @inbounds for prob in problems
        name = prob.name
        X, y = prob.make_data()

        # Save a picture of the raw problem if requested
        if save_problem_plots && figs_dir !== nothing
            saveProblemPlot(X, y; filepath=joinpath(figs_dir, "$(name).png"), pblm_type=pblm_type)
        end

        metric_name = something(metric, pblm_type == :classification ? :accuracy : :rmse)

        results = DataFrame(method=String[], param=Float64[], nedges=Int[])
        results[!, metric_name] = Float64[]

        for method in methods
            for param in method.params
                rule = method.rule(param)

                # Build graph once
                nbr_idx, nbr_dist, nearest_idx = build_sparseGraph_threaded(X, rule)

                # Count edges for structural comparison
                nedges = count_edges(nbr_idx; directed=directed)

                # Predict using the prebuilt graph
                if pblm_type == :classification
                    _, y_pred, _ = predictProbaFromGraph_threaded(
                        nbr_idx, nbr_dist, nearest_idx, y; p=p, fallback=fallback
                    )
                else
                    y_pred = predictValuesFromGraph_threaded(
                        nbr_idx, nbr_dist, nearest_idx, y; p=p, fallback=fallback
                    )
                end

                # Score and store
                score, _ = computeMetric(y, y_pred; pblm_type=pblm_type, metric=metric_name)
                push!(results, (String(method.name), Float64(param), nedges, score))
            end
        end

        # Save sheet
        if workbook_path !== nothing
            if first_sheet
                XLSX.writetable(workbook_path, overwrite=true, name => results)
                first_sheet = false
            else
                XLSX.openxlsx(workbook_path, mode="rw") do xf
                    sheet = XLSX.addsheet!(xf, name)
                    XLSX.writetable!(sheet, results)
                end
            end
        end

        # Optional summary figure after sheet is written
        if summary_plotter !== nothing && figs_dir !== nothing && workbook_path !== nothing
            summary_plotter(name, workbook_path, figs_dir, metric_name)
        end
    end

    return nothing
end

# ============================================================
# 13. EXPERIMENT DRIVER: PDF / POSTERIOR METRICS
# ============================================================
#
# Same idea as above, except now each problem also provides:
# - classpdf
# - priors
#
# and the scoring compares graph probabilities against Bayes posteriors.
function runGraphExperimentsWithPDF(
    problems::Vector;
    methods::Vector,
    workbook_path::Union{String,Nothing}=nothing,
    figs_dir::Union{String,Nothing}=nothing,
    metric::Symbol=:l2,
    p::Int=2,
    fallback::Symbol=:nearest,
    save_problem_plots::Bool=true,
    summary_plotter::Union{Function,Nothing}=nothing,
    directed::Bool=false
)
    metric in (:l1, :l2) || throw(ArgumentError("metric must be :l1 or :l2"))

    if workbook_path !== nothing
        mkpath(dirname(workbook_path))
        isfile(workbook_path) && rm(workbook_path; force=true)
    end

    figs_dir !== nothing && mkpath(figs_dir)

    first_sheet = true
    posterior_col = Symbol(metric)

    @inbounds for prob in problems
        name = prob.name
        X, y = prob.make_data()
        _, classpdf, priors = prob.make_pdf()

        if save_problem_plots && figs_dir !== nothing
            saveProblemPlot(X, y; filepath=joinpath(figs_dir, "$(name).png"), pblm_type=:classification)
        end

        # Precompute the target posterior once for this dataset
        lambda_true, y_bayes, _ = precompute_true_posteriors(
            X, y; class_pdf=classpdf, priors=priors
        )

        results = DataFrame(
            method=String[],
            param=Float64[],
            nedges=Int[],
            accuracy=Float64[],
            accuracy_error=Float64[]
        )

        results[!, posterior_col] = Float64[]

        for method in methods
            for param in method.params
                rule = method.rule(param)

                # Build graph once
                nbr_idx, nbr_dist, nearest_idx = build_sparseGraph_threaded(X, rule)
                nedges = count_edges(nbr_idx; directed=directed)

                # Compare graph-based probability predictions to Bayes posterior
                out = scoreGraphPredictionWithPDF_threaded(
                    nbr_idx, nbr_dist, nearest_idx, y, lambda_true, y_bayes;
                    p=p, fallback=fallback, metric=metric
                )

                push!(results, (
                    String(method.name),
                    Float64(param),
                    nedges,
                    out.accuracy,
                    1 - out.accuracy,
                    out.error
                ))
            end
        end

        # Save sheet
        if workbook_path !== nothing
            if first_sheet
                XLSX.writetable(workbook_path, overwrite=true, name => results)
                first_sheet = false
            else
                XLSX.openxlsx(workbook_path, mode="rw") do xf
                    sheet = XLSX.addsheet!(xf, name)
                    XLSX.writetable!(sheet, results)
                end
            end
        end

        # Optional summary figure
        if summary_plotter !== nothing && figs_dir !== nothing && workbook_path !== nothing
            summary_plotter(name, workbook_path, figs_dir, posterior_col)
        end
    end

    return nothing
end